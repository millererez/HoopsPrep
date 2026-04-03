"""
nodes/data_specialist.py
────────────────────────
Node 1 — Data Specialist (100% ESPN API, zero LLM, zero Tavily).
Fetches standings, player stats, and H2H game history.
"""

import requests
from datetime import datetime

from core.state import GraphState, ESPN_SEASON_YEAR, EST, ESPN_TEAMS, extract_teams, ordinal

# ---------------------------------------------------------------------------
# ESPN API endpoints
# ---------------------------------------------------------------------------

_ESPN_STATS_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"
    "/statistics/byathlete?region=us&lang=en&contentorigin=espn"
    "&isqualified=true&sort=offensive.avgPoints%3Adesc&limit=500"
)
_ESPN_STANDINGS_URL = (
    "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _espn_fetch(url: str) -> dict:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _build_standings_lookup() -> dict[str, dict]:
    """
    Return {espn_id: dict} for every team.
    Includes: wins, losses, ppg, opp_ppg, ppg_rank, def_rank,
              conf, conf_seed, streak, l10.
    """
    data = _espn_fetch(_ESPN_STANDINGS_URL)
    rows = []
    for conf_entry in data.get("children", []):
        conf_name = conf_entry.get("name", "")
        for entry in conf_entry.get("standings", {}).get("entries", []):
            tid   = entry["team"]["id"]
            stats = {s["name"]: s["displayValue"] for s in entry.get("stats", [])}
            rows.append({
                "id":        tid,
                "wins":      int(float(stats.get("wins", 0))),
                "losses":    int(float(stats.get("losses", 0))),
                "ppg":       float(stats.get("avgPointsFor", 0) or 0),
                "opp_ppg":   float(stats.get("avgPointsAgainst", 0) or 0),
                "conf":      conf_name,
                "conf_seed": int(float(stats.get("playoffSeed", 0) or 0)),
                "streak":    stats.get("streak", "—"),
                "l10":       stats.get("Last Ten Games", "—"),
            })

    ppg_sorted = sorted(rows, key=lambda r: r["ppg"], reverse=True)
    ppg_rank   = {r["id"]: i + 1 for i, r in enumerate(ppg_sorted)}
    def_sorted = sorted(rows, key=lambda r: r["opp_ppg"])
    def_rank   = {r["id"]: i + 1 for i, r in enumerate(def_sorted)}

    return {
        r["id"]: {**r, "ppg_rank": ppg_rank[r["id"]], "def_rank": def_rank[r["id"]]}
        for r in rows
    }


def _build_player_lookup() -> list[dict]:
    """
    Fetch league-wide per-game stats (sorted by PPG descending).
    ESPN JSON layout (verified against live API):
      categories[0] = general   → totals[1]  = MIN,  totals[11] = RPG
      categories[1] = offensive → totals[0]  = PPG,  totals[3]  = FG%,
                                  totals[6]  = 3P%,  totals[7]  = APG,
                                  totals[9]  = FT%,  totals[11] = TOV
      categories[2] = defensive → totals[0]  = STL,  totals[1]  = BLK
    """
    data = _espn_fetch(_ESPN_STATS_URL)
    players = []
    for a in data.get("athletes", []):
        athlete = a.get("athlete", {})
        cats    = a.get("categories", [])
        if len(cats) < 3:
            continue
        try:
            mins   = cats[0]["totals"][1]
            rpg    = cats[0]["totals"][11]
            ppg    = cats[1]["totals"][0]
            fg_pct = cats[1]["totals"][3]
            tpp    = cats[1]["totals"][6]
            apg    = cats[1]["totals"][7]
            ft_pct = cats[1]["totals"][9]
            tov    = cats[1]["totals"][11]
            stl    = cats[2]["totals"][0]
            blk    = cats[2]["totals"][1]
        except (IndexError, KeyError):
            continue
        players.append({
            "name":    athlete.get("displayName", "Unknown"),
            "team_id": athlete.get("teamId", ""),
            "mins": mins, "ppg": ppg, "fg_pct": fg_pct,
            "tpp": tpp, "ft_pct": ft_pct,
            "rpg": rpg, "apg": apg,
            "stl": stl, "blk": blk, "tov": tov,
        })
    return players


def _fetch_h2h_games(
    t1_id: str, t2_id: str,
    t1_name: str, t2_name: str,
) -> str:
    """
    Fetch all COMPLETED regular-season games between t1 and t2 this season
    directly from the ESPN schedule endpoint.

    Returns a formatted multi-line string with exact scores, dates, home team,
    and arena — no LLM involved, no guessing.

    Dates are converted from UTC to US Eastern time (game-night local date).
    """
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
        f"/teams/{t1_id}/schedule?season={ESPN_SEASON_YEAR}"
    )
    data = _espn_fetch(url)

    lines = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        if not comp["status"]["type"]["completed"]:
            continue
        competitor_ids = [c["team"]["id"] for c in comp["competitors"]]
        if t2_id not in competitor_ids:
            continue

        # UTC → Eastern local date (NBA games end after midnight UTC)
        utc_dt     = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        local_date = utc_dt.astimezone(EST).strftime("%B %d, %Y")

        venue = comp.get("venue", {})
        arena = venue.get("fullName", "unknown arena")
        city  = venue.get("address", {}).get("city", "")

        scores    = {}
        home_name = ""
        for c in comp["competitors"]:
            tid  = c["team"]["id"]
            name = c["team"]["displayName"]
            sc   = int(float(c["score"]["displayValue"]))
            scores[tid] = {"name": name, "score": sc}
            if c["homeAway"] == "home":
                home_name = name

        t1 = scores.get(t1_id, {"name": t1_name, "score": 0})
        t2 = scores.get(t2_id, {"name": t2_name, "score": 0})

        if t1["score"] > t2["score"]:
            win_name, win_sc = t1["name"], t1["score"]
            los_name, los_sc = t2["name"], t2["score"]
        else:
            win_name, win_sc = t2["name"], t2["score"]
            los_name, los_sc = t1["name"], t1["score"]

        lines.append(
            f"• {local_date}: {win_name} def. {los_name} {win_sc}-{los_sc} "
            f"| Home team: {home_name} | Arena: {arena}, {city}"
        )

    if not lines:
        return "No completed H2H games found this season."
    return "\n".join(lines)


def _fetch_injuries(team_id: str, team_name: str) -> list[str]:
    """
    Fetch current injury list for a team from ESPN roster API.
    Returns list of strings: "PlayerName — STATUS"
    Only includes players with an active injury entry.
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
    data = _espn_fetch(url)
    results = []
    for athlete in data.get("athletes", []):
        injuries = athlete.get("injuries", [])
        if not injuries:
            continue
        # Use the most recent injury entry
        latest = max(injuries, key=lambda i: i.get("date", ""))
        status = latest.get("status", "")
        if status:
            results.append(f"{athlete['displayName']} — {status}")
    return results


def _fetch_recent_form(
    team_id: str, team_name: str, n_games: int = 3
) -> list[str]:
    """
    Fetch the top scorer from each of the last n completed games for a team.
    Uses ESPN schedule + game summary APIs.
    Returns list of strings: "Date vs Opponent: PlayerName scored X pts (W/L score)"
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule?season={ESPN_SEASON_YEAR}"
    data = _espn_fetch(url)

    completed = [
        e for e in data.get("events", [])
        if e["competitions"][0]["status"]["type"]["completed"]
    ]
    recent = completed[-n_games:]

    lines = []
    for event in recent:
        comp    = event["competitions"][0]
        game_id = comp["id"]
        utc_dt  = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        date_str = utc_dt.astimezone(EST).strftime("%b %d")

        # Determine opponent and result
        team_score = opp_score = opp_name = ""
        result = ""
        for c in comp["competitors"]:
            if c["team"]["id"] == team_id:
                team_score = int(float(c["score"]["displayValue"]))
                result = "W" if c.get("winner") else "L"
            else:
                opp_score = int(float(c["score"]["displayValue"]))
                opp_name  = c["team"]["abbreviation"]

        # Fetch top scorer from game summary
        try:
            summary = _espn_fetch(
                f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
            )
            top_scorer = top_pts = None
            for team_leaders in summary.get("leaders", []):
                if team_leaders["team"]["id"] == team_id:
                    for cat in team_leaders.get("leaders", []):
                        if cat.get("displayName") == "Points":
                            leader = cat["leaders"][0]
                            top_scorer = leader["athlete"]["displayName"]
                            top_pts    = leader["displayValue"]
                            break
        except Exception:
            top_scorer = top_pts = None

        scorer_str = f"{top_scorer} {top_pts} pts" if top_scorer else "scorer unavailable"
        lines.append(
            f"{date_str} vs {opp_name} ({result} {team_score}-{opp_score}): {scorer_str}"
        )

    return lines


def _analyze_form(form_lines: list[str], team_name: str) -> str:
    """
    Pre-analyze recent form lines and return a structured summary for the composer.
    Covers: dominant scorer, their point range, win/loss arc.
    Format: bullet points the LLM must use verbatim as the basis for its narrative.
    """
    import re
    from collections import Counter

    # Parse each line: "Mon DD vs OPP (W/L score-score): Player N pts"
    wins = losses = 0
    scorer_counts: Counter = Counter()
    scorer_pts: dict[str, list[int]] = {}
    scorer_peak_opp: dict[str, str] = {}   # name -> opponent for their peak game
    scorer_opps: dict[str, list[str]] = {}  # name -> list of opponents in games they led
    recent_results = []

    for line in form_lines:
        opp_m    = re.search(r'vs (\w+)', line)
        result_m = re.search(r'\((W|L)\s+\d+-\d+\)', line)
        scorer_m = re.search(r':\s+(.+?)\s+(\d+)\s+pts', line)
        opp = opp_m.group(1) if opp_m else ""
        if result_m:
            if result_m.group(1) == "W":
                wins += 1
                recent_results.append("W")
            else:
                losses += 1
                recent_results.append("L")
        if scorer_m:
            name = scorer_m.group(1)
            pts  = int(scorer_m.group(2))
            scorer_counts[name] += 1
            scorer_opps.setdefault(name, []).append(opp)
            if pts > max(scorer_pts.get(name, [0])):
                scorer_peak_opp[name] = opp
            scorer_pts.setdefault(name, []).append(pts)

    total = len(form_lines)
    bullets = []

    # Win/loss record
    bullets.append(f"  • Record in last {total} games: {wins}-{losses}")

    # Win/loss arc — pre-written phrase for LLM to incorporate
    if recent_results and total >= 6:
        first_half  = recent_results[:total//2]
        second_half = recent_results[total//2:]
        first_w  = first_half.count("W")
        second_w = second_half.count("W")
        first_l  = len(first_half) - first_w
        second_l = len(second_half) - second_w
        if second_w - first_w >= 2 and second_w >= len(second_half) // 2 + 1:
            bullets.append(
                f"  • SURGE ARC — incorporate this: '{team_name} have found their footing, "
                f"winning {second_w} of their last {len(second_half)} after going {first_w}-{first_l} to open this stretch.'"
            )
        elif first_w - second_w >= 2:
            bullets.append(
                f"  • COLLAPSE ARC — incorporate this: '{team_name} have fallen off sharply, "
                f"dropping {second_l} of their last {len(second_half)} after going {first_w}-{first_l} to start this stretch.'"
            )

    # Dominant scorer — directive only, no game counts
    if scorer_counts:
        top_name, top_count = scorer_counts.most_common(1)[0]
        pts_list = scorer_pts[top_name]
        pts_min, pts_max = min(pts_list), max(pts_list)
        opps_str = " and ".join(scorer_opps.get(top_name, []))
        if top_count >= round(total * 0.6):
            bullets.append(
                f"  • Offensive engine: {top_name} — led team scoring against {opps_str}, "
                f"posting {pts_min}–{pts_max} pts in those games. "
                f"Describe as the team's go-to scorer. Do NOT say 'over the last N games'."
            )
        elif top_count >= round(total * 0.4):
            bullets.append(
                f"  • Go-to scorer: {top_name} — led team scoring against {opps_str}, "
                f"posting {pts_min}–{pts_max} pts in those games. "
                f"Describe as the primary offensive option. Do NOT say 'over the last N games'."
            )
        else:
            bullets.append(
                f"  • Spread offense: no consistent individual scorer in the last {total} games. "
                f"Describe as a collective effort with no single go-to option."
            )

        # Notable single-game peaks — skip if player already named as engine/go-to
        # (their peak opponent is already embedded in that bullet)
        all_peaks = [(name, max(pts)) for name, pts in scorer_pts.items() if max(pts) >= 30]
        all_peaks.sort(key=lambda x: -x[1])
        for name, peak in all_peaks[:2]:
            if name == top_name:
                continue  # already covered in the offensive engine / go-to bullet
            opp = scorer_peak_opp.get(name, "")
            vs_str = f" against {opp}" if opp else ""
            bullets.append(f"  • Notable performance: {name} scored {peak} pts{vs_str}")

    return "\n".join(bullets)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def data_specialist_node(state: GraphState) -> dict:
    """
    Node 1 — Data Specialist (100% ESPN API, zero LLM, zero Tavily).
    Fetches:
      • Standings: W/L, seed, team PPG/Opp-PPG with league rank, streak, L10
      • Player stats: top-5 scorers per team (PPG, FG%, MIN, RPG, APG)
      • H2H: all completed games this season between the two teams,
             with exact scores, local dates, home team, and arena
    """
    query = state["query"]
    teams = extract_teams(query)
    print(f"[DataSpecialist]    Extracted teams: {[t[0] for t in teams]}")


    if not teams:
        print("[DataSpecialist]    Could not identify any NBA teams in query.")
        return {"player_stats_table": "No teams identified.", "h2h_summary": ""}

    print("[DataSpecialist]    Fetching standings ...")
    try:
        standings = _build_standings_lookup()
        print(f"[DataSpecialist]    Standings: {len(standings)} teams")
    except Exception as exc:
        print(f"[DataSpecialist]    Standings failed: {exc}")
        standings = {}

    print("[DataSpecialist]    Fetching player stats ...")
    try:
        all_players = _build_player_lookup()
        print(f"[DataSpecialist]    Players: {len(all_players)} qualified")
    except Exception as exc:
        print(f"[DataSpecialist]    Player stats failed: {exc}")
        all_players = []

    # ── Fetch injuries first (needed to filter OUT players from roster) ───────
    injuries_by_team: dict[str, list[str]] = {}
    out_names_by_team: dict[str, set[str]] = {}
    injury_parts = []
    for full_name, _, espn_id in teams:
        print(f"[DataSpecialist]    Fetching injuries: {full_name} ...")
        try:
            injured = _fetch_injuries(espn_id, full_name)
            injuries_by_team[espn_id] = injured
            # Collect names of OUT players (not Day-To-Day) for roster filtering
            out_names_by_team[espn_id] = {
                entry.split(" — ")[0]
                for entry in injured
                if "Out" in entry and "Day-To-Day" not in entry
            }
            if injured:
                lines = "\n".join(f"  • {p}" for p in injured)
                injury_parts.append(f"### {full_name}\n{lines}")
                print(f"[DataSpecialist]    {full_name}: {len(injured)} injured player(s)")
                for p in injured:
                    print(f"  • {p}")
            else:
                injury_parts.append(f"### {full_name}\n  No injuries reported.")
                print(f"[DataSpecialist]    {full_name}: No injuries reported.")
        except Exception as exc:
            print(f"[DataSpecialist]    Injuries fetch failed for {full_name}: {exc}")
            injuries_by_team[espn_id] = []
            out_names_by_team[espn_id] = set()
            injury_parts.append(f"### {full_name}\n  Data unavailable.")

    injury_summary = "\n\n".join(injury_parts)

    # ── Build per-team stats sections (OUT players excluded) ───────────────
    sections = []
    for full_name, nickname, espn_id in teams:
        info    = standings.get(espn_id, {})
        wins    = info.get("wins", "?")
        losses  = info.get("losses", "?")
        seed    = info.get("conf_seed", "?")
        conf    = info.get("conf", "Conference")
        streak  = info.get("streak", "—")
        l10     = info.get("l10", "—")
        ppg     = info.get("ppg", 0)
        opp_ppg = info.get("opp_ppg", 0)
        ppg_r   = info.get("ppg_rank", "?")
        def_r   = info.get("def_rank", "?")

        ppg_str  = f"{ppg:.1f} PPG ({ordinal(ppg_r)} in NBA)" if ppg else "PPG N/A"
        def_str  = f"{opp_ppg:.1f} Opp PPG ({ordinal(def_r)} in NBA)" if opp_ppg else "Def N/A"
        seed_str = f"#{seed} {conf.replace(' Conference','')}" if seed != "?" else "N/A"

        out_names = out_names_by_team.get(espn_id, set())
        roster = [
            p for p in all_players
            if p["team_id"] == espn_id and p["name"] not in out_names
        ]

        out_count    = len(out_names)
        active_count = len(roster)
        # Only flag in header when genuinely short-handed (≤8 active)
        availability = (
            f" | Active roster: {active_count} players ({out_count} OUT)"
            if active_count <= 8 else ""
        )
        rows = [
            f"### {full_name}",
            f"Record: W {wins} / L {losses} | Seed: {seed_str} | Streak: {streak} | Last 10: {l10}{availability}",
            f"Team Stats: {ppg_str} | {def_str}",
            "| Player | MIN | PPG | FG% | 3P% | FT% | RPG | APG | STL | BLK | TOV |",
            "|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|",
        ]
        if roster:
            for p in roster:
                rows.append(
                    f"| {p['name']} | {p['mins']} | {p['ppg']} | {p['fg_pct']} "
                    f"| {p['tpp']} | {p['ft_pct']} | {p['rpg']} | {p['apg']} "
                    f"| {p['stl']} | {p['blk']} | {p['tov']} |"
                )
        else:
            rows.append("| Stats unavailable | — | — | — | — | — |")

        sections.append("\n".join(rows))
        print(
            f"[DataSpecialist]    {full_name}: W{wins}/L{losses} | {seed_str} | "
            f"Streak:{streak} | L10:{l10} | {ppg_str} | {def_str} | {len(roster)} active players"
        )

    # ── Fetch H2H via ESPN schedule API ────────────────────────────────────
    h2h_text = ""
    if len(teams) == 2:
        t1_full, _, t1_id = teams[0]
        t2_full, _, t2_id = teams[1]
        print(f"[DataSpecialist]    Fetching H2H: {t1_full} vs {t2_full} ...")
        try:
            h2h_text = _fetch_h2h_games(t1_id, t2_id, t1_full, t2_full)
            game_count = h2h_text.count("•")
            print(f"[DataSpecialist]    H2H: {game_count} completed game(s) found")
            for line in h2h_text.splitlines():
                print(f"  {line}")
        except Exception as exc:
            print(f"[DataSpecialist]    H2H fetch failed: {exc}")
            h2h_text = "H2H data unavailable."

    # ── Fetch recent form (last 10 games top scorer, OUT players excluded) ──
    form_parts = []
    for full_name, _, espn_id in teams:
        print(f"[DataSpecialist]    Fetching recent form: {full_name} ...")
        out_names = out_names_by_team.get(espn_id, set())
        active_names = {p["name"] for p in all_players if p["team_id"] == espn_id}
        try:
            form_lines = _fetch_recent_form(espn_id, full_name, n_games=5)
            # Keep only games where top scorer is active and not OUT
            form_lines = [
                l for l in form_lines
                if any(name in l for name in active_names)
                and not any(out_name in l for out_name in out_names)
            ]
            if form_lines:
                for l in form_lines:
                    print(f"  • {l}")
                # ── Pre-analyze form for the composer ──────────────────────
                summary = _analyze_form(form_lines, full_name)
                form_parts.append(
                    f"### {full_name} Recent Form — ANALYSIS ONLY (do NOT use raw game lines)\n"
                    f"{summary}"
                )
            else:
                form_parts.append(f"### {full_name} Recent Form\n  No recent games found.")
        except Exception as exc:
            print(f"[DataSpecialist]    Recent form fetch failed for {full_name}: {exc}")
            form_parts.append(f"### {full_name} Recent Form\n  Data unavailable.")

    recent_form = "\n\n".join(form_parts)

    return {
        "player_stats_table": "\n\n".join(sections),
        "h2h_summary":        h2h_text,
        "injury_summary":     injury_summary,
        "recent_form":        recent_form,
    }
