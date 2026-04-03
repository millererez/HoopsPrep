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

    # ── Build per-team stats sections ──────────────────────────────────────
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

        roster = [p for p in all_players if p["team_id"] == espn_id]

        rows = [
            f"### {full_name}",
            f"Record: W {wins} / L {losses} | Seed: {seed_str} | Streak: {streak} | Last 10: {l10}",
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
            f"Streak:{streak} | L10:{l10} | {ppg_str} | {def_str} | {len(roster)} players"
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

    return {
        "player_stats_table": "\n\n".join(sections),
        "h2h_summary":        h2h_text,
    }
