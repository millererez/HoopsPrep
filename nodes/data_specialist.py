"""
nodes/data_specialist.py
────────────────────────
Node 1 — Data Specialist (100% ESPN API, zero LLM, zero Tavily).
Fetches standings, player stats, injuries, recent form, and H2H game history.
"""

import re
from collections import Counter

from core.state import GraphState, ordinal, extract_teams, parse_home_away
from nodes.espn_client import (
    build_standings_lookup,
    build_player_lookup,
    fetch_h2h_games,
    fetch_injuries,
    fetch_full_active_roster,
    fetch_recent_form,
)


# ---------------------------------------------------------------------------
# Form analysis
# ---------------------------------------------------------------------------

def _analyze_form(
    form_lines: list[str],
    team_name: str,
    season_star: tuple[str, float] | None = None,
    season_ppg_lookup: dict[str, float] | None = None,
) -> str:
    """
    Pre-analyze recent form lines and return a structured summary for the composer.
    Covers: dominant scorer, their point range, win/loss arc.
    Format: bullet points the LLM must use verbatim as the basis for its narrative.
    """
    wins = losses = 0
    scorer_counts: Counter = Counter()
    scorer_pts: dict[str, list[int]] = {}
    scorer_peak_opp: dict[str, str] = {}
    scorer_opps: dict[str, list[str]] = {}
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

    bullets.append(f"  • Record in last {total} games: {wins}-{losses}")

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

    if scorer_counts:
        top_name, top_count = scorer_counts.most_common(1)[0]
        pts_list = scorer_pts[top_name]
        opps_list = scorer_opps.get(top_name, [])
        per_game_str = " and ".join(
            f"{pts} pts vs {opp}" for pts, opp in zip(pts_list, opps_list)
        )
        if top_count >= round(total * 0.6):
            bullets.append(
                f"  • Offensive engine: {top_name} — led team scoring: {per_game_str}. "
                f"Open directly: '{top_name} scored...' — no preamble or characterization."
            )
        elif top_count >= round(total * 0.4):
            bullets.append(
                f"  • Go-to scorer: {top_name} — led team scoring: {per_game_str}. "
                f"Open directly: '{top_name} scored...' — no preamble or characterization."
            )
        else:
            bullets.append(
                f"  • Spread offense: no consistent individual scorer. "
                f"INCLUDE THIS SENTENCE VERBATIM: '{team_name} operate with a spread offense, with no single go-to scorer.'"
            )

        all_peaks = [(name, max(pts)) for name, pts in scorer_pts.items() if max(pts) >= 30]
        all_peaks.sort(key=lambda x: -x[1])
        for name, peak in all_peaks[:2]:
            if name == top_name:
                continue
            opp = scorer_peak_opp.get(name, "")
            vs_str = f" against {opp}" if opp else ""
            bullets.append(f"  • Notable performance: {name} scored {peak} pts{vs_str}")

    # ── Season-star override ──────────────────────────────────────────────────
    # Two triggers:
    # 1. Star differs from form leader and has ≥3 PPG gap over them (original)
    # 2. Star leads team by ≥5 PPG over the second-best player (dominance override)
    #    — catches cases where star ties for form leader or scores in spread offense
    if season_star:
        star_name, star_ppg = season_star
        all_ppgs = sorted((season_ppg_lookup or {}).values(), reverse=True)
        second_best_ppg = all_ppgs[1] if len(all_ppgs) > 1 else 0.0
        dominant = star_ppg - second_best_ppg >= 5

        form_leader = scorer_counts.most_common(1)[0][0] if scorer_counts else None
        form_leader_season_ppg = (season_ppg_lookup or {}).get(form_leader, 0.0) if form_leader else 0.0
        gap_override = form_leader and star_name != form_leader and star_ppg - form_leader_season_ppg >= 3

        if dominant or gap_override:
            bullets = [b for b in bullets if "Offensive engine:" not in b and "Go-to scorer:" not in b and "Spread offense:" not in b]
            star_pts = scorer_pts.get(star_name, [])
            if star_pts:
                star_opps = scorer_opps.get(star_name, [])
                per_game_str = " and ".join(
                    f"{pts} pts vs {opp}" for pts, opp in zip(star_pts, star_opps)
                )
                pts_clause = f"scoring {per_game_str} when leading" if per_game_str else f"scoring {star_pts[0]} pts when leading"
                bullets.append(
                    f"  • Offensive engine: {star_name} ({star_ppg:.1f} PPG season) — "
                    f"{pts_clause}. Open directly: '{star_name} scored...' — no preamble or characterization."
                )
            else:
                bullets.append(
                    f"  • Offensive engine: {star_name} ({star_ppg:.1f} PPG season) — "
                    f"primary scorer. Did not lead team scoring in this sample but is the clear go-to option."
                )
            if form_leader:
                form_pts = scorer_pts.get(form_leader, [])
                if form_pts and max(form_pts) >= 25 and form_leader != star_name:
                    opp = scorer_peak_opp.get(form_leader, "")
                    vs_str = f" against {opp}" if opp else ""
                    bullets.append(f"  • Notable performance: {form_leader} scored {max(form_pts)} pts{vs_str}")

    return "\n".join(bullets)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def data_specialist_node(state: GraphState) -> dict:
    """
    Node 1 — Data Specialist (100% ESPN API, zero LLM, zero Tavily).
    Fetches:
      • Standings: W/L, seed, team PPG/Opp-PPG with league rank, streak, L10
      • Player stats: top scorers per team (PPG, FG%, MIN, RPG, APG, etc.)
      • H2H: all completed games this season between the two teams
      • Injuries: current injury report per team
      • Recent form: last 5 games top scorer with pre-analyzed summary
    """
    query = state["query"]
    teams = extract_teams(query)
    if len(teams) == 2:
        away_team, home_team = parse_home_away(query, teams)
        teams = [home_team, away_team]
    print(f"[DataSpecialist]    Extracted teams (home first): {[t[0] for t in teams]}")

    if not teams:
        print("[DataSpecialist]    Could not identify any NBA teams in query.")
        return {"player_stats_table": "No teams identified.", "h2h_summary": ""}

    print("[DataSpecialist]    Fetching standings ...")
    try:
        standings = build_standings_lookup()
        print(f"[DataSpecialist]    Standings: {len(standings)} teams")
    except Exception as exc:
        print(f"[DataSpecialist]    Standings failed: {exc}")
        standings = {}

    print("[DataSpecialist]    Fetching player stats ...")
    try:
        all_players = build_player_lookup()
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
            injured = fetch_injuries(espn_id, full_name)
            injuries_by_team[espn_id] = injured
            out_names_by_team[espn_id] = {
                entry.split(" \u2014 ")[0]
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

        if active_count <= 6:
            try:
                all_injured = out_names | {
                    e.split(" — ")[0]
                    for e in injuries_by_team.get(espn_id, [])
                }
                qualified_names = {p["name"] for p in roster}
                full_active = fetch_full_active_roster(espn_id, all_injured)
                fillers = [n for n in full_active if n not in qualified_names]
                if fillers:
                    rows.append(
                        f"EMERGENCY ROSTER (G League / 10-day — no qualifying stats): "
                        + ", ".join(fillers)
                    )
            except Exception:
                pass

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
            h2h_text = fetch_h2h_games(t1_id, t2_id, t1_full, t2_full)
            game_count = h2h_text.count("•")
            print(f"[DataSpecialist]    H2H: {game_count} completed game(s) found")
            for line in h2h_text.splitlines():
                print(f"  {line}")
        except Exception as exc:
            print(f"[DataSpecialist]    H2H fetch failed: {exc}")
            h2h_text = "H2H data unavailable."

    # ── Build season-star and PPG lookups for form override ──────────────────
    season_stars: dict[str, tuple[str, float]] = {}
    season_ppg_by_team: dict[str, dict[str, float]] = {}
    for full_name, _, espn_id in teams:
        out_names = out_names_by_team.get(espn_id, set())
        team_roster = [p for p in all_players if p["team_id"] == espn_id and p["name"] not in out_names]
        if team_roster:
            star = team_roster[0]  # all_players is sorted PPG desc
            season_stars[espn_id] = (star["name"], float(star["ppg"]))
            season_ppg_by_team[espn_id] = {p["name"]: float(p["ppg"]) for p in team_roster}

    # ── Fetch recent form (last 5 games, OUT players excluded) ──────────────
    form_parts = []
    for full_name, _, espn_id in teams:
        print(f"[DataSpecialist]    Fetching recent form: {full_name} ...")
        out_names = out_names_by_team.get(espn_id, set())
        active_names = {p["name"] for p in all_players if p["team_id"] == espn_id}
        try:
            form_lines = fetch_recent_form(espn_id, full_name, n_games=5)
            form_lines = [
                l for l in form_lines
                if any(name in l for name in active_names)
                and not any(out_name in l for out_name in out_names)
            ]
            if form_lines:
                for l in form_lines:
                    print(f"  • {l}")
                summary = _analyze_form(
                    form_lines, full_name,
                    season_star=season_stars.get(espn_id),
                    season_ppg_lookup=season_ppg_by_team.get(espn_id),
                )
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
