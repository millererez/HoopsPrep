"""
nodes/data_specialist.py
────────────────────────
Node 1 — Data Specialist (100% ESPN API, zero LLM, zero Tavily).
Fetches standings, player stats, injuries, recent form, and H2H game history.
"""

import re
from collections import Counter

from core.state import GraphState, ordinal, extract_teams, parse_home_away
from nodes.utils import fmt_num
from nodes.espn_client import (
    build_standings_lookup,
    build_player_lookup,
    build_jersey_lookup,
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
    out_star: tuple[str, float] | None = None,
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
    scorer_dates: dict[str, list[str]] = {}
    recent_results = []

    for line in form_lines:
        date_m   = re.search(r'^(\w+ \d+)\s+vs', line)
        opp_m    = re.search(r'vs ([A-Za-z ]+?) \(', line)
        result_m = re.search(r'\((W|L)\s+\d+-\d+\)', line)
        scorer_m = re.search(r':\s+(.+?)\s+(\d+)\s+pts', line)
        opp  = opp_m.group(1) if opp_m else ""
        date = date_m.group(1) if date_m else ""
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
            scorer_dates.setdefault(name, []).append(date)
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
        # Tie-break: when two players share the top count, prefer the season star.
        # Prevents insertion-order randomness from picking the wrong engine.
        top_candidates = scorer_counts.most_common(2)
        if (
            len(top_candidates) >= 2
            and top_candidates[0][1] == top_candidates[1][1]
            and season_star
            and season_star[0] in {n for n, _ in top_candidates}
        ):
            top_name  = season_star[0]
            top_count = top_candidates[0][1]
        else:
            top_name, top_count = top_candidates[0]
        pts_list  = scorer_pts[top_name]
        opps_list = scorer_opps.get(top_name, [])
        dates_list = scorer_dates.get(top_name, [])
        # Pre-build the verbatim opening sentence — LLM must copy it exactly
        top2 = sorted(zip(pts_list, opps_list, dates_list), key=lambda x: -x[0])[:2]
        top2 = [(pts, opp, date) for pts, opp, date in top2 if opp.strip()]
        _use_dates = len({opp for _, opp, _ in top2}) < len(top2)  # True if duplicate opponent
        verbatim = f"{top_name} scored " + " and ".join(
            (f"{pts} on {date} against the {opp}" if _use_dates else f"{pts} against the {opp}")
            for pts, opp, date in top2
        ) + "."
        per_game_str = " and ".join(
            f"{pts} pts vs {opp} ({date})" for pts, opp, date in zip(pts_list, opps_list, dates_list)
        )
        if top_count >= round(total * 0.6):
            bullets.append(
                f"  • Offensive engine: {top_name} — led team scoring: {per_game_str}. "
                f"VERBATIM OPENING SENTENCE (copy exactly): '{verbatim}'"
            )
        elif top_count >= round(total * 0.4):
            bullets.append(
                f"  • Go-to scorer: {top_name} — led team scoring: {per_game_str}. "
                f"VERBATIM OPENING SENTENCE (copy exactly): '{verbatim}'"
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
            star_pts = scorer_pts.get(star_name, [])
            if star_pts:
                # Star appeared in recent games — override the form scorer
                bullets = [b for b in bullets if "Offensive engine:" not in b and "Go-to scorer:" not in b and "Spread offense:" not in b]
                star_opps  = scorer_opps.get(star_name, [])
                star_dates = scorer_dates.get(star_name, [])
                top2_star = sorted(zip(star_pts, star_opps, star_dates), key=lambda x: -x[0])[:2]
                _use_dates_star = len({opp for _, opp, _ in top2_star}) < len(top2_star)
                verbatim_star = f"{star_name} scored " + " and ".join(
                    (f"{pts} on {date} against the {opp}" if _use_dates_star else f"{pts} against the {opp}")
                    for pts, opp, date in top2_star
                ) + "."
                per_game_str = " and ".join(
                    f"{pts} pts vs {opp} ({date})" for pts, opp, date in zip(star_pts, star_opps, star_dates)
                )
                bullets.append(
                    f"  • Offensive engine: {star_name} ({fmt_num(star_ppg)} PPG season) — "
                    f"{per_game_str}. VERBATIM OPENING SENTENCE (copy exactly): '{verbatim_star}'"
                )
                if form_leader:
                    form_pts = scorer_pts.get(form_leader, [])
                    if form_pts and max(form_pts) >= 25 and form_leader != star_name:
                        opp = scorer_peak_opp.get(form_leader, "")
                        vs_str = f" against {opp}" if opp else ""
                        bullets.append(f"  • Notable performance: {form_leader} scored {max(form_pts)} pts{vs_str}")
            else:
                # Star didn't appear in the recent form window (e.g., injured/DTD).
                # Keep the form scorer — they are the correct recent offensive driver.
                # Add the star separately so the LLM knows their season role.
                bullets.append(
                    f"  • Season star (not in recent form window): {star_name} ({fmt_num(star_ppg)} PPG season). "
                    f"Mention season average only — do NOT attribute recent game scores to this player."
                )

    # Absent star — OUT player who would otherwise be the team's top scorer.
    # Inject as an explicit signal so the LLM frames their absence correctly.
    if out_star:
        out_name, out_ppg = out_star
        form_scorer = scorer_counts.most_common(1)[0][0] if scorer_counts else "the team"
        bullets.append(
            f"  • ABSENT STAR: {out_name} ({fmt_num(out_ppg)} PPG season avg) is OUT tonight. "
            f"Open this paragraph with exactly one sentence acknowledging his absence, e.g.: "
            f"'With {out_name} sidelined, {form_scorer} has taken over as the primary scorer.' "
            f"Do NOT describe {out_name} as contributing, scoring, or influencing tonight's game."
        )

    # Drop any "Notable performance" bullet for a player already named in a VERBATIM sentence.
    # Prevents the composer from generating two separate sentences about the same player.
    verbatim_players = set()
    for b in bullets:
        if "VERBATIM OPENING SENTENCE" in b or "INCLUDE THIS SENTENCE VERBATIM" in b:
            # Extract player name — it's the first word(s) before " scored" or " operate"
            import re as _re3
            m = _re3.match(r"\s*•\s*(?:Offensive engine|Go-to scorer|ABSENT STAR|Season star)?[^:]*:\s*(\w[\w\s\-'\.]+?)\s+(?:scored|operate)", b)
            if m:
                verbatim_players.add(m.group(1).strip())
    if verbatim_players:
        bullets = [
            b for b in bullets
            if not (
                b.strip().startswith("• Notable performance:")
                and any(p in b for p in verbatim_players)
            )
        ]

    return "\n".join(bullets)


# ---------------------------------------------------------------------------
# Stakes context
# ---------------------------------------------------------------------------

def _compute_stakes_context(
    full_name: str,
    seed: int,
    conf: str,
    wins: int,
    losses: int,
    standings: dict,
) -> str:
    """
    Compute a grounded stakes string for one team.
    Reports every meaningful boundary within games_remaining distance — both
    upside (can reach) and risk (can slip to). Uses correct NBA play-in rules.
    """
    games_remaining = 82 - wins - losses
    if games_remaining <= 0 or seed <= 0:
        return f"{full_name}: no games remaining"

    # Eliminated teams (seed 11+): no meaningful boundaries to report.
    # Avoids contradicting the "eliminated" label with reachable play-in distances.
    if seed > 10:
        return f"{full_name}: #{seed} — eliminated from playoff contention | {games_remaining} games remaining"

    # Seed → (wins, losses) for same conference
    conf_seed_map: dict[int, tuple[int, int]] = {}
    for d in standings.values():
        if d.get("conf") == conf:
            s = d.get("conf_seed", 0)
            if s > 0:
                conf_seed_map[s] = (d.get("wins", 0), d.get("losses", 0))

    def gb_from(target: int) -> float | None:
        """Positive = target is above us. Negative = target is below us."""
        if target not in conf_seed_map or target == seed:
            return None
        tw, tl = conf_seed_map[target]
        return ((tw - wins) + (losses - tl)) / 2

    def _playoff_tier(s: int) -> int:
        """0 = home court (1-4), 1 = direct playoff (5-6), 2 = play-in (7-10), 3 = eliminated."""
        if s <= 4:  return 0
        if s <= 6:  return 1
        if s <= 10: return 2
        return 3

    current_tier = _playoff_tier(seed)

    # Meaningful boundary seeds and what crossing them means
    BOUNDARIES: dict[int, str] = {
        4:  "home court in first round",
        5:  "losing home court (falls to direct playoff #5)",
        6:  "direct playoff berth",
        7:  "play-in double-chance (hosts 7/8, guaranteed two attempts)",
        8:  "play-in with second chance if first game lost",
        9:  "play-in must-win first game",
        10: "play-in last chance (lose = immediately eliminated)",
    }

    # Determine if current tier is mathematically clinched:
    # clinched = team leads EVERY immediate lower-tier boundary by more than games_remaining.
    lower_tier_leads: list[float] = []
    for _ts in BOUNDARIES:
        if _ts == seed:
            continue
        _gb = gb_from(_ts)
        if _gb is None:
            continue
        _tt = _playoff_tier(_ts)
        if _tt != current_tier + 1:
            continue          # only check the immediate next-worse tier
        if _gb < 0:           # team is ahead of this boundary
            lower_tier_leads.append(abs(_gb))
    clinched = bool(lower_tier_leads) and all(lead > games_remaining for lead in lower_tier_leads)

    # Current position label — using correct play-in rules
    if seed <= 4:
        if clinched:
            pos_label = f"#{seed} — clinched home-court playoff position"
        else:
            pos_label = f"#{seed} — currently holding home-court position (not yet clinched)"
    elif seed <= 6:
        if clinched:
            pos_label = f"#{seed} — clinched direct playoff berth (no home court)"
        else:
            pos_label = f"#{seed} — currently in direct playoff position, not yet clinched"
    elif seed == 7:
        pos_label = "play-in #7 — hosts 7/8 game; guaranteed two chances"
    elif seed == 8:
        pos_label = "play-in #8 — win vs #7 = straight to playoffs as 7-seed; lose = host winner of 9/10 for 8-seed"
    elif seed == 9:
        pos_label = "play-in #9 — must win 9/10 game; winner plays loser of 7/8 for 8-seed"
    elif seed == 10:
        pos_label = "play-in #10 — must win 9/10 or immediately eliminated"
    else:
        pos_label = f"#{seed} — eliminated from playoff contention"

    reachable: list[str] = []
    at_risk:   list[str] = []

    for target_seed in sorted(BOUNDARIES.keys()):
        if target_seed == seed:
            continue
        gb = gb_from(target_seed)
        if gb is None:
            continue
        if 0 < gb <= games_remaining:
            reachable.append(f"{fmt_num(gb)} back of #{target_seed} — {BOUNDARIES[target_seed]}")
        elif -games_remaining <= gb < 0:
            target_tier = _playoff_tier(target_seed)
            # Skip same-tier risk: falling within the same tier changes seeding but not status.
            if target_tier == current_tier:
                continue
            # Skip multi-tier risk: you can't reach #7 without first passing #5.
            # Only show the immediately next tier down (current_tier + 1).
            if target_tier > current_tier + 1:
                continue
            at_risk.append(f"{fmt_num(abs(gb))} ahead of #{target_seed} — {BOUNDARIES[target_seed]}")

    result = f"{full_name}: {pos_label} | {games_remaining} games remaining"
    if reachable:
        result += " | UPSIDE: " + "; ".join(reachable)
    if at_risk:
        result += " | RISK: " + "; ".join(at_risk)
    return result


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

    # ── Build jersey lookup from roster endpoints (stats endpoint lacks jersey) ─
    jersey_lookup: dict[str, str] = {}
    for _, _, espn_id in teams:
        try:
            jersey_lookup.update(build_jersey_lookup(espn_id))
        except Exception:
            pass
    for p in all_players:
        if p.get("jersey") in ("—", "", None) and p["name"] in jersey_lookup:
            p["jersey"] = jersey_lookup[p["name"]]

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

    # ── Python-verified OUT players summary (passed to reviewer/rewriter) ───
    out_summary_lines = []
    for full_name, _, espn_id in teams:
        out_names = out_names_by_team.get(espn_id, set())
        names_str = ", ".join(sorted(out_names)) if out_names else "none"
        out_summary_lines.append(f"{full_name} OUT: {names_str}")
    out_players_summary = "\n".join(out_summary_lines)

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

        ppg_str  = f"{fmt_num(ppg)} PPG ({ordinal(ppg_r)} in NBA)" if ppg else "PPG N/A"
        def_str  = f"{fmt_num(opp_ppg)} Opp PPG ({ordinal(def_r)} in NBA)" if opp_ppg else "Def N/A"
        seed_str = f"#{seed} {conf.replace(' Conference','')}" if seed != "?" else "N/A"

        out_names = out_names_by_team.get(espn_id, set())
        roster = [
            p for p in all_players
            if p["team_id"] == espn_id and p["name"] not in out_names
        ]

        active_count = len(roster)
        rows = [
            f"### {full_name}",
            f"Record: W {wins} / L {losses} | Seed: {seed_str} | Streak: {streak} | Last 10: {l10}{availability}",
            f"",
            f"Team Stats: {ppg_str} | {def_str}",
            "| # | Name | POS | Age | GP | MIN | PPG | FGA | FG% | 3PA | 3P% | FTA | FT% | REB | APG | STL | BLK | TOV | PF |",
            "|---|--------|-----|-----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|",
        ]
        if roster:
            for p in roster:
                rows.append(
                    f"| {p.get('jersey', '—')} | {p['name']} | {p.get('pos', '—')} | {p.get('age', '—')} "
                    f"| {p.get('gp', '—')} | {fmt_num(p['mins'])} "
                    f"| {fmt_num(p['ppg'])} | {fmt_num(p['fga'])} | {p['fg_pct']} "
                    f"| {fmt_num(p['tpa'])} | {p['tpp']} | {fmt_num(p['fta'])} | {p['ft_pct']} "
                    f"| {fmt_num(p['rpg'])} | {fmt_num(p['apg'])} | {fmt_num(p['stl'])} | {fmt_num(p['blk'])} "
                    f"| {fmt_num(p['tov'])} | {p.get('pf', '—')} |"
                )
        else:
            rows.append("| — | Stats unavailable | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |")

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
    out_stars: dict[str, tuple[str, float]] = {}
    for full_name, _, espn_id in teams:
        out_names = out_names_by_team.get(espn_id, set())
        all_team = sorted(
            [p for p in all_players if p["team_id"] == espn_id],
            key=lambda p: float(p["ppg"]), reverse=True,
        )
        # If top PPG player is OUT, record them as an absent star
        if all_team and all_team[0]["name"] in out_names:
            out_stars[espn_id] = (all_team[0]["name"], float(all_team[0]["ppg"]))
        team_roster = [p for p in all_team if p["name"] not in out_names]
        if team_roster:
            star = team_roster[0]  # highest PPG among active players
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
                    out_star=out_stars.get(espn_id),
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

    # ── Compute stakes context for both teams ────────────────────────────────
    stakes_parts = []
    for full_name, _, espn_id in teams:
        info = standings.get(espn_id, {})
        stakes_parts.append(_compute_stakes_context(
            full_name  = full_name,
            seed       = info.get("conf_seed", 0),
            conf       = info.get("conf", ""),
            wins       = info.get("wins", 0),
            losses     = info.get("losses", 0),
            standings  = standings,
        ))
    stakes_context = "\n".join(stakes_parts)

    # ── Build player-team map (top 5 PPG per team, active only) ────────────────
    map_lines = []
    role = ["HOME TEAM", "AWAY TEAM"]
    for i, (full_name, _, espn_id) in enumerate(teams):
        ppg_lookup = season_ppg_by_team.get(espn_id, {})
        top5 = sorted(ppg_lookup.items(), key=lambda x: -x[1])[:5]
        names = ", ".join(n for n, _ in top5) if top5 else "No data"
        map_lines.append(f"{role[i] if i < 2 else 'TEAM'} ({full_name}): {names}")
    player_team_map = "\n".join(map_lines)

    return {
        "player_stats_table":  "\n\n".join(sections),
        "h2h_summary":         h2h_text,
        "injury_summary":      injury_summary,
        "recent_form":         recent_form,
        "stakes_context":      stakes_context,
        "player_team_map":     player_team_map,
        "out_players_summary": out_players_summary,
    }
