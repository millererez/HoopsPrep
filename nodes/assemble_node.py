"""
nodes/assemble_node.py
──────────────────────
Final assembly node — runs after report_composer + storylines_composer.
Combines prose, storylines, injury block, H2H block, and stats table
into the final report string in the correct order.
"""

import re

from core.state import GraphState, extract_teams, parse_home_away


def _injury_line(injury_summary: str, team_name: str) -> str:
    marker = f"### {team_name}"
    if marker not in injury_summary:
        return f"{team_name} \u2014 None reported."
    section = injury_summary.split(marker)[1].split("###")[0]
    entries = [l.strip(" \u2022").strip() for l in section.splitlines() if l.strip().startswith("\u2022")]
    if not entries:
        return f"{team_name} \u2014 None reported."
    formatted = ", ".join(
        f"{e.split(' \u2014 ')[0]} ({'OUT' if 'Out' in e else 'Day-To-Day'})"
        for e in entries if " \u2014 " in e
    )
    return f"{team_name} \u2014 {formatted}" if formatted else f"{team_name} \u2014 None reported."


def _h2h_prose(raw: str) -> str:
    """Convert raw H2H bullet lines into natural prose sentences."""
    if not raw or raw.startswith("No completed"):
        return raw
    sentences = []
    for line in raw.splitlines():
        line = line.strip().lstrip("\u2022 ")
        if not line:
            continue
        m = re.match(
            r"(\w+ \d+, \d+): (.+?) def\. (.+?) (\d+-\d+) \| Home team: (.+?) \| Arena: (.+)",
            line,
        )
        if not m:
            sentences.append(("", line))
            continue
        date, winner, loser, score, home, arena_city = m.groups()
        date_short = re.sub(r"\b0(\d)\b", r"\1", date.rsplit(",", 1)[0])
        arena_natural = arena_city.replace(", ", " in ")
        sentences.append((winner, f"On {date_short}, {winner} won {score} at {arena_natural}."))

    if not sentences:
        return ""
    result = [sentences[0][1]]
    first_winner = sentences[0][0]
    for winner, sentence in sentences[1:]:
        if winner and winner == first_winner:
            sentence = re.sub(
                r"^On (\S+ \d+), " + re.escape(winner),
                r"On \1, they",
                sentence,
            )
        result.append(sentence)
    return " ".join(result)


def _series_so_far_table(h2h_raw: str, home_full: str, away_full: str) -> str:
    """
    Build a 'Series So Far' markdown table from raw H2H bullet lines.
    Returns empty string if no completed games (Game 1).
    """
    lines = [l.strip() for l in h2h_raw.splitlines() if l.strip().startswith("•")]
    if not lines:
        return ""

    home_wins = away_wins = 0
    rows: list[str] = []

    for i, line in enumerate(lines, start=1):
        # • April 19, 2026: Cleveland Cavaliers def. Orlando Magic 112-98 | Home team: ... | Arena: ..., City
        m = re.match(
            r"•\s+(\w+ \d+),?\s+\d+: ([\w ]+?) def\. ([\w ]+?) (\d+-\d+)",
            line,
        )
        if not m:
            continue
        raw_date, winner, _, score = m.groups()
        # Short date: "April 19"
        date_short = re.sub(r"\b0(\d)\b", r"\1", raw_date)
        # Location from "Home team: X"
        loc_m = re.search(r"Home team: ([\w ]+)", line)
        location = loc_m.group(1).strip() if loc_m else "—"
        # Shorten to city (everything except last word)
        parts = location.split()
        location = " ".join(parts[:-1]) if len(parts) > 1 else location

        if winner.strip() == home_full:
            home_wins += 1
        else:
            away_wins += 1

        rows.append(f"| {i} | {date_short} | {winner.strip()} | {score} | {location} |")

    if not rows:
        return ""

    if home_wins > away_wins:
        header_leader = f"{home_full} leads {home_wins}-{away_wins}"
    elif away_wins > home_wins:
        header_leader = f"{away_full} leads {away_wins}-{home_wins}"
    else:
        header_leader = f"Series tied {home_wins}-{away_wins}"

    table_lines = [
        f"{header_leader}:",
        "",
        "| Game | Date | Winner | Score | Location |",
        "|------|------|--------|-------|----------|",
    ] + rows
    return "\n".join(table_lines)


def assemble_node(state: GraphState) -> dict:
    narrative    = state.get("narrative_section", "")
    injury_sum   = state.get("injury_summary", "")
    h2h_raw      = state.get("h2h_summary", "")
    table        = state.get("player_stats_table", "")
    query        = state.get("query", "")
    season_phase = state.get("season_phase", "regular")

    teams = extract_teams(query)
    if len(teams) >= 2:
        away_team, home_team = parse_home_away(query, teams)
        home_full, away_full = home_team[0], away_team[0]
    else:
        home_full = away_full = ""

    # Injury block
    if home_full and away_full:
        injury_block = (
            "Injury Report:\n\n"
            + _injury_line(injury_sum, home_full) + "\n\n"
            + _injury_line(injury_sum, away_full)
        )
    else:
        injury_block = "Injury Report:\n\nNo injury data."

    # H2H / Series block — varies by season phase
    if season_phase == "playoffs":
        series_table = _series_so_far_table(h2h_raw, home_full, away_full) if (home_full and away_full) else ""
        h2h_block = ("Series So Far:\n\n" + series_table) if series_table else ""
    else:
        h2h_block = "H2H This Season:\n\n" + (_h2h_prose(h2h_raw) if h2h_raw else "No completed H2H games this season.")

    parts = [narrative, injury_block]
    if h2h_block:
        parts.append(h2h_block)
    if table:
        parts.append(table)

    final = "\n\n".join(parts)
    print(f"[AssembleNode]      Final report ready ({len(final)} chars)")
    return {"final_report": final}
