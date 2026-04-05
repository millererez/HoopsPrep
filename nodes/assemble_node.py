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


def assemble_node(state: GraphState) -> dict:
    narrative   = state.get("narrative_section", "")
    injury_sum  = state.get("injury_summary", "")
    h2h_raw     = state.get("h2h_summary", "")
    table       = state.get("player_stats_table", "")
    query       = state.get("query", "")

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

    # H2H block
    h2h_block = "H2H This Season:\n\n" + (_h2h_prose(h2h_raw) if h2h_raw else "No completed H2H games this season.")

    parts = [narrative, injury_block, h2h_block]
    if table:
        parts.append(table)

    final = "\n\n".join(parts)
    print(f"[AssembleNode]      Final report ready ({len(final)} chars)")
    return {"final_report": final}
