"""
core/state.py
─────────────
Shared state definition, constants, team registry, and query-parsing helpers
used by every node in the graph.
"""

from datetime import timezone, timedelta
from typing import TypedDict

# ---------------------------------------------------------------------------
# Season constants
# ---------------------------------------------------------------------------

CURRENT_SEASON   = "2025-26"
ESPN_SEASON_YEAR = "2026"          # ESPN uses the end-year for season filtering
EST              = timezone(timedelta(hours=-5))   # Eastern Standard Time

# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    query: str
    player_stats_table: str      # ESPN API: W/L, seed, form, player stats
    h2h_summary: str             # ESPN API: completed H2H games with scores + venue
    team_narrative_bullets: str  # Tavily: injuries + standout player trend
    final_report: str

# ---------------------------------------------------------------------------
# ESPN team registry  (full_name -> (nickname, espn_id))
# Verified against the live standings API
# ---------------------------------------------------------------------------

ESPN_TEAMS: dict[str, tuple[str, str]] = {
    "Atlanta Hawks":          ("Hawks",        "1"),
    "Boston Celtics":         ("Celtics",      "2"),
    "Brooklyn Nets":          ("Nets",         "17"),
    "Charlotte Hornets":      ("Hornets",      "30"),
    "Chicago Bulls":          ("Bulls",        "4"),
    "Cleveland Cavaliers":    ("Cavaliers",    "5"),
    "Dallas Mavericks":       ("Mavericks",    "6"),
    "Denver Nuggets":         ("Nuggets",      "7"),
    "Detroit Pistons":        ("Pistons",      "8"),
    "Golden State Warriors":  ("Warriors",     "9"),
    "Houston Rockets":        ("Rockets",      "10"),
    "Indiana Pacers":         ("Pacers",       "11"),
    "LA Clippers":            ("Clippers",     "12"),
    "Los Angeles Lakers":     ("Lakers",       "13"),
    "Memphis Grizzlies":      ("Grizzlies",    "29"),
    "Miami Heat":             ("Heat",         "14"),
    "Milwaukee Bucks":        ("Bucks",        "15"),
    "Minnesota Timberwolves": ("Timberwolves", "16"),
    "New Orleans Pelicans":   ("Pelicans",     "3"),
    "New York Knicks":        ("Knicks",       "18"),
    "Oklahoma City Thunder":  ("Thunder",      "25"),
    "Orlando Magic":          ("Magic",        "19"),
    "Philadelphia 76ers":     ("76ers",        "20"),
    "Phoenix Suns":           ("Suns",         "21"),
    "Portland Trail Blazers": ("Trail Blazers","22"),
    "Sacramento Kings":       ("Kings",        "23"),
    "San Antonio Spurs":      ("Spurs",        "24"),
    "Toronto Raptors":        ("Raptors",      "28"),
    "Utah Jazz":              ("Jazz",         "26"),
    "Washington Wizards":     ("Wizards",      "27"),
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def ordinal(n: int) -> str:
    """1 → '1st', 2 → '2nd', 3 → '3rd', 11 → '11th', etc."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th','th','th','th','th','th'][n % 10]}"


def extract_teams(query: str) -> list[tuple[str, str, str]]:
    """
    Return up to 2 (full_name, nickname, espn_id) tuples whose name or
    nickname appears in the query string.
    """
    q = query.lower()
    found: list[tuple[str, str, str]] = []
    for full, (nick, eid) in ESPN_TEAMS.items():
        if full.lower() in q or nick.lower() in q:
            found.append((full, nick, eid))
        if len(found) == 2:
            break
    return found


def parse_home_away(
    query: str,
    teams: list[tuple[str, str, str]],
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """
    Return (away_team, home_team) as (full_name, nickname, espn_id) tuples.

    Detection priority:
      1. Explicit "at <Team>" in the query → that team is home.
      2. NBA broadcast convention: first team mentioned = away, second = home.
    """
    if len(teams) < 2:
        raise ValueError("parse_home_away requires exactly 2 teams.")
    t1, t2 = teams[0], teams[1]
    q = query.lower()
    for full, nick, eid in teams:
        if f"at {full.lower()}" in q or f"at {nick.lower()}" in q:
            home = (full, nick, eid)
            away = t2 if home == t1 else t1
            return away, home
    return t1, t2   # default: first = away, second = home


def extract_roster_names(table: str) -> list[str]:
    """
    Parse player names out of the Markdown stats table produced by
    data_specialist_node, for use as the composer's roster allowlist.
    """
    names: list[str] = []
    for line in table.splitlines():
        if (
            line.startswith("| ")
            and not line.startswith("| Player")
            and not line.startswith("| Stats")
            and "---" not in line
        ):
            parts = [p.strip() for p in line.split("|")]
            name = parts[1] if len(parts) > 1 else ""
            if name and name not in ("Stats unavailable", "—", ""):
                names.append(name)
    return names
