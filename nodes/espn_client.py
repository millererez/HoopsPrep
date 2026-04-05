"""
nodes/espn_client.py
────────────────────
Pure ESPN API layer — HTTP fetchers only, no domain logic, no LLM, no state.
"""

import requests
from datetime import datetime

from core.state import ESPN_SEASON_YEAR, EST

# ---------------------------------------------------------------------------
# ESPN API endpoints
# ---------------------------------------------------------------------------

_ESPN_STATS_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"
    "/statistics/byathlete?region=us&lang=en&contentorigin=espn"
    "&sort=offensive.avgPoints%3Adesc&limit=500"
)
_MIN_GAMES = 15  # minimum games played to appear in stats table
_ESPN_STANDINGS_URL = (
    "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
)


def espn_fetch(url: str) -> dict:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def build_standings_lookup() -> dict[str, dict]:
    """
    Return {espn_id: dict} for every team.
    Includes: wins, losses, ppg, opp_ppg, ppg_rank, def_rank,
              conf, conf_seed, streak, l10.
    """
    data = espn_fetch(_ESPN_STANDINGS_URL)
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


def build_player_lookup() -> list[dict]:
    """
    Fetch league-wide per-game stats (sorted by PPG descending).
    ESPN JSON layout (verified against live API):
      categories[0] = general   → totals[1]  = MIN,  totals[11] = RPG
      categories[1] = offensive → totals[0]  = PPG,  totals[3]  = FG%,
                                  totals[6]  = 3P%,  totals[7]  = APG,
                                  totals[9]  = FT%,  totals[11] = TOV
      categories[2] = defensive → totals[0]  = STL,  totals[1]  = BLK
    """
    data = espn_fetch(_ESPN_STATS_URL)
    players = []
    for a in data.get("athletes", []):
        athlete = a.get("athlete", {})
        cats    = a.get("categories", [])
        if len(cats) < 3:
            continue
        try:
            gp     = int(float(cats[0]["totals"][0]))
            mins   = cats[0]["totals"][1]
            rpg    = cats[0]["totals"][11]
            ppg    = cats[1]["totals"][0]
            fga    = cats[1]["totals"][2]
            fg_pct = cats[1]["totals"][3]
            tpa    = cats[1]["totals"][5]
            tpp    = cats[1]["totals"][6]
            apg    = cats[1]["totals"][7]
            ft_pct = cats[1]["totals"][9]
            tov    = cats[1]["totals"][11]
            stl    = cats[2]["totals"][0]
            blk    = cats[2]["totals"][1]
        except (IndexError, KeyError):
            continue
        # FTA index is uncertain — fetch separately so a wrong index doesn't drop the player
        try:
            fta = cats[1]["totals"][10]
        except (IndexError, KeyError):
            fta = "—"
        if gp < _MIN_GAMES:
            continue
        players.append({
            "name":    athlete.get("displayName", "Unknown"),
            "team_id": athlete.get("teamId", ""),
            "mins": mins, "ppg": ppg,
            "fga": fga, "fg_pct": fg_pct,
            "tpa": tpa, "tpp": tpp,
            "fta": fta, "ft_pct": ft_pct,
            "rpg": rpg, "apg": apg,
            "stl": stl, "blk": blk, "tov": tov,
        })
    return players


def fetch_h2h_games(
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
    data = espn_fetch(url)

    lines = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        if not comp["status"]["type"]["completed"]:
            continue
        competitor_ids = [c["team"]["id"] for c in comp["competitors"]]
        if t2_id not in competitor_ids:
            continue

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


def fetch_injuries(team_id: str, team_name: str) -> list[str]:
    """
    Fetch current injury list for a team from ESPN roster API.
    Returns list of strings: "PlayerName — STATUS"
    Only includes players with an active injury entry.
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
    data = espn_fetch(url)
    results = []
    for athlete in data.get("athletes", []):
        injuries = athlete.get("injuries", [])
        if not injuries:
            continue
        latest = max(injuries, key=lambda i: i.get("date", ""))
        status = latest.get("status", "")
        if status:
            results.append(f"{athlete['displayName']} \u2014 {status}")
    return results


def fetch_full_active_roster(team_id: str, injured_names: set[str]) -> list[str]:
    """
    Fetch all active (non-injured) players from the ESPN roster endpoint.
    Used when the qualified stats table is very short (≤6 players) to surface
    G League call-ups and 10-day contract players who have no qualifying stats.
    Returns a list of display names not already in the stats table.
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
    data = espn_fetch(url)
    active = []
    for athlete in data.get("athletes", []):
        name = athlete.get("displayName", "")
        if name and name not in injured_names:
            active.append(name)
    return active


def fetch_recent_form(
    team_id: str, team_name: str, n_games: int = 3
) -> list[str]:
    """
    Fetch the top scorer from each of the last n completed games for a team.
    Uses ESPN schedule + game summary APIs.
    Returns list of strings: "Date vs Opponent: PlayerName scored X pts (W/L score)"
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule?season={ESPN_SEASON_YEAR}"
    data = espn_fetch(url)

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

        team_score = opp_score = opp_name = ""
        result = ""
        for c in comp["competitors"]:
            if c["team"]["id"] == team_id:
                team_score = int(float(c["score"]["displayValue"]))
                result = "W" if c.get("winner") else "L"
            else:
                opp_score = int(float(c["score"]["displayValue"]))
                opp_name  = c["team"]["displayName"]

        try:
            summary = espn_fetch(
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
