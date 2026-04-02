import os
import sys
import requests
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding="utf-8")
from typing import TypedDict
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START

load_dotenv()

CURRENT_SEASON       = "2025-26"
ESPN_SEASON_YEAR     = "2026"          # ESPN uses the end-year for season filtering
_EST                 = timezone(timedelta(hours=-5))   # Eastern Standard Time

# ---------------------------------------------------------------------------
# 1. Graph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    query: str
    player_stats_table: str    # ESPN API: W/L, seed, form, player PPG/FG%/MIN/RPG/APG
    h2h_summary: str           # ESPN API: completed H2H games with exact scores + venue
    team_narrative_bullets: str  # Tavily: injuries + standout player trend
    final_report: str


# ---------------------------------------------------------------------------
# 2. LLM + Tavily client factory
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def _tavily() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------

# ESPN team ID mapping — verified against live standings API
_ESPN_TEAMS: dict[str, tuple[str, str]] = {
    "Atlanta Hawks":          ("Hawks",         "1"),
    "Boston Celtics":         ("Celtics",        "2"),
    "Brooklyn Nets":          ("Nets",           "17"),
    "Charlotte Hornets":      ("Hornets",        "30"),
    "Chicago Bulls":          ("Bulls",          "4"),
    "Cleveland Cavaliers":    ("Cavaliers",      "5"),
    "Dallas Mavericks":       ("Mavericks",      "6"),
    "Denver Nuggets":         ("Nuggets",        "7"),
    "Detroit Pistons":        ("Pistons",        "8"),
    "Golden State Warriors":  ("Warriors",       "9"),
    "Houston Rockets":        ("Rockets",        "10"),
    "Indiana Pacers":         ("Pacers",         "11"),
    "LA Clippers":            ("Clippers",       "12"),
    "Los Angeles Lakers":     ("Lakers",         "13"),
    "Memphis Grizzlies":      ("Grizzlies",      "29"),
    "Miami Heat":             ("Heat",           "14"),
    "Milwaukee Bucks":        ("Bucks",          "15"),
    "Minnesota Timberwolves": ("Timberwolves",   "16"),
    "New Orleans Pelicans":   ("Pelicans",       "3"),
    "New York Knicks":        ("Knicks",         "18"),
    "Oklahoma City Thunder":  ("Thunder",        "25"),
    "Orlando Magic":          ("Magic",          "19"),
    "Philadelphia 76ers":     ("76ers",          "20"),
    "Phoenix Suns":           ("Suns",           "21"),
    "Portland Trail Blazers": ("Trail Blazers",  "22"),
    "Sacramento Kings":       ("Kings",          "23"),
    "San Antonio Spurs":      ("Spurs",          "24"),
    "Toronto Raptors":        ("Raptors",        "28"),
    "Utah Jazz":              ("Jazz",           "26"),
    "Washington Wizards":     ("Wizards",        "27"),
}


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd','th','th','th','th','th','th'][n % 10]}"


def extract_teams(query: str) -> list[tuple[str, str, str]]:
    """Return up to 2 (full_name, nickname, espn_id) tuples found in the query."""
    q = query.lower()
    found = []
    for full, (nick, eid) in _ESPN_TEAMS.items():
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
    Return (away_team, home_team).
    Rule 1: explicit "at <Team>" → that team is home.
    Rule 2: NBA broadcast convention — first team mentioned = away, second = home.
    """
    if len(teams) < 2:
        raise ValueError("Need exactly 2 teams to determine home/away.")
    t1, t2 = teams[0], teams[1]
    q = query.lower()
    for full, nick, eid in teams:
        if f"at {full.lower()}" in q or f"at {nick.lower()}" in q:
            home = (full, nick, eid)
            away = t2 if home == t1 else t1
            return away, home
    return t1, t2


def _extract_roster_names(table: str) -> list[str]:
    """Pull player names from the Markdown stats table for the composer allowlist."""
    names = []
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


# ---------------------------------------------------------------------------
# ESPN API helpers
# ---------------------------------------------------------------------------

_ESPN_STATS_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"
    "/statistics/byathlete?region=us&lang=en&contentorigin=espn"
    "&isqualified=true&sort=offensive.avgPoints%3Adesc&limit=500"
)
_ESPN_STANDINGS_URL = (
    "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
)


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
    ESPN JSON layout (verified):
      categories[0] = general   → totals[1]  = MIN,  totals[11] = RPG
      categories[1] = offensive → totals[0]  = PPG,  totals[3]  = FG%,
                                  totals[7]  = APG
    """
    data = _espn_fetch(_ESPN_STATS_URL)
    players = []
    for a in data.get("athletes", []):
        athlete = a.get("athlete", {})
        cats    = a.get("categories", [])
        if len(cats) < 2:
            continue
        try:
            ppg    = cats[1]["totals"][0]
            fg_pct = cats[1]["totals"][3]
            apg    = cats[1]["totals"][7]
            mins   = cats[0]["totals"][1]
            rpg    = cats[0]["totals"][11]
        except (IndexError, KeyError):
            continue
        players.append({
            "name":    athlete.get("displayName", "Unknown"),
            "team_id": athlete.get("teamId", ""),
            "ppg": ppg, "fg_pct": fg_pct,
            "apg": apg, "mins":   mins, "rpg": rpg,
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
    url  = (
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
        local_date = utc_dt.astimezone(_EST).strftime("%B %d, %Y")

        venue      = comp.get("venue", {})
        arena      = venue.get("fullName", "unknown arena")
        city       = venue.get("address", {}).get("city", "")

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
# 4. Nodes
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

        ppg_str  = f"{ppg:.1f} PPG ({_ordinal(ppg_r)} in NBA)" if ppg else "PPG N/A"
        def_str  = f"{opp_ppg:.1f} Opp PPG ({_ordinal(def_r)} in NBA)" if opp_ppg else "Def N/A"
        seed_str = f"#{seed} {conf.replace(' Conference','')}" if seed != "?" else "N/A"

        top5 = [p for p in all_players if p["team_id"] == espn_id][:5]

        rows = [
            f"### {full_name}",
            f"Record: W {wins} / L {losses} | Seed: {seed_str} | Streak: {streak} | Last 10: {l10}",
            f"Team Stats: {ppg_str} | {def_str}",
            "| Player | MIN | PPG | FG% | RPG | APG |",
            "|--------|-----|-----|-----|-----|-----|",
        ]
        if top5:
            for p in top5:
                rows.append(
                    f"| {p['name']} | {p['mins']} | {p['ppg']} "
                    f"| {p['fg_pct']} | {p['rpg']} | {p['apg']} |"
                )
        else:
            rows.append("| Stats unavailable | — | — | — | — | — |")

        sections.append("\n".join(rows))
        print(
            f"[DataSpecialist]    {full_name}: W{wins}/L{losses} | {seed_str} | "
            f"Streak:{streak} | L10:{l10} | {ppg_str} | {def_str} | {len(top5)} players"
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


def context_extractor_node(state: GraphState) -> dict:
    """
    Node 2 — Context Extractor (narrative only; H2H now handled by DataSpecialist).

    TWO Tavily searches:
      1. Injuries & lineup absences
      2. Standout player recent scoring trend (star players only)

    TWO extraction elements:
      A) Injuries & Absences
      B) Standout Player Trend — chain-of-thought: verbatim quote first, then number

    H2H data is no longer fetched here — it comes from the ESPN API via state["h2h_summary"].
    """
    query      = state["query"]
    today_iso  = datetime.now().strftime("%Y-%m-%d")
    today_long = datetime.now().strftime("%A, %d %B %Y")
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    teams = extract_teams(query)
    if len(teams) >= 2:
        away_team, home_team = parse_home_away(query, teams)
        away_full, away_nick = away_team[0], away_team[1]
        home_full, home_nick = home_team[0], home_team[1]
    else:
        away_full = away_nick = teams[0][0] if teams else "Team 1"
        home_full = home_nick = "Team 2"

    tavily = _tavily()
    search_kwargs = dict(
        max_results=4,
        include_domains=["espn.com", "nba.com", "statmuse.com", "basketball-reference.com"],
    )

    # Search 1 — Injuries & lineup absences
    q1 = f"{away_nick} {home_nick} injury report OUT questionable 2026"
    r1 = tavily.search(query=q1, **search_kwargs)

    # Search 2 — Standout player recent form (star players, last 5 games)
    q2 = (
        f"{away_nick} {home_nick} star player last 5 games "
        f"points scoring streak record 2026"
    )
    r2 = tavily.search(query=q2, **search_kwargs)

    print("[ContextExtractor]  === Tavily result titles (pre-LLM) ===")
    for label, res in [("Injuries", r1), ("Standout", r2)]:
        for item in res.get("results", []):
            print(f"  [{label}] {item.get('title', '(no title)')} | {item.get('url', '')}")
    print("[ContextExtractor]  =========================================")

    def fmt(res: dict) -> str:
        return "\n\n".join(
            f"[URL: {r['url']}]\nTitle: {r.get('title', '')}\n{r['content']}"
            for r in res.get("results", [])
        )

    combined_raw = (
        f"=== SEARCH 1: Injuries & Lineup News ===\n{fmt(r1)}\n\n"
        f"=== SEARCH 2: Standout Player Recent Form ===\n{fmt(r2)}"
    )

    system_prompt = (
        f"Today's date is {today_long} ({today_iso}). "
        f"The current NBA season is {CURRENT_SEASON} (started October 2025). "
        "You are a data analyst preparing narrative context for an NBA broadcast briefing. "
        "STRICT RULE: Use ONLY information from the provided search results below. "
        "Never recall or invent numbers from training memory."
    )

    user_prompt = f"""Matchup: {away_full} (AWAY) at {home_full} (HOME)

Live search results (fetched {fetched_at}):
{combined_raw}

Extract EXACTLY these two elements. Label each section clearly.

─── ELEMENT A — INJURIES & ABSENCES ───
For each confirmed injured or absent player on either team, state:
  team name, player name, status (OUT / DOUBTFUL / QUESTIONABLE), and the specific injury.
Skip vague "day-to-day" entries that do not name a specific injury.
One bullet per player. Each bullet ends with [Source: <url>].
If none confirmed: "No confirmed injury data found."

─── ELEMENT B — STANDOUT PLAYER RECENT TREND ───
Identify ONE player per team with a notable, INTERESTING recent trend over the last 3-5 games.
"Interesting" = a significant deviation from season average by a star player:
  - Hot streak by the team's #1 or #2 scorer (e.g., 40+ PPG over last 5 games).
  - Notable consecutive-game milestone (e.g., breaking a scoring record).
  - A sharp recent cold spell by a star (e.g., shooting 25% over last 4 games).
  - A bench player's trend is NOT interesting — focus on the team's star(s).

CHAIN-OF-THOUGHT EXTRACTION — follow this process for EVERY number you extract:
  Step 1: Copy the EXACT sentence from the search result text that contains the number,
          wrapped in double quotes.
  Step 2: Then state your extracted fact using only the numbers from that quoted sentence.

Example format:
  "Quoted sentence: '[exact text from result]'"
  Extracted fact: [player] has scored X points in Y consecutive games.

NUMBER ACCURACY RULE: The number in your extracted fact MUST exactly match the number
in the quoted sentence. Do not round, paraphrase, or substitute a different number.

STALENESS FILTER: Every search result has a publication date. Compare that date to
today's date ({today_iso}). If the article containing the standout stat is MORE THAN
2 DAYS OLD, treat it as stale and DO NOT include it. Write
"No recent standout data (source too old)" instead. We only want narratives that
reflect the player's form in the last 48 hours.

If nothing notable is confirmed for a team: "No standout trend confirmed for [team]."
One entry per team (quote + extracted fact). Each entry ends with [Source: <url>].

ABSOLUTE RULES:
- Every entry MUST end with [Source: <url>] from the results. Never invent a URL.
- Only quote sentences that actually appear verbatim in the search result text above."""

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    bullets = response.content.strip()
    bullets_with_ts = (
        f"[Fetched: {fetched_at} | Away: {away_full} | Home: {home_full}]\n"
        f"{bullets}"
    )
    print(f"[ContextExtractor]  Extracted {len(bullets.splitlines())} context lines")
    return {"team_narrative_bullets": bullets_with_ts}


def report_composer_node(state: GraphState) -> dict:
    """
    Node 3 — Report Composer.
    Combines:
      • player_stats_table  — ESPN API (W/L, seed, form, player stats)
      • h2h_summary         — ESPN API (exact scores, venues, dates)
      • team_narrative_bullets — Tavily (injuries + standout trend)

    Output: clean TV broadcast prose. Zero URLs, zero citations.
    Home/away stated once (sentence 1 only). Roster integrity enforced.
    """
    bullets     = state.get("team_narrative_bullets", "")
    table       = state.get("player_stats_table", "")
    h2h_summary = state.get("h2h_summary", "No H2H data available.")
    query       = state.get("query", "the requested matchup")

    teams = extract_teams(query)
    if len(teams) >= 2:
        away_team, home_team = parse_home_away(query, teams)
        away_full = away_team[0]
        home_full = home_team[0]
    else:
        away_full = teams[0][0] if teams else "Visiting Team"
        home_full = "Home Team"

    authorized_names = _extract_roster_names(table)
    roster_allowlist = "\n".join(f"  - {n}" for n in authorized_names) or "  (none parsed)"

    prompt = f"""You are a professional NBA broadcast statistician.
Write a pre-game broadcast briefing in ENGLISH — clean, spoken prose, TV-ready.

═══════════ MATCHUP ═══════════
{away_full} (AWAY)  at  {home_full} (HOME)
═══════════════════════════════

─── SECTION 1: OFFICIAL STATS (ESPN API) ───
{table}

─── SECTION 2: H2H THIS SEASON (ESPN schedule API — exact, verified) ───
{h2h_summary}

─── SECTION 3: AUTHORIZED PLAYER ROSTER ───
ONLY players listed here may be named in the report. Do NOT mention anyone else.
{roster_allowlist}

─── SECTION 4: NARRATIVE CONTEXT (Tavily — injuries + standout trends) ───
{bullets}

═══════════ OUTPUT RULES ═══════════

Write exactly 2 paragraphs, blank line between them. No headers.

PARAGRAPH 1 — {away_full} (visiting):
• First sentence ONLY: "{away_full} travel to {home_full}'s arena tonight."
  Never mention home/away again after this sentence.
• State exact W/L record and conference seed (from Section 1).
• State team PPG and Opp PPG with NBA rank (from Section 1 Team Stats line).
• State current win/loss streak and last-10 record (from Section 1).
• Weave in standout player trend (Section 4, Element B) — 1 sentence, authorized players only.
• Include any confirmed injuries (Section 4, Element A) — authorized players only.

PARAGRAPH 2 — {home_full} (home):
• Open directly with their record and seed. Do NOT restate home/away.
• State team PPG and Opp PPG with NBA rank.
• State current streak and last-10.
• Weave in standout player trend — 1 sentence, authorized players only.
• Include confirmed injuries — authorized players only.
• Weave in H2H context from Section 2: include exact scores, local dates, and home team
  for each game. Connect tonight's venue: e.g., "having won at home in the earlier meeting,
  they now host again" or "split the season series — once at each arena."

ABSOLUTE RULES:
1. ZERO source citations. ZERO URLs. ZERO parentheses containing sources.
   Pure spoken broadcast prose.
2. ROSTER INTEGRITY: Do NOT name any player absent from Section 3.
3. Home/away: sentence 1 of paragraph 1 only.
4. H2H: use Section 2 exclusively. Do not use any H2H claim from Section 4.
5. All numbers (W/L, PPG, seed, H2H scores) must match Sections 1 and 2 exactly.
6. PLAYER PARAGRAPH OWNERSHIP: A player belongs EXCLUSIVELY to their own team's paragraph.
   Never name a player from Team A inside Team B's paragraph, and vice versa.
   The only permitted exception is an explicit matchup-dynamic sentence
   (e.g., "{home_full} will need to contain [opposing player]") — and even then,
   only if that dynamic is directly supported by the narrative context provided.
7. BANNED: "looking to", "look to", "aiming to", "hoping to", "momentum",
   "bounce back", "impressive", "strong", "resilience", "capitalize",
   "intensity", "will look", "seek to".
8. Tone: dry and factual — a stats sheet read aloud, not a feature article.

After the 2 paragraphs, reproduce Sections 1 stats tables EXACTLY — do not alter them:

{table}"""

    response = llm.invoke(prompt)
    report = response.content.strip()
    print(f"[ReportComposer]    Final report ready ({len(report)} chars)")
    return {"final_report": report}


# ---------------------------------------------------------------------------
# 5. Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("data_specialist",   data_specialist_node)
    workflow.add_node("context_extractor", context_extractor_node)
    workflow.add_node("report_composer",   report_composer_node)

    workflow.add_edge(START, "data_specialist")
    workflow.add_edge(START, "context_extractor")

    workflow.add_edge("data_specialist",   "report_composer")
    workflow.add_edge("context_extractor", "report_composer")

    workflow.add_edge("report_composer", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# 6. Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    graph = build_graph()

    test_query = "Prepare a pre-game briefing for Los Angeles Lakers vs Oklahoma City Thunder"

    print("=" * 60)
    print("HoopsPrep — ESPN API Stats + Tavily Narrative Briefing")
    print("=" * 60)
    print(f"Query: {test_query}\n")

    initial_state: GraphState = {
        "query":                  test_query,
        "player_stats_table":     "",
        "h2h_summary":            "",
        "team_narrative_bullets": "",
        "final_report":           "",
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL REPORT:")
    print("=" * 60)
    print(result["final_report"])
