"""
nodes/context_extractor.py
──────────────────────────
Node 2 — Context Extractor (narrative only; H2H handled by DataSpecialist).

TWO Tavily searches:
  1. Injuries & lineup absences
  2. Standout player recent scoring trend (star players only)

TWO extraction elements:
  A) Injuries & Absences
  B) Standout Player Trend — chain-of-thought: verbatim quote first, then number
"""

import os
from datetime import datetime

from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.state import GraphState, CURRENT_SEASON, extract_teams, parse_home_away

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _tavily() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


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
        include_domains=["espn.com", "nba.com", "statmuse.com", "basketball-reference.com", "cbssports.com", "rotowire.com"],
    )

    # Search 1 — Today's official injury report only (days=1 cuts off stale articles at API level)
    q1 = f"{away_nick} {home_nick} official injury report today {today_iso}"
    r1 = tavily.search(query=q1, **search_kwargs, days=1)

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
STALENESS FILTER: Check the publication date of every injury result.
If the article is MORE THAN 1 DAY OLD (published before {today_iso} minus 1 day),
DISCARD it entirely — do not report any injury from that article.
Injury status changes game-to-game; a week-old report is meaningless and dangerous.

For each confirmed injured or absent player, output EXACTLY this format:
[TEAM: {away_full}] PlayerName — STATUS — injury
[TEAM: {home_full}] PlayerName — STATUS — injury
If no injury for a team: [TEAM: {away_full}] No confirmed injury data found.

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

For each team, output EXACTLY this format (including the [TEAM: ...] tag):
[TEAM: {away_full}] Quoted sentence: '...' | Extracted fact: ...  [Source: <url>]
[TEAM: {home_full}] Quoted sentence: '...' | Extracted fact: ...  [Source: <url>]

If nothing notable for a team: [TEAM: {away_full}] No standout trend confirmed.
If source is stale:            [TEAM: {away_full}] No recent standout data (source too old).

ABSOLUTE RULES:
- Every entry MUST have a [TEAM: exact_team_name] prefix. Never omit it.
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
