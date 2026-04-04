"""
nodes/storylines_composer.py
────────────────────────────
Node 3 — Storylines Composer.
Queries ChromaDB for retrieved team context, generates the Storylines
section via LLM. Runs in parallel with report_composer (stage 2).
"""

import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from core.state import GraphState, extract_teams, parse_home_away
from db.chroma import get_collection

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    return _llm

_N_RESULTS = 4


def _momentum_summary(table: str, team_name: str) -> str:
    """Extract record, seed, streak, last-10 for a team from the stats table."""
    m = re.search(
        rf"### {re.escape(team_name)}.*?Record: W (\d+) / L (\d+) \| Seed: ([^|]+)\| Streak: (\S+) \| Last 10: ([\d-]+)",
        table, re.DOTALL
    )
    if not m:
        return f"{team_name}: record unavailable"
    wins, losses, seed, streak, last10 = m.groups()
    seed = seed.strip().replace(" Conference", "").replace(" Eastern", " East").replace(" Western", " West")
    direction = "winning" if streak.startswith("W") else "losing"
    n = streak[1:]
    return f"{team_name}: {wins}-{losses}, {seed}, {direction} streak of {n}, {last10} last 10"


def _retrieve(team_name: str) -> str:
    try:
        results = get_collection().query(
            query_texts=[f"{team_name} NBA season storyline arc key players"],
            n_results=_N_RESULTS,
            where={"team_name": team_name},
        )
        docs = results.get("documents", [[]])[0]
        return "\n\n".join(docs) if docs else ""
    except Exception as exc:
        print(f"[StorylinesComposer] ChromaDB query failed for {team_name}: {exc}")
        return ""


def storylines_composer_node(state: GraphState) -> dict:
    query = state["query"]

    teams = extract_teams(query)
    if len(teams) < 2:
        return {"storylines_section": ""}

    away_team, home_team = parse_home_away(query, teams)
    home_full, away_full = home_team[0], away_team[0]

    table = state.get("player_stats_table", "")
    home_momentum = _momentum_summary(table, home_full)
    away_momentum = _momentum_summary(table, away_full)

    print(f"[StorylinesComposer] Retrieving context for {home_full} + {away_full} ...")
    home_ctx = _retrieve(home_full)
    away_ctx = _retrieve(away_full)

    print(f"[StorylinesComposer] === Retrieved chunks for {home_full} ===")
    print(home_ctx[:800] if home_ctx else "  (empty)")
    print(f"[StorylinesComposer] === Retrieved chunks for {away_full} ===")
    print(away_ctx[:800] if away_ctx else "  (empty)")

    if not home_ctx and not away_ctx:
        print("[StorylinesComposer] No ChromaDB content — skipping storylines.")
        return {"storylines_section": ""}

    prompt = f"""You are an NBA broadcast writer. Write a Storylines section for tonight's game.

MATCHUP: {home_full} (HOME) vs {away_full} (AWAY)

─── CURRENT STANDINGS (verified ESPN data — use these numbers, do not invent others) ───
{home_momentum}
{away_momentum}

─── {home_full} NARRATIVE CONTEXT (retrieved) ───
{home_ctx or "No context available."}

─── {away_full} NARRATIVE CONTEXT (retrieved) ───
{away_ctx or "No context available."}

Write EXACTLY 3 bullets:
1. **{home_full}** — 2-3 sentences.
2. **{away_full}** — 2-3 sentences.
3. **Matchup Angle** — 1-2 sentences. Name the specific players or factors that make this game interesting. No generic contrasts.

RULES:
- Do NOT restate records, streaks, or last-10 — those are already in the briefing above. The reader already knows them.
- Every sentence must contain at least one specific fact from the narrative context: a player name, a trade, a stat, a milestone, a result.
- When the narrative context contains specific numbers (PPG, win percentages, etc.), use them — do not replace with vague phrases like "several categories" or "league-leading".
- Records and streaks from CURRENT STANDINGS may be used only for framing momentum — not as the main point of a sentence.
- Narrative facts must come from NARRATIVE CONTEXT only. Do not invent.
- Name players explicitly — never "a veteran" or "a rookie" without a name.
- Dry, factual tone. No hype, no clichés.
- Forbidden: "looking to", "hoping to", "aiming to", "bounce back", "capitalize",
  "impressive", "dominant", "make a statement", "showcasing", "potential",
  "will look", "contributions", "struggling", "recent successes", "intrigue",
  "formidable", "remarkable", "contrasting momentum", "pivotal".
- Output only the 3 bullets. No headers, no intro line."""

    print(f"[StorylinesComposer] Generating storylines ...")
    response = _get_llm().invoke([HumanMessage(content=prompt)])
    storylines = response.content.strip()
    print(f"[StorylinesComposer] Done ({len(storylines)} chars)")
    return {"storylines_section": storylines}
