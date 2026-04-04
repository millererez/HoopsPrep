"""
nodes/storylines_composer.py
────────────────────────────
Node 3 — Storylines Composer.
Queries ChromaDB for retrieved team context, generates the Storylines
section via LLM. Runs in parallel with report_composer (stage 2).
"""

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

    print(f"[StorylinesComposer] Retrieving context for {home_full} + {away_full} ...")
    home_ctx = _retrieve(home_full)
    away_ctx = _retrieve(away_full)

    if not home_ctx and not away_ctx:
        print("[StorylinesComposer] No ChromaDB content — skipping storylines.")
        return {"storylines_section": ""}

    prompt = f"""You are an NBA broadcast writer. Write a Storylines section for tonight's game.

MATCHUP: {home_full} (HOME) vs {away_full} (AWAY)

─── {home_full} CONTEXT (retrieved) ───
{home_ctx or "No context available."}

─── {away_full} CONTEXT (retrieved) ───
{away_ctx or "No context available."}

Write EXACTLY 3 bullets:
1. **{home_full}** — 2-3 sentences. Season arc, key player storyline, or what's at stake.
2. **{away_full}** — 2-3 sentences. Same format.
3. **Matchup Angle** — 1-2 sentences. What makes this specific game interesting. Contrast the two teams.

RULES:
- Use ONLY information from the context above. Do not invent stats or facts.
- If context is thin for a team, write fewer sentences but stay grounded.
- Dry, factual tone. No hype, no clichés.
- Forbidden phrases: "looking to", "hoping to", "aiming to", "bounce back", "capitalize",
  "impressive", "dominant", "make a statement", "showcasing", "potential".
- Output only the 3 bullets. No headers, no intro line."""

    print(f"[StorylinesComposer] Generating storylines ...")
    response = _get_llm().invoke([HumanMessage(content=prompt)])
    storylines = response.content.strip()
    print(f"[StorylinesComposer] Done ({len(storylines)} chars)")
    return {"storylines_section": storylines}
