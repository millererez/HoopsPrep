import os
import pandas as pd
from datetime import datetime
from typing import TypedDict
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Graph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    query: str
    player_stats_table: str
    team_narrative_bullets: str
    final_report: str


# ---------------------------------------------------------------------------
# 2. LLM
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------

def data_specialist_node(state: GraphState) -> dict:
    """
    Node 1 — Data Specialist.
    No LLM. Reads nba_players.csv directly, filters to teams mentioned in the
    query, sorts by PPG descending, and formats a Markdown table.
    """
    query = state["query"].lower()

    df = pd.read_csv("data/nba_players.csv")

    # Filter to only the teams mentioned in the query
    team_map = {
        "denver":    "Denver Nuggets",
        "minnesota": "Minnesota Timberwolves",
    }
    selected_teams = [full for keyword, full in team_map.items() if keyword in query]
    if selected_teams:
        df = df[df["team"].isin(selected_teams)]

    df = df.sort_values("ppg", ascending=False).reset_index(drop=True)

    sections = []
    for team in df["team"].unique():
        team_df = df[df["team"] == team]
        rows = [
            f"### {team}",
            "| Player | PPG | RPG | APG |",
            "|--------|-----|-----|-----|",
        ]
        for _, row in team_df.iterrows():
            rows.append(f"| {row['player']} | {row['ppg']} | {row['rpg']} | {row['apg']} |")
        sections.append("\n".join(rows))

    table = "\n\n".join(sections)
    print(f"[DataSpecialist]    Built {len(sections)} team tables ({len(df)} players, sorted by PPG)")
    return {"player_stats_table": table}


def context_extractor_node(state: GraphState) -> dict:
    """
    Node 2 — Context Extractor.
    Uses Tavily live web search to fetch current standings, streaks, and H2H
    data, then uses the LLM to distill results into factual bullet points.
    """
    query = state["query"]

    # Build a targeted search query from the user's request
    search_query = f"NBA {query} standings win loss streak head to head 2025-26 season"

    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = tavily.search(query=search_query, max_results=3)

    raw_text = "\n\n".join(
        f"Source: {r['url']}\n{r['content']}"
        for r in results.get("results", [])
    )

    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt = f"""You are a data analyst preparing inputs for a broadcast briefing.

Query: {query}

Live web search results (fetched {fetched_at}):
{raw_text}

Task: Extract ONLY the factual points from the search results that are relevant to the query.
Include ONLY: standings, win-loss records, winning/losing streaks, and head-to-head results.
Do NOT include: tactical analysis, player tendencies, coaching strategy, or style-of-play notes.
Format: one concise bullet point per fact. Output bullet points only — no headers, no extra commentary."""

    response = llm.invoke(prompt)
    bullets = response.content.strip()
    bullets_with_ts = f"[Data fetched: {fetched_at}]\n{bullets}"
    print(f"[ContextExtractor]  Extracted {len(bullets.splitlines())} narrative bullets (live search, {fetched_at})")
    return {"team_narrative_bullets": bullets_with_ts}


def report_composer_node(state: GraphState) -> dict:
    """
    Node 3 — Report Composer.
    Receives player_stats_table and team_narrative_bullets from the state.
    Writes exactly 2 paragraphs of professional pre-game briefing (facts only,
    no tactical analysis), then appends the stats table unchanged.
    """
    bullets = state.get("team_narrative_bullets", "")
    table   = state.get("player_stats_table", "")

    prompt = f"""You are a professional NBA broadcast writer. Produce a pre-game briefing in ENGLISH.

Factual data — use ONLY what is listed below, no outside knowledge:
{bullets}

Writing rules:
- Write exactly 2 paragraphs — no headers between them.
- Paragraph 1: Denver Nuggets. Cover their standing, record, and recent form.
- Paragraph 2: Minnesota Timberwolves. Cover their standing, record, recent form,
  and their head-to-head record against Denver this season.
- Do NOT include tactical analysis, player descriptions, or style-of-play commentary.
- Keep the tone professional and factual, suitable for a broadcast pre-game sheet.

After the 2 paragraphs, output the following player stats tables exactly as shown — do not alter them, do not merge them, do not add or remove columns:

{table}"""

    response = llm.invoke(prompt)
    report = response.content.strip()
    print(f"[ReportComposer]    Final report ready ({len(report)} chars)")
    return {"final_report": report}


# ---------------------------------------------------------------------------
# 4. Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("data_specialist",   data_specialist_node)
    workflow.add_node("context_extractor", context_extractor_node)
    workflow.add_node("report_composer",   report_composer_node)

    # Both specialist nodes run in parallel from START
    workflow.add_edge(START, "data_specialist")
    workflow.add_edge(START, "context_extractor")

    # LangGraph joins both branches before executing report_composer
    workflow.add_edge("data_specialist",   "report_composer")
    workflow.add_edge("context_extractor", "report_composer")

    workflow.add_edge("report_composer", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# 5. Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    graph = build_graph()

    test_query = "Prepare a pre-game briefing for Denver Nuggets vs Minnesota Timberwolves"

    print("=" * 60)
    print("HoopsPrep — Stable Trio Multi-Agent Briefing")
    print("=" * 60)
    print(f"Query: {test_query}\n")

    initial_state: GraphState = {
        "query":                  test_query,
        "player_stats_table":     "",
        "team_narrative_bullets": "",
        "final_report":           "",
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL REPORT:")
    print("=" * 60)
    print(result["final_report"])
