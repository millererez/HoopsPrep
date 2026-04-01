from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

from nba_tools import get_player_stats
from simple_rag import get_top_insights

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Graph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    user_query: str
    player_stats: str
    narrative_insights: list[str]
    final_report: str
    tools_needed: list[str]


# ---------------------------------------------------------------------------
# 2. LLM
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# 3. Nodes
# ---------------------------------------------------------------------------

def reasoning_node(state: GraphState) -> dict:
    """Analyze the user query and decide which tools are needed."""
    query = state["user_query"]

    prompt = f"""You are an NBA analytics assistant. Given the user query below, decide which data sources are needed.

Query: {query}

Available tools:
- "stats"      : Fetch live player statistics (PPG, RPG, APG) — use when specific players are mentioned.
- "narratives" : Fetch tactical/strategic insights via RAG — use when strategy, matchups, or team play are discussed.

Respond with a comma-separated list of required tool names only.
Examples: "stats,narratives"  |  "stats"  |  "narratives"
Output nothing else."""

    response = llm.invoke(prompt)
    raw = response.content.strip().lower()
    tools_needed = [t.strip() for t in raw.split(",") if t.strip() in ("stats", "narratives")]

    print(f"[Reasoning] Query     : {query}")
    print(f"[Reasoning] Tools     : {tools_needed}")

    return {"tools_needed": tools_needed}


def fetch_stats_node(state: GraphState) -> dict:
    """Fetch player statistics for every player mentioned in the query."""
    query = state["user_query"]

    # Use the LLM to extract player names from the query
    prompt = f"""Extract NBA player names (in English) from this query.
Query: {query}
Output: comma-separated English player names only. Nothing else.
Example: "Nikola Jokic,Anthony Edwards" """

    response = llm.invoke(prompt)
    players = [p.strip() for p in response.content.strip().split(",") if p.strip()]

    lines = []
    for player in players:
        result = get_player_stats(player)
        lines.append(result)
        print(f"[Stats] {result}")

    return {"player_stats": "\n".join(lines)}


def fetch_narratives_node(state: GraphState) -> dict:
    """Retrieve relevant tactical insights from the vector store."""
    query = state["user_query"]

    prompt = f"""Convert this NBA query to a concise English search phrase for semantic retrieval.
Query: {query}
Output: English search phrase only."""

    response = llm.invoke(prompt)
    english_query = response.content.strip()

    insights = get_top_insights(english_query, k=3)
    print(f"[Narratives] Search   : {english_query}")
    print(f"[Narratives] Retrieved: {len(insights)} insights")

    return {"narrative_insights": insights}


def generator_node(state: GraphState) -> dict:
    """Synthesize all gathered data into a final English broadcast report."""
    query       = state.get("user_query", "")
    stats       = state.get("player_stats", "")
    narratives  = state.get("narrative_insights", [])

    narratives_text = "\n".join(f"- {n}" for n in narratives) if narratives else "No additional insights."
    stats_text      = stats if stats else "No stats found."

    prompt = f"""You are a professional NBA broadcast writer. Write a comprehensive broadcast report in English based on the data below.

User query:
{query}

Player statistics:
{stats_text}

Tactical insights:
{narratives_text}

Writing guidelines:
- Write in clear, engaging English.
- Open with a short summary of the matchup.
- Analyze each player's strengths and their impact on the game.
- Describe the tactical dynamics between the teams.
- Close with concrete strategic recommendations for Denver."""

    response = llm.invoke(prompt)
    report = response.content
    print(f"\n[Generator] Report ready ({len(report)} chars)")

    return {"final_report": report}


# ---------------------------------------------------------------------------
# 4. Routing
# ---------------------------------------------------------------------------

def route_after_reasoning(state: GraphState) -> list[str]:
    """Fan-out to whichever tool nodes the reasoning step requested."""
    tools = state.get("tools_needed", [])
    destinations = [t for t in ("fetch_stats", "fetch_narratives") if t.replace("fetch_", "") in tools]
    return destinations if destinations else ["generator"]


# ---------------------------------------------------------------------------
# 5. Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("reasoning",         reasoning_node)
    workflow.add_node("fetch_stats",       fetch_stats_node)
    workflow.add_node("fetch_narratives",  fetch_narratives_node)
    workflow.add_node("generator",         generator_node)

    # START → reasoning
    workflow.add_edge(START, "reasoning")

    # reasoning → (fetch_stats and/or fetch_narratives) or directly generator
    workflow.add_conditional_edges(
        "reasoning",
        route_after_reasoning,
        ["fetch_stats", "fetch_narratives", "generator"],
    )

    # Both tool nodes converge on generator (LangGraph joins when all branches complete)
    workflow.add_edge("fetch_stats",      "generator")
    workflow.add_edge("fetch_narratives", "generator")

    # generator → END
    workflow.add_edge("generator", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# 6. Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    graph = build_graph()

    test_query = "Analyze the matchup between Nikola Jokic and Anthony Edwards. Provide their key stats and explain the tactical approach Denver should take to win."

    print("=" * 60)
    print("NBA Broadcast Agent — LangGraph Brain")
    print("=" * 60)
    print(f"Query: {test_query}\n")

    initial_state = {
        "user_query":         test_query,
        "player_stats":       "",
        "narrative_insights": [],
        "final_report":       "",
        "tools_needed":       [],
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL REPORT:")
    print("=" * 60)
    print(result["final_report"])
