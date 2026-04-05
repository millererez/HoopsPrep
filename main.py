"""
main.py
───────
HoopsPrep entry point.
Wires the LangGraph, loads env, and runs a test query.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END, START

from core.state import GraphState
from nodes.data_specialist import data_specialist_node
from nodes.context_extractor import context_extractor_node
from nodes.narrative_composer import narrative_composer_node
from nodes.reviewer_node import reviewer_node
from nodes.rewrite_node import rewrite_node
from nodes.assemble_node import assemble_node


def _route_after_review(state: GraphState) -> str:
    return "rewrite_node" if state.get("review_issues", "").strip() else "assemble_node"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("data_specialist",    data_specialist_node)
    workflow.add_node("context_extractor",  context_extractor_node)
    workflow.add_node("narrative_composer", narrative_composer_node)
    workflow.add_node("reviewer",           reviewer_node)
    workflow.add_node("rewrite_node",       rewrite_node)
    workflow.add_node("assemble_node",      assemble_node)

    # Stage 1 — parallel
    workflow.add_edge(START, "data_specialist")
    workflow.add_edge(START, "context_extractor")

    # Stage 2 — fan-in: narrative_composer waits for both stage-1 nodes
    workflow.add_edge("data_specialist",   "narrative_composer")
    workflow.add_edge("context_extractor", "narrative_composer")

    # Stage 3 — reviewer checks, routes to rewrite or directly to assemble
    workflow.add_edge("narrative_composer", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {"rewrite_node": "rewrite_node", "assemble_node": "assemble_node"},
    )

    # Stage 4 — rewrite goes directly to assemble (never back to reviewer)
    workflow.add_edge("rewrite_node", "assemble_node")
    workflow.add_edge("assemble_node", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()

    test_query = "Prepare a pre-game briefing for Philadelphia 76ers vs Minnesota Timberwolves"

    print("=" * 60)
    print("HoopsPrep — ESPN API Stats + Tavily Narrative Briefing")
    print("=" * 60)
    print(f"Query: {test_query}\n")

    initial_state: GraphState = {
        "query":                  test_query,
        "player_stats_table":     "",
        "h2h_summary":            "",
        "injury_summary":         "",
        "recent_form":            "",
        "team_narrative_bullets": "",
        "stakes_context":         "",
        "narrative_section":      "",
        "review_issues":          "",
        "final_report":           "",
    }

    result = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL REPORT:")
    print("=" * 60)
    print(result["final_report"])
