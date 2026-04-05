"""
nodes/rewrite_node.py
─────────────────────
Node 4 — Rewrite.
Receives the narrative + specific issues from the reviewer.
Fixes exactly those issues using full grounding context from state.
Runs once — routes directly to assemble_node, never back to reviewer.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from core.state import GraphState
from nodes.utils import find_violations

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    return _llm


def rewrite_node(state: GraphState) -> dict:
    narrative      = state.get("narrative_section", "")
    review_issues  = state.get("review_issues", "")
    stakes_context = state.get("stakes_context", "")
    injury_summary = state.get("injury_summary", "")
    recent_form    = state.get("recent_form", "")
    h2h_summary    = state.get("h2h_summary", "")
    table          = state.get("player_stats_table", "")

    if not review_issues.strip():
        return {"narrative_section": narrative, "review_issues": ""}

    prompt = f"""You are rewriting a pre-game NBA briefing to fix specific errors.

RULE: Fix ONLY the sentences listed in ISSUES. Every other sentence must be copied
character-for-character. Do not rephrase, improve, or restructure anything else.

═══════════ ORIGINAL NARRATIVE ═══════════
{narrative}

═══════════ ISSUES TO FIX ═══════════
{review_issues}

═══════════ GROUNDING DATA (use to fix issues — do not use to add new content) ═══════════

─── STAKES CONTEXT ───
{stakes_context or "Unavailable."}

─── INJURY REPORT ───
{injury_summary or "Unavailable."}

─── RECENT FORM ───
{recent_form or "Unavailable."}

─── H2H THIS SEASON ───
{h2h_summary or "Unavailable."}

─── PLAYER STATS ───
{table or "Unavailable."}

═══════════ BANNED PHRASES ═══════════
After fixing the listed issues, scan every sentence for these banned phrases and rewrite
any sentence that contains one — even if it was not listed in ISSUES:
  look to | will look | aim to | aims to | seek to | seeking to | will be critical |
  will be crucial | will be essential | will be vital | been crucial | is crucial |
  are crucial | highlighting | contributions | has been a | relying on | consistent |
  which ranks | maintain their momentum | significant contributor | key contributor

═══════════ OUTPUT ═══════════
Return the complete corrected 4-paragraph text.
No commentary, no explanations. Just the 4 paragraphs."""

    print(f"[Rewriter]  Fixing {review_issues.count('ISSUE')} issue(s) ...")
    response = _get_llm().invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip()

    violations = find_violations(rewritten)
    if violations:
        print(f"[Rewriter]  WARNING — banned phrases remain: {violations}")
    else:
        print(f"[Rewriter]  Done ({len(rewritten)} chars)")

    return {"narrative_section": rewritten, "review_issues": ""}
