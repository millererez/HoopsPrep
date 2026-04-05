"""
nodes/reviewer_node.py
──────────────────────
Node 3 — Reviewer (checker only).
Runs a structured checklist against the narrative_composer output.
Returns a list of specific issues found, or empty string if approved.
Does NOT rewrite — issues are sent to rewrite_node for fixing.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from core.state import GraphState
from nodes.utils import BANNED_PHRASES, find_violations

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    return _llm


def reviewer_node(state: GraphState) -> dict:
    narrative      = state.get("narrative_section", "")
    stakes_context = state.get("stakes_context", "")
    injury_summary = state.get("injury_summary", "")
    recent_form    = state.get("recent_form", "")

    if not narrative.strip():
        return {"review_issues": ""}

    # Run Python banned-phrase check first — fast, free
    violations = find_violations(narrative)
    python_issues = (
        f"BANNED PHRASES detected by scanner: {', '.join(violations)}\n"
        if violations else ""
    )

    banned_str = " | ".join(BANNED_PHRASES)

    prompt = f"""You are a fact-checker reviewing a pre-game NBA briefing.
Your ONLY job is to IDENTIFY problems — do not fix or rewrite anything.

For each check below, state either PASS or ISSUE.
If ISSUE: quote the exact failing sentence and describe what is wrong.
Be specific — the writer needs enough information to fix it without seeing this prompt.

═══════════ NARRATIVE TO REVIEW ═══════════
{narrative}

═══════════ GROUNDING DATA ═══════════

─── STAKES CONTEXT ───
{stakes_context or "Unavailable."}

─── INJURY REPORT ───
{injury_summary or "Unavailable."}

─── RECENT FORM ───
{recent_form or "Unavailable."}

═══════════ CHECKLIST ═══════════

CHECK 1 — DTD NEGATIVE STATEMENT:
  PASS if: no paragraph contains a sentence that explicitly states a team has zero injuries,
  zero Day-To-Day players, or no injury concerns (e.g., "No injuries reported", "all players
  are available", "no injury concerns", "no Day-To-Day players").
  ISSUE if: you find such a sentence. Quote it exactly.
  NOTE: if you do NOT find such a sentence, output nothing for this check — do not comment on
  its absence. Silence on CHECK 1 = PASS.

CHECK 2 — POSITION LABELS:
  PASS if: play-in teams (#7–#10 in STAKES CONTEXT) are not described as having a "playoff spot"
  or "making the playoffs" (play-in ≠ direct playoff berth).
  ISSUE if: a play-in team is described as securing or having a playoff spot. Quote the sentence
  and state the correct label from STAKES CONTEXT.

CHECK 3 — STAKES NUMBERS:
  PASS if: paragraph 1 includes at least one specific games-back number for EACH team
  (e.g., "3.5 back of #4", "1.5 ahead of #9"). Both teams must have a number — not just one.
  ISSUE if: either team's playoff situation is described with no games-back number, even if the
  other team has one. State which team is missing the number and what number STAKES CONTEXT provides.

CHECK 4 — ATTRIBUTION:
  PASS if: the player credited with specific recent game scores in paragraphs 2–3 matches
  the "Offensive engine" or "Go-to scorer" in the RECENT FORM block for that team.
  IMPORTANT: if the RECENT FORM block shows a player as engine AND that player is listed as
  OUT in the INJURY REPORT, then the replacement scorer (the next player in RECENT FORM) is
  the correct engine — do NOT flag the replacement as wrong.
  ISSUE if: game scores are attributed to a player who is not the engine or go-to scorer
  in RECENT FORM. Quote the sentence and state the correct player name from RECENT FORM.

CHECK 5 — H2H IN NARRATIVE:
  PASS if: no paragraph references a prior meeting or previous game result between the teams.
  ISSUE if: any sentence mentions the previous meeting score or outcome. Quote it.

CHECK 6 — BANNED PHRASES:
  These phrases are forbidden: {banned_str}
  PASS if: none of these appear.
  ISSUE if: any appear. Quote the exact sentence containing the banned phrase.

CHECK 8 — OUT PLAYER FRAMING:
  From INJURY REPORT, read the exact names listed as OUT (not Day-To-Day). Only those exact
  names are OUT players — do not infer or assume any other player is OUT.
  ALLOWED FRAMING: A sentence of the form "With [OUT player] sidelined, [active player] has
  taken over / stepped up / leads the offense" is PERMITTED — this is the correct absence
  framing. Do NOT flag it. The OUT player is not described as contributing tonight; the
  sentence explicitly transfers the role to an active player.
  PASS if: the OUT player appears only in an absence/sidelined sentence (as above) OR is
  not mentioned at all.
  ISSUE if: an OUT player (by exact name from INJURY REPORT) is described as scoring,
  contributing, being the engine, or influencing tonight's game in any sentence that does
  NOT contain the word "sidelined", "out", or "unavailable". Quote the exact sentence and
  name the OUT player. Do NOT flag active players as OUT.

CHECK 7 — FACTLESS CLOSING:
  PASS if: the final sentence of paragraphs 2 and 3 each contain a player name AND a specific
  number (stat, score, or percentage).
  ISSUE if: a closing sentence is missing a player name or a number — descriptive adjectives
  ("strong", "critical", "pivotal", "dominant") do not count as facts. Quote the sentence and
  state which element is missing (player name, number, or both).

═══════════ OUTPUT FORMAT ═══════════
List only the issues found, one per line, in this format:
  ISSUE [check number]: [exact quoted sentence] → [what to fix]

If no issues found, output exactly: APPROVED

No other text. No PASS lines. Only issues or APPROVED."""

    print("[Reviewer]  Running checklist ...")
    response = _get_llm().invoke([HumanMessage(content=prompt)])
    llm_output = response.content.strip()

    if llm_output == "APPROVED" and not python_issues:
        print("[Reviewer]  APPROVED — no issues found")
        return {"review_issues": ""}

    issues = python_issues + (
        "" if llm_output == "APPROVED" else llm_output
    )
    print(f"[Reviewer]  Issues found — sending to rewrite:\n{issues}")
    return {"review_issues": issues}
