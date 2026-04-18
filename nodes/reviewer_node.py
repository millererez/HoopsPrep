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
    narrative            = state.get("narrative_section", "")
    stakes_context       = state.get("stakes_context", "")
    injury_summary       = state.get("injury_summary", "")
    recent_form          = state.get("recent_form", "")
    player_team_map      = state.get("player_team_map", "")
    out_players_summary  = state.get("out_players_summary", "")
    season_phase         = state.get("season_phase", "regular")

    if not narrative.strip():
        return {"review_issues": ""}

    # Run Python banned-phrase check first — fast, free
    violations = find_violations(narrative)
    python_issues = (
        f"BANNED PHRASES detected by scanner: {', '.join(violations)}\n"
        if violations else ""
    )

    banned_str = " | ".join(BANNED_PHRASES)

    # ── Phase-aware CHECK 2 and CHECK 3 ─────────────────────────────────────
    if season_phase == "regular":
        check2 = """CHECK 2 — POSITION LABELS:
  PASS if: play-in teams (#7–#10 in STAKES CONTEXT) are not described as having a "playoff spot"
  or "making the playoffs" (play-in ≠ direct playoff berth).
  ISSUE if: a play-in team is described as securing or having a playoff spot. Quote the wrong
  sentence, then write a broadcast-ready replacement sentence using the exact play-in label from
  STAKES CONTEXT (e.g., "The [Team] are the play-in #9 seed, 3 games behind the #7 seed.").
  Format: ISSUE 2: "[wrong sentence]" → Replace with: "[your replacement sentence]"
  The replacement must be pure broadcast prose — no meta-commentary."""

        check3 = """CHECK 3 — STAKES NUMBERS:
  PASS if: paragraph 1 includes at least one specific games-back number for EACH team
  (e.g., "3.5 back of #4", "1.5 ahead of #9", "1 game back of #8"). Both teams must have a number — not just one.
  NOTE: a sentence like "The Trail Blazers are the play-in #9 seed, 1 game back of the #8 seed"
  DOES contain a games-back number ("1 game back") — do NOT flag it.
  If ISSUE 2 already fires for a team's sentence, do NOT also fire ISSUE 3 for the same sentence.
  ISSUE if: either team's playoff situation is described with no games-back number, even if the
  other team has one.
  Format: ISSUE 3: "[exact wrong sentence]" → Replace with: "[rewrite the sentence so it
  includes the games-back number from STAKES CONTEXT]"
  The replacement must be pure broadcast prose."""
    elif season_phase == "playin":
        # Play-in: seed positions locked, no games-back numbers.
        check2 = """CHECK 2 — POSITION LABELS (PLAY-IN):
  PASS if: paragraph 1 opener contains only records and seeds with NO positional gap comparison.
  A correct opener looks like: "The Charlotte Hornets (44-38, #9 East) host the Miami Heat (43-39, #10 East)."
  ISSUE if: the opener adds any positional comparison — "games back", "games ahead", "behind the #",
  "ahead of the #", or any gap between the two seeds. Quote the sentence.
  Format: ISSUE 2: "[wrong sentence]" → Replace with: "[opener using only records and seeds,
  e.g., 'The Charlotte Hornets (44-38, #9 East) host the Miami Heat (43-39, #10 East).']"
  Do NOT add any positional gap to the replacement."""

        check3 = """CHECK 3 — STAKES NUMBERS:
  SKIP — play-in games use outcome framing (who advances, who is eliminated),
  not games-back numbers. Do NOT flag the absence of games-back language. Output nothing for this check."""

    else:
        # Playoffs: opener must include game number and series score — no games-back language.
        check2 = """CHECK 2 — POSITION LABELS (PLAYOFFS):
  PASS if: paragraph 1 opener contains records, seeds, and game number. If it is Game 2 or later, it must also contain the series score. NO games-back comparison is allowed.
  A correct Game 1 opener looks like: "The Cleveland Cavaliers (64-18, #1 East) host the Orlando Magic (41-41, #8 East) in Game 1 of the first round."
  ISSUE if: the opener adds any regular-season positional comparison ("games back", "ahead of", etc.) or invents a series score for Game 1 (like "tied 0-0").
  Format: ISSUE 2: "[wrong sentence]" → Replace with: "[corrected opener]"
  CRITICAL: If the opener is correct, output EXACTLY AND ONLY "PASS". Do NOT output "No issue found, but...". Do NOT ask to add "0-0" to Game 1."""

        check3 = """CHECK 3 — STAKES NUMBERS:
  SKIP — playoff games use series-score framing (who leads the series, game number),
  not games-back numbers. Do NOT flag the absence of games-back language. Output nothing for this check."""

    # ─────────────────────────────────────────────────────────────────────────

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

─── PLAYER-TEAM ROSTER ───
{player_team_map or "Unavailable."}
Paragraph 2 = home team. Paragraph 3 = away team.

─── OUT PLAYERS (Python-verified — this is the only authoritative source) ───
{out_players_summary or "Unavailable."}
CRITICAL: only the players listed above as OUT are OUT. Any player NOT listed here is ACTIVE,
regardless of what you know from training data or any other source.

═══════════ CHECKLIST ═══════════

CHECK 1 — DTD NEGATIVE STATEMENT:
  PASS if: no paragraph contains a sentence that explicitly states a team has zero injuries,
  zero Day-To-Day players, or no injury concerns (e.g., "No injuries reported", "all players
  are available", "no injury concerns", "no Day-To-Day players").
  ISSUE if: you find such a sentence. Quote it exactly.
  NOTE: if you do NOT find such a sentence, output nothing for this check — do not comment on
  its absence. Silence on CHECK 1 = PASS.

{check2}

{check3}

CHECK 4 — ATTRIBUTION:
  Step 1: For each team, find the line starting with "Offensive engine:" or "Go-to scorer:"
  in that team's RECENT FORM block. Read the player name on that line. That name is the
  authoritative engine — do NOT substitute the season PPG leader, the season star, or any
  other player for it. The RECENT FORM block already accounts for season star overrides.
  Step 2: Verify the narrative credits that player with recent game scores.
  Step 3: Verify no player from one team's roster appears in the other team's paragraph
  (using PLAYER-TEAM ROSTER above).
  IMPORTANT: if the RECENT FORM block shows a player as engine AND that player is listed as
  OUT in the OUT PLAYERS list, then the replacement scorer (the next player in RECENT FORM) is
  the correct engine — do NOT flag the replacement as wrong.
  ISSUE if: specific recent game scores (e.g., "scored 30 against the Clippers") are
  attributed to a player who is not the engine or go-to scorer in RECENT FORM, OR a player
  from one team appears in the other team's paragraph.
  NOTE: mentioning a player's season PPG average is NOT an attribution error — only flag
  sentences that credit specific recent game scores to the wrong player.
  Quote the exact sentence, name the wrong player, name the correct player, and specify
  which paragraph (e.g., "In the Portland Trail Blazers paragraph, ...").

CHECK 5 — H2H IN NARRATIVE:
  PASS if: no paragraph references a regular-season prior meeting or regular-season game result.
  NOTE: Historical playoff meetings in paragraph 1 (e.g., "all-time playoff series", "last met in 2024", "first playoff meeting") are REQUIRED and ALLOWED. Do NOT flag playoff history.
  ISSUE if: any sentence mentions a specific regular-season game result. Quote it.

CHECK 6 — BANNED PHRASES:
  These phrases are forbidden: {banned_str}
  PASS if: none of these appear.
  ISSUE if: any appear. Quote the exact sentence containing the banned phrase.

CHECK 8 — OUT PLAYER FRAMING:
  Use ONLY the OUT PLAYERS list above — it is Python-verified and overrides everything else.
  Do not use the INJURY REPORT prose, your training knowledge, or any other source to determine
  who is OUT. If a player is not in the OUT PLAYERS list, they are ACTIVE — period.
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
    try:
        response = _get_llm().invoke([HumanMessage(content=prompt)])
        llm_output = response.content.strip()
    except Exception as e:
        print(f"[Reviewer] ❌ LLM API Error (Quota/Timeout): {e}")
        return {"review_issues": ""}

    # Python-guard: drop ISSUE 8 lines that name a player not in out_players_summary.
    # The reviewer LLM sometimes hallucinates OUT status from training knowledge.
    if llm_output != "APPROVED" and out_players_summary:
        out_names_lower: set[str] = set()
        for line in out_players_summary.splitlines():
            if "OUT:" in line:
                after = line.split("OUT:", 1)[1].strip()
                if after.lower() != "none":
                    for n in after.split(","):
                        out_names_lower.add(n.strip().lower())
        import re as _re2
        filtered: list[str] = []
        for line in llm_output.splitlines():
            if _re2.match(r'\s*ISSUE\s+(CHECK\s+)?8\s*:', line, _re2.IGNORECASE):
                if not any(name in line.lower() for name in out_names_lower):
                    print(f"[Reviewer]  Dropping false CHECK 8 (player not in OUT list): {line.strip()}")
                    continue
            filtered.append(line)
        llm_output = "\n".join(filtered).strip() or "APPROVED"

    # Python-guard: drop ISSUE 4 lines where the flagged sentence contains no specific
    # game score (e.g. "scored X against Y"). Season PPG averages are NOT attribution errors.
    import re as _re3
    if llm_output != "APPROVED":
        filtered4: list[str] = []
        for line in llm_output.splitlines():
            if _re3.match(r'\s*ISSUE\s+(CHECK\s+)?4\s*:', line, _re3.IGNORECASE):
                # Keep only if the quoted sentence contains a specific game score pattern
                if not _re3.search(r'scored\s+\d+|(\d+)\s+points?\s+(against|vs)', line, _re3.IGNORECASE):
                    print(f"[Reviewer]  Dropping false CHECK 4 (no game score in sentence): {line.strip()}")
                    continue
            filtered4.append(line)
        llm_output = "\n".join(filtered4).strip() or "APPROVED"

    # Python-guard: drop false ISSUE 7 (factless closing) when the quoted sentence
    # already contains both a player name (two+ capitalised words) and a number.
    import re as _re4
    if llm_output != "APPROVED":
        filtered7: list[str] = []
        for line in llm_output.splitlines():
            if _re4.match(r'\s*ISSUE\s+(CHECK\s+)?7\s*:', line, _re4.IGNORECASE):
                quoted = _re4.search(r'"([^"]+)"', line)
                if quoted:
                    sent = quoted.group(1)
                    has_number = bool(_re4.search(r'\d', sent))
                    has_name   = bool(_re4.search(r'[A-Z][a-z]+ [A-Z][a-z]+', sent))
                    if has_number and has_name:
                        print(f"[Reviewer]  Dropping false CHECK 7 (sentence has name + number): {line.strip()}")
                        continue
            filtered7.append(line)
        llm_output = "\n".join(filtered7).strip() or "APPROVED"

    if llm_output == "APPROVED" and not python_issues:
        print("[Reviewer]  APPROVED — no issues found")
        return {"review_issues": ""}

    issues = python_issues + (
        "" if llm_output == "APPROVED" else llm_output
    )
    print(f"[Reviewer]  Issues found — sending to rewrite:\n{issues}")
    return {"review_issues": issues}
