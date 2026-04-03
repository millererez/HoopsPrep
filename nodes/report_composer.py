"""
nodes/report_composer.py
────────────────────────
Node 3 — Report Composer.
Combines ESPN API stats, ESPN H2H data, and Tavily narrative context
into a clean TV broadcast briefing.
"""

from langchain_openai import ChatOpenAI

from core.state import GraphState, extract_teams, parse_home_away, extract_roster_names

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def _bullets_for_team(bullets: str, team_full: str) -> str:
    lines = bullets.splitlines()
    return "\n".join(
        line for line in lines
        if f"[TEAM: {team_full}]" in line or line.startswith("[Fetched")
    )

def report_composer_node(state: GraphState) -> dict:
    """
    Node 3 — Report Composer.
    Combines:
      • player_stats_table     — ESPN API (W/L, seed, form, player stats)
      • h2h_summary            — ESPN API (exact scores, venues, dates)
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
    elif len(teams) == 1:
        away_full = teams[0][0]
        home_full = "Home Team"
    else:
        away_full = "Visiting Team"
        home_full = "Home Team"

    away_bullets = _bullets_for_team(bullets, away_full)
    home_bullets = _bullets_for_team(bullets, home_full)

    authorized_names = extract_roster_names(table)
    roster_allowlist = "\n".join(f"  - {n}" for n in authorized_names) or "  (none parsed)"

    prompt = f"""You are a professional NBA broadcast statistician.
Write a pre-game broadcast briefing in ENGLISH — clean, spoken prose, TV-ready.

═══════════ MATCHUP ═══════════
{home_full} (HOME)  vs  {away_full} (AWAY)
═══════════════════════════════

─── SECTION 1: OFFICIAL STATS (ESPN API) ───
{table}

─── SECTION 2: H2H THIS SEASON (ESPN schedule API — exact, verified) ───
{h2h_summary}

─── SECTION 3: AUTHORIZED PLAYER ROSTER ───
ONLY players listed here may be named in the report. Do NOT mention anyone else.
{roster_allowlist}

─── SECTION 4A: NARRATIVE CONTEXT — {home_full} ONLY ───
{home_bullets}

─── SECTION 4B: NARRATIVE CONTEXT — {away_full} ONLY ───
{away_bullets}

═══════════ OUTPUT RULES ═══════════

Write exactly 2 paragraphs, blank line between them. No headers.

PARAGRAPH 1 — {home_full} (HOME). Use ONLY Section 4A. Do not use anything from 4B.
• Opening sentence: naturally merge the hosting context with their record and seed in one sentence.
  Example: "The {home_full} (49-28, #3 East) host {away_full} tonight, averaging X points per game..."
  Never mention home/away again after this sentence.
• State team PPG and Opp PPG with NBA rank (from Section 1 Team Stats line).
• State current win/loss streak and last-10 record (from Section 1).
• If there is a confirmed standout player trend (Section 4A, Element B): weave in 1 sentence, authorized players only.
  If there is NO standout trend: omit entirely — do not write any sentence about it.
• If there are confirmed injuries (Section 4A, Element A): state player name, status, and body part only — no procedure details.
  If no confirmed injuries: omit entirely — do not write any sentence about it.
• Weave in H2H context from Section 2: include exact scores, local dates, and home team
  for each game. Connect tonight's venue.

PARAGRAPH 2 — {away_full} (AWAY). Use ONLY Section 4B. Do not use anything from 4A.
• Open directly with their record and seed. Do NOT restate home/away.
• State team PPG and Opp PPG with NBA rank.
• State current streak and last-10.
• If there is a confirmed standout player trend (Section 4B, Element B): weave in 1 sentence, authorized players only.
  If there is NO standout trend: omit entirely — do not write any sentence about it.
• If there are confirmed injuries (Section 4B, Element A): state player name, status, and body part only — no procedure details.
  If no confirmed injuries: omit entirely — do not write any sentence about it.

ABSOLUTE RULES:
1. ZERO source citations. ZERO URLs. ZERO parentheses containing sources.
   Pure spoken broadcast prose.
2. ROSTER INTEGRITY: Do NOT name any player absent from Section 3.
   If Section 4A or 4B contains no injury data, write "No confirmed injuries" — do NOT recall from memory.
3. Home/away: sentence 1 of paragraph 1 only.
4. H2H: use Section 2 exclusively. Do not use any H2H claim from Section 4.
   Copy arena names VERBATIM from Section 2. Never substitute from memory.
5. All numbers (W/L, PPG, seed, H2H scores) must match Sections 1 and 2 exactly.
6. PLAYER PARAGRAPH OWNERSHIP — HARD WALL:
   PARAGRAPH 1 contains ONLY {home_full} players. PARAGRAPH 2 contains ONLY {away_full} players.
   ZERO exceptions. Never name an opposing player for any reason.
7. BANNED WORDS — these words must NOT appear anywhere in the output:
   "looking to", "look to", "aiming to", "hoping to", "momentum",
   "bounce back", "impressive", "strong", "resilience", "capitalize",
   "intensity", "will look", "seek to", "just", "only".
   Before submitting, scan every sentence for these words and remove them.
8. Tone: dry and factual — a stats sheet read aloud, not a feature article.

Write ONLY the 2 paragraphs. Do not reproduce any tables."""

    response = llm.invoke(prompt)
    report = response.content.strip() + "\n\n" + table
    print(f"[ReportComposer]    Final report ready ({len(report)} chars)")
    return {"final_report": report}


