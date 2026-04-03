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
    else:
        away_full = teams[0][0] if teams else "Visiting Team"
        home_full = "Home Team"

    authorized_names = extract_roster_names(table)
    roster_allowlist = "\n".join(f"  - {n}" for n in authorized_names) or "  (none parsed)"

    prompt = f"""You are a professional NBA broadcast statistician.
Write a pre-game broadcast briefing in ENGLISH — clean, spoken prose, TV-ready.

═══════════ MATCHUP ═══════════
{away_full} (AWAY)  at  {home_full} (HOME)
═══════════════════════════════

─── SECTION 1: OFFICIAL STATS (ESPN API) ───
{table}

─── SECTION 2: H2H THIS SEASON (ESPN schedule API — exact, verified) ───
{h2h_summary}

─── SECTION 3: AUTHORIZED PLAYER ROSTER ───
ONLY players listed here may be named in the report. Do NOT mention anyone else.
{roster_allowlist}

─── SECTION 4: NARRATIVE CONTEXT (Tavily — injuries + standout trends) ───
{bullets}

═══════════ OUTPUT RULES ═══════════

Write exactly 2 paragraphs, blank line between them. No headers.

PARAGRAPH 1 — {away_full} (visiting):
• First sentence ONLY: "{away_full} travel to {home_full}'s arena tonight."
  Never mention home/away again after this sentence.
• State exact W/L record and conference seed (from Section 1).
• State team PPG and Opp PPG with NBA rank (from Section 1 Team Stats line).
• State current win/loss streak and last-10 record (from Section 1).
• Weave in standout player trend (Section 4, Element B) — 1 sentence, authorized players only.
• Include any confirmed injuries (Section 4, Element A) — authorized players only.

PARAGRAPH 2 — {home_full} (home):
• Open directly with their record and seed. Do NOT restate home/away.
• State team PPG and Opp PPG with NBA rank.
• State current streak and last-10.
• Weave in standout player trend — 1 sentence, authorized players only.
• Include confirmed injuries — authorized players only.
• Weave in H2H context from Section 2: include exact scores, local dates, and home team
  for each game. Connect tonight's venue: e.g., "having won at home in the earlier meeting,
  they now host again" or "split the season series — once at each arena."

ABSOLUTE RULES:
1. ZERO source citations. ZERO URLs. ZERO parentheses containing sources.
   Pure spoken broadcast prose.
2. ROSTER INTEGRITY: Do NOT name any player absent from Section 3.
3. Home/away: sentence 1 of paragraph 1 only.
4. H2H: use Section 2 exclusively. Do not use any H2H claim from Section 4.
5. All numbers (W/L, PPG, seed, H2H scores) must match Sections 1 and 2 exactly.
6. PLAYER PARAGRAPH OWNERSHIP: A player belongs EXCLUSIVELY to their own team's paragraph.
   ZERO exceptions. Never name an opposing team's player for any reason —
   not for matchup context, not for comparison, not for defensive framing.
   Paragraph 1 contains ONLY {away_full} players. Paragraph 2 contains ONLY {home_full} players.
7. BANNED WORDS — these words must NOT appear anywhere in the output:
   "looking to", "look to", "aiming to", "hoping to", "momentum",
   "bounce back", "impressive", "strong", "resilience", "capitalize",
   "intensity", "will look", "seek to", "just", "only".
   Before submitting, scan every sentence for these words and remove them.
8. Tone: dry and factual — a stats sheet read aloud, not a feature article.

After the 2 paragraphs, reproduce Sections 1 stats tables EXACTLY — do not alter them:

{table}"""

    response = llm.invoke(prompt)
    report = response.content.strip()
    print(f"[ReportComposer]    Final report ready ({len(report)} chars)")
    return {"final_report": report}
