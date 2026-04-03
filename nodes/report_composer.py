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

def _injury_names_for_team(bullets: str, team_full: str) -> list[str]:
    """Extract player names from Element A injury lines for a specific team."""
    import re
    names = []
    for line in bullets.splitlines():
        if f"[TEAM: {team_full}]" not in line:
            continue
        # Skip non-injury lines (standout trends, no-data lines)
        if "No confirmed" in line or "No recent" in line or "Quoted sentence" in line:
            continue
        # Pattern: [TEAM: X] PlayerName — STATUS — ...
        m = re.search(rf"\[TEAM: {re.escape(team_full)}\]\s+([^—]+)—", line)
        if m:
            name = m.group(1).strip()
            if name:
                names.append(name)
    return names

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
    table          = state.get("player_stats_table", "")
    h2h_summary    = state.get("h2h_summary", "No H2H data available.")
    injury_summary = state.get("injury_summary", "")
    recent_form    = state.get("recent_form", "")
    query          = state.get("query", "the requested matchup")

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

─── SECTION 3: AUTHORIZED STATS ROSTER ───
These players may be named for performance/stats mentions.
{roster_allowlist}

─── SECTION 4: INJURY REPORT (ESPN API — current) ───
{injury_summary}

─── SECTION 5: RECENT FORM — last 3 games top scorer per team ───
{recent_form}

═══════════ OUTPUT RULES ═══════════

Write exactly 3 paragraphs, blank line between them. No headers.

PARAGRAPH 1 — {home_full} (HOME):
• Opening sentence: MUST include both the hosting context AND their record and seed, merged naturally.
  Example: "The {home_full} (25-51, #11 West) host the {away_full} tonight..."
  This sentence is mandatory — never skip the hosting reference.
  Never mention home/away again after this sentence.
• State team PPG and Opp PPG with NBA rank (from Section 1 Team Stats line).
• State current win/loss streak and last-10 record (from Section 1).
• Weave in 1–2 sentences about their recent form using Section 5.
  Guidelines for natural phrasing:
  - If one player leads scoring in 5+ of 10 games: acknowledge their offensive role naturally.
    Example: "Doncic has been the engine offensively, shouldering the scoring load throughout the stretch."
  - If one player leads in 3–4 of 10: note them as a key contributor without overstating.
    Example: "Barrett has emerged as Toronto's go-to scorer in recent weeks, leading the team on five occasions."
  - If scoring is truly spread (no player leads more than 2–3 times): describe the collective trend.
    Example: "Memphis has spread its offense across the roster with no single player establishing a consistent rhythm."
  - If there is a hot/cold arc in the win/loss column: describe the direction.
  Never write mechanical counts like "X led scoring in N of N games."
  Use ONLY players in Section 3. Do not invent or recall from memory.
• Weave in H2H context from Section 2: include exact scores, local dates, and home team for each game.

PARAGRAPH 2 — {away_full} (AWAY):
• Open directly with their record and seed. Do NOT restate home/away.
• State team PPG and Opp PPG with NBA rank.
• State current streak and last-10.
• Weave in 1–2 sentences about their recent form using Section 5.
  - If one player leads scoring in 5+ of 10 games: acknowledge their offensive role naturally.
  - If one player leads in 3–4 of 10: note them as a key contributor without overstating.
  - If scoring is truly spread: describe the collective trend.
  - If there is a hot/cold arc: describe the direction.
  Never write mechanical counts like "X led scoring in N of N games."
  Use ONLY players in Section 3. Do not invent or recall from memory.

PARAGRAPH 3 — INJURY REPORT:
Write exactly three lines, nothing more:
  Line 1: "Injury Report:"
  Line 2: "{home_full} — [player] (OUT), [player] (Day-To-Day), ..." or "{home_full} — None reported."
  Line 3: "{away_full} — [player] (OUT), [player] (Day-To-Day), ..." or "{away_full} — None reported."
Use ONLY players listed in Section 4. Do NOT recall from memory. Do NOT add any other text.

ABSOLUTE RULES:
1. ZERO source citations. ZERO URLs. ZERO parentheses containing sources.
   Pure spoken broadcast prose.
2. ROSTER INTEGRITY:
   • Performance/stats/recent form mentions: player must be in Section 3.
   • Injury mentions: player must appear in Section 4. Do NOT name injured players from memory.
   If a team has no entry in Section 4, omit injury sentences for that team entirely.
3. Home/away: sentence 1 of paragraph 1 only.
4. H2H: use Section 2 exclusively. Do not use any H2H claim from Section 4.
   Copy arena names VERBATIM from Section 2. Never substitute from memory.
5. All numbers (W/L, PPG, seed, H2H scores) must match Sections 1 and 2 exactly.
6. PLAYER PARAGRAPH OWNERSHIP — HARD WALL:
   PARAGRAPH 1 contains ONLY {home_full} players. PARAGRAPH 2 contains ONLY {away_full} players.
   ZERO exceptions. Never name an opposing player for any reason.
7. BANNED PHRASES — these must NOT appear anywhere in the output:
   "looking to", "look to", "aiming to", "hoping to",
   "bounce back", "capitalize", "will look", "seek to", "standout".
   Before submitting, scan every sentence and remove them.
8. Tone: dry and factual — a stats sheet read aloud, not a feature article.

Write ONLY the 2 paragraphs. Do not reproduce any tables."""

    response = llm.invoke(prompt)
    prose = response.content.strip()
    # Ensure "Injury Report:" header is always present before the injury lines
    if "Injury Report:" not in prose:
        # Find the last blank line and insert the header before the injury paragraph
        parts = prose.rsplit("\n\n", 1)
        if len(parts) == 2:
            prose = parts[0] + "\n\nInjury Report:\n" + parts[1]
    report = prose + "\n\n" + table
    print(f"[ReportComposer]    Final report ready ({len(report)} chars)")
    return {"final_report": report}


