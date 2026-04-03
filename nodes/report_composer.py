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

BEFORE WRITING ANYTHING — memorize this banned word list. These phrases are FORBIDDEN.
If you write any of them, your output is rejected. There are no exceptions:
  looking to | look to | aiming to | hoping to | bounce back | capitalize | will look |
  seek to | stand out | impressive | dominant | come into | turn around |
  highlighted by | highlighted | struggles | find their footing | found their footing |
  firing on all cylinders | showcasing | potential | when healthy | make a statement |
  of their last | of their past | X of Y games | N of N games |
  leading the offense in | leading the scoring in | led scoring in | led the offense in |
  averaging between | bringing an average | in their last N games | over the last N games |
  in his last N games | in her last N games | in the last N games

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

Write exactly 2 paragraphs, blank line between them. No headers. Do NOT write an injury section — that is handled separately.

PARAGRAPH 1 — {home_full} (HOME):
• Opening sentence: state the team name, their actual W-L record and conference seed from Section 1,
  and the opponent. End with a period. Nothing else after "tonight."
  Example: "The Boston Celtics (48-28, #2 East) host the Miami Heat tonight."
  Use the real numbers from Section 1 — do not write placeholder text like "W-L" or "N Conf".
  Do NOT append any clause after "tonight." — no "looking to", no "as they", no "coming off", nothing.
  Never mention home/away again after this sentence.
• State team PPG and Opp PPG with NBA rank (from Section 1 Team Stats line).
• State current win/loss streak and last-10 record (from Section 1).
• Weave in 1–2 sentences about their recent form using the ANALYSIS block in Section 5.
  CRITICAL RULES FOR ANALYSIS BLOCK:
  - COLLAPSE ARC / SURGE ARC: If present, you MUST incorporate that arc. Use the quoted phrase
    directly or paraphrase it closely. Do NOT omit or reduce it to a vague summary.
  - "Offensive engine" / "Go-to scorer": Name that player and their point range.
    The range covers only the games they led scoring — NOT all games in the window.
    Frame it as "in his recent top performances" or name the specific opponents
    (e.g., "scoring 26 and 32 against Denver and Golden State").
    NEVER write "over the last N games" or "in their last N games" with a point range —
    that implies they scored that range every game, which is false.
  - "Spread offense": Describe collective output. Do not single out one scorer.
  - "Notable performance": If the ANALYSIS block names an opponent for the peak game, mention it
    (e.g., "peaking at 32 against Golden State"). If no opponent is named, state the number only —
    never write vague phrases like "in one of those games", "in one of these outings", "in a recent game."
  Example: "Orlando have fallen off sharply, dropping seven of their last five after going 4-1 to open this stretch."
  Never use vague filler: "several recent games", "multiple times", "consistently."
  Never write mechanical counts: "X of their last Y", "N of N games."
  Use ONLY players in Section 3. Do not invent or recall from memory.
• Weave in H2H context from Section 2: include exact scores, local dates, and home team for each game.
  H2H appears ONLY here — do NOT repeat or reference it in Paragraph 2.

PARAGRAPH 2 — {away_full} (AWAY):
• Open directly with their record and seed. Do NOT restate home/away.
• State team PPG and Opp PPG with NBA rank.
• State current streak and last-10.
• Weave in 1–2 sentences about their recent form using the ANALYSIS block in Section 5.
  Same CRITICAL RULES as Paragraph 1 — COLLAPSE ARC / SURGE ARC must be included if present.
  Never use vague filler: "several recent games", "multiple times", "consistently."
  Never write mechanical counts: "X of their last Y", "N of N games."
• Do NOT mention H2H here. It was already covered in Paragraph 1.
  Use ONLY players in Section 3. Do not invent or recall from memory.


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
   RECORD FORMAT: Always write records as "W-L" with a dash and numerals (e.g., "40-36").
   Never write records in words ("40 wins and 36 losses") or any other format.
6. PLAYER PARAGRAPH OWNERSHIP — HARD WALL:
   PARAGRAPH 1 contains ONLY {home_full} players. PARAGRAPH 2 contains ONLY {away_full} players.
   ZERO exceptions. Never name an opposing player for any reason.
7. BANNED PHRASES — see the list at the top of this prompt. Every one of them is forbidden.
   Reread your output word by word before finalizing. Any sentence containing a banned phrase
   must be discarded and rewritten from scratch.
8. STREAK ACCURACY: The current streak is in Section 1 as "Streak: W/LN". Use it exactly.
   Never infer streak from recent form or from ANALYSIS — Section 1 is the ONLY source for streak.
   The ANALYSIS block does NOT contain a streak — ignore any streak-like language there.
9. ACTIVE ROSTER: Never state or infer how many players are available. Do not mention
   roster size unless Section 1 explicitly shows "Active roster: N players" in the record line.
10. Tone: dry and factual — a stats sheet read aloud, not a feature article.

Write ONLY the 2 paragraphs. Do not reproduce any tables."""

    response = llm.invoke(prompt)
    prose = response.content.strip()

    # ── Build injury block in Python — never trust LLM for this ──────────
    def _injury_line(team_name: str) -> str:
        marker = f"### {team_name}"
        if marker not in injury_summary:
            return f"{team_name} — None reported."
        section = injury_summary.split(marker)[1].split("###")[0]
        entries = [l.strip(" •").strip() for l in section.splitlines() if l.strip().startswith("•")]
        if not entries:
            return f"{team_name} — None reported."
        formatted = ", ".join(
            f"{e.split(' — ')[0]} ({'OUT' if 'Out' in e else 'Day-To-Day'})"
            for e in entries if " — " in e
        )
        return f"{team_name} — {formatted}" if formatted else f"{team_name} — None reported."

    injury_block = (
        "Injury Report:\n"
        + _injury_line(home_full) + "\n"
        + _injury_line(away_full)
    )

    report = prose + "\n\n" + injury_block + "\n\n" + table
    print(f"[ReportComposer]    Final report ready ({len(report)} chars)")
    return {"final_report": report}


