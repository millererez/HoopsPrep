"""
nodes/report_composer.py
────────────────────────
Node 3 — Report Composer.
Combines ESPN API stats, ESPN H2H data, and Tavily narrative context
into a clean TV broadcast briefing.
"""

import re as _re_global

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from core.state import GraphState, extract_teams, parse_home_away, extract_roster_names

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# Banned phrases — single source of truth used in prompt + retry checker
# ---------------------------------------------------------------------------

_BANNED_PHRASES = [
    "looking to", "look to", "aiming to", "hoping to", "bounce back", "capitalize",
    "will look", "seek to", "stand out", "standout", "impressive", "dominant",
    "come into", "comes into", "turn around", "highlighted by", "highlighted",
    "struggles", "find their footing", "found their footing", "firing on all cylinders",
    "showcasing", "potential", "when healthy", "make a statement",
    "of their last", "of their past",
    "leading the offense in", "leading the scoring in", "led scoring in", "led the offense in",
    "averaging between", "posting between", "scoring between", "between n and",
    "bringing an average", "emerged", "has emerged", "have emerged",
    "a strong showing", "strong showing", "collective offensive", "offensive rhythm",
    "offensive output", "won all", "all ten of", "notable performances",
    "recent success", "recent form", "recent stretch", "performing well",
    "their ability to", "has seen them", "contributing to their",
    "winning four", "winning n of", "consistent", "surge in", "effective in",
    "demonstrating", "generating points", "strong collective", "in their recent",
    "adds uncertainty", "if he suits up", "pending availability",
    "at the bottom", "which ranks", "which is nth",
    "notable performance", "contributing",
    "has shown improvement", "shown improvement", "offensive presence",
    "demonstrating", "showcasing", "displaying", "exhibiting",
    "contributions", "has been a", "relying on", "peaking at",
]

# Patterns for "N of N games" style (any digits)
_BANNED_PATTERNS = [
    r"\d+ of their last \d+",
    r"\d+ of \d+ games",
    r"in their last \d+ games",
    r"over the last \d+ games",
    r"in his last \d+ games",
    r"in the last \d+ games",
    r"all \d+ of their",
    r"won all \d+",
    r"winning \d+ of",
]


def _find_violations(text: str) -> list[str]:
    found = []
    lower = text.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in lower:
            found.append(f'"{phrase}"')
    for pattern in _BANNED_PATTERNS:
        if _re_global.search(pattern, lower):
            found.append(f"/{pattern}/")
    return found

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

    # Split the combined table into per-team sections so the LLM
    # cannot confuse which stats belong to which team.
    def _extract_team_section(t: str, full_name: str) -> str:
        marker = f"### {full_name}"
        if marker not in t:
            return f"### {full_name}\nStats unavailable."
        after = t.split(marker)[1]
        # Everything up to the next ### or end of string
        next_section = after.find("\n###")
        return marker + (after[:next_section] if next_section != -1 else after)

    home_stats = _extract_team_section(table, home_full)
    away_stats  = _extract_team_section(table, away_full)

    # Build the opening sentence in Python — do not let the LLM decide who hosts.
    import re as _re
    def _parse_record_seed(section: str) -> tuple[str, str]:
        m = _re.search(r"Record: W (\d+) / L (\d+) \| Seed: (#\d+ \w+)", section)
        if m:
            record = f"{m.group(1)}-{m.group(2)}"
            seed   = m.group(3).replace(" Conference", "").replace(" Eastern", " East").replace(" Western", " West")
            return record, seed
        return "?-?", "?"

    home_record, home_seed = _parse_record_seed(home_stats)
    opening_sentence = (
        f"The {home_full} ({home_record}, {home_seed}) "
        f"host the {away_full} tonight."
    )

    # Pre-compute streak+last-10 text — skip streak if it's exactly 1 game.
    _NUM_WORDS = {
        2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
        7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
        12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    }

    def _streak_text(section: str) -> str:
        m = _re.search(r"Streak: (W|L)(\d+).*Last 10: ([\d-]+)", section)
        if not m:
            return ""
        direction, n, last10 = m.group(1), int(m.group(2)), m.group(3)
        word = "winning" if direction == "W" else "losing"
        n_word = _NUM_WORDS.get(n, str(n))
        if n == 1:
            return f"State only: 'They have a record of {last10} in their last ten contests.' Do NOT mention the {word} streak."
        if direction == "W" and n >= 10:
            return f"State: 'They are on a {n_word}-game winning streak.' Do NOT include the last-10 record."
        return f"State: 'They are on a {n_word}-game {word} streak and have a record of {last10} in their last ten contests.'"

    home_streak_text = _streak_text(home_stats)
    away_streak_text = _streak_text(away_stats)

    # Pre-compute emergency roster note in Python — never let the LLM decide.
    def _emergency_note(section: str, team: str) -> str:
        for line in section.splitlines():
            if line.startswith("EMERGENCY ROSTER"):
                players = line.split(": ", 1)[1] if ": " in line else line
                return (
                    f'INCLUDE THIS SENTENCE VERBATIM: "{team} are operating with a depleted '
                    f'roster due to injuries and are relying on G League and short-term '
                    f'players: {players}."'
                )
        return f"SKIP — {team} has no emergency roster situation. Do NOT mention roster depletion for this team."

    home_emergency = _emergency_note(home_stats, home_full)
    away_emergency = _emergency_note(away_stats, away_full)

    # Pre-compute Day-To-Day star notes — flag top-3 PPG players who are DTD.
    def _dtd_stars(stats_section: str, inj_summary: str, team: str) -> str:
        # Parse PPG from stats table rows for this team
        player_ppg: list[tuple[str, float]] = []
        in_table = False
        for line in stats_section.splitlines():
            if line.startswith("| Player"):
                in_table = True
                continue
            if in_table and line.startswith("| ") and "---" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) > 3:
                    name = parts[1]
                    try:
                        ppg = float(parts[3])
                        player_ppg.append((name, ppg))
                    except ValueError:
                        pass
        top3 = {name for name, _ in sorted(player_ppg, key=lambda x: -x[1])[:3]}
        qualified_names = {name for name, _ in player_ppg}

        # Find DTD players for this team in injury summary
        # Flag: top-3 PPG players who are DTD, AND any DTD player not in the
        # stats table at all (injured most of the season — more newsworthy, not less).
        dtd: list[str] = []
        in_team = False
        for line in inj_summary.splitlines():
            if line.startswith(f"### {team}"):
                in_team = True
                continue
            if line.startswith("###"):
                in_team = False
            if in_team and "Day-To-Day" in line:
                m = _re.match(r"\s*•\s*(.+?)\s*\u2014", line)
                if m:
                    name = m.group(1)
                    if name in top3 or name not in qualified_names:
                        dtd.append(name)

        if not dtd:
            return ""
        names = ", ".join(dtd)
        verb = "is" if len(dtd) == 1 else "are"
        return (
            f"NOTE: {names} {verb} listed as Day-To-Day. "
            f"Write exactly one sentence stating this — e.g. '{names} {'is' if len(dtd) == 1 else 'are'} listed as Day-To-Day.' "
            f"Day-To-Day already implies uncertainty. Do NOT add 'if he suits up', 'adds uncertainty', "
            f"'pending availability', or any other uncertainty phrase. Just state the status, nothing more."
        )

    home_dtd_note = _dtd_stars(home_stats, injury_summary, home_full)
    away_dtd_note = _dtd_stars(away_stats, injury_summary, away_full)

    prompt = f"""You are a professional NBA broadcast statistician.
Write a pre-game broadcast briefing in ENGLISH — clean, spoken prose, TV-ready.

BEFORE WRITING ANYTHING — memorize this banned word list. These phrases are FORBIDDEN.
If you write any of them, your output is rejected and you will be asked to rewrite. There are no exceptions:
  {" | ".join(_BANNED_PHRASES)}
  Also forbidden: any phrase matching "N of their last N", "N of N games", "in their last N games",
  "over the last N games", "all N of their", "won all N", "winning N of" (where N is any number).

═══════════ MATCHUP ═══════════
{home_full} (HOME)  vs  {away_full} (AWAY)
═══════════════════════════════

─── SECTION 1A: HOME TEAM STATS — {home_full} ───
{home_stats}

─── SECTION 1B: AWAY TEAM STATS — {away_full} ───
{away_stats}

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
• THIS PARAGRAPH IS ABOUT {home_full} ONLY. Stats come from SECTION 1A.
• FIRST SENTENCE IS PRE-WRITTEN. Copy it verbatim, do not change a single word:
  "{opening_sentence}"
  Never mention home/away again after this sentence.
• State team PPG and Opp PPG with NBA rank (from Section 1A). Format both as "[X] points per game, ranking [N]th in the NBA" — do not use "which ranks" or "which is".
• STREAK/LAST-10 (pre-written — copy verbatim): {home_streak_text}
• EMERGENCY ROSTER: {home_emergency}
• DAY-TO-DAY STARS: {home_dtd_note if home_dtd_note else f"None — do NOT add availability caveats for {home_full} players."}
• Weave in 1–2 sentences about their recent form using the ANALYSIS block in Section 5.
  CRITICAL RULES FOR ANALYSIS BLOCK:
  - COLLAPSE ARC / SURGE ARC: If present, you MUST incorporate that arc. Use the quoted phrase
    directly or paraphrase it closely. Do NOT omit or reduce it to a vague summary.
  - "Offensive engine" / "Go-to scorer": Name that player and their point range.
    The range covers only the games they led scoring — NOT all games in the window.
    Open directly with "[Player] scored..." — NEVER open with "[Player] has been..." or any characterization.
    Frame it as naming each score with its opponent
    (e.g., "scoring 26 against the Denver Nuggets and 32 against the Golden State Warriors").
    Use the full team name from the analysis block — never shorten to just a city name (e.g. "Los Angeles" alone is ambiguous — write "the Los Angeles Lakers" or "the Los Angeles Clippers").
    NEVER write "between X and Y points" — always name the individual scores.
    NEVER write "over the last N games" or "in their last N games" with a point range —
    that implies they scored that range every game, which is false.
  - "Spread offense": ONLY mention spread offense if the ANALYSIS block contains a bullet explicitly labeled "Spread offense". If no such bullet exists, do NOT write anything about spread offense. If the bullet exists, it contains a pre-written sentence marked VERBATIM — copy it exactly, do not rephrase, extend, or add any clause.
  - "Notable performance": If the ANALYSIS block names an opponent for the peak game, mention it
    (e.g., "scoring 32 against Golden State"). If no opponent is named, state the number only —
    never write vague phrases like "in one of those games", "in one of these outings", "in a recent game."
  Example: "Orlando have fallen off sharply, dropping seven of their last five after going 4-1 to open this stretch."
  Never use vague filler: "several recent games", "multiple times", "consistently."
  Never write mechanical counts: "X of their last Y", "N of N games."
  Use ONLY players in Section 3. Do not invent or recall from memory.
• Do NOT include H2H — it is displayed separately after the injury report.

PARAGRAPH 2 — {away_full} (AWAY):
• THIS PARAGRAPH IS ABOUT {away_full} ONLY. Stats come from SECTION 1B.
• Open directly with their record and seed. Do NOT restate home/away.
• State team PPG and Opp PPG with NBA rank (from Section 1B). Format both as "[X] points per game, ranking [N]th in the NBA" — do not use "which ranks" or "which is".
• STREAK/LAST-10 (pre-written — copy verbatim): {away_streak_text}
• EMERGENCY ROSTER: {away_emergency}
• DAY-TO-DAY STARS: {away_dtd_note if away_dtd_note else f"None — do NOT add availability caveats for {away_full} players."}
• Weave in 1–2 sentences about their recent form using the ANALYSIS block in Section 5.
  Same CRITICAL RULES as Paragraph 1 — COLLAPSE ARC / SURGE ARC must be included if present.
  Never use vague filler: "several recent games", "multiple times", "consistently."
  Never write mechanical counts: "X of their last Y", "N of N games."
• Do NOT mention H2H here. It is displayed separately.
  Use ONLY players in Section 3. Do not invent or recall from memory.


ABSOLUTE RULES:
0. PARAGRAPH ASSIGNMENT IS FIXED AND NON-NEGOTIABLE:
   Paragraph 1 = {home_full}. Paragraph 2 = {away_full}.
   If paragraph 1 contains the name "{away_full}" as the subject, you are wrong. Rewrite it.
   If paragraph 2 contains the name "{home_full}" as the subject, you are wrong. Rewrite it.
1. ZERO source citations. ZERO URLs. ZERO parentheses containing sources.
   Pure spoken broadcast prose.
2. ROSTER INTEGRITY:
   • Performance/stats/recent form mentions: player must be in Section 3.
   • Injury mentions: player must appear in Section 4. Do NOT name injured players from memory.
   If a team has no entry in Section 4, omit injury sentences for that team entirely.
3. Home/away: sentence 1 of paragraph 1 only.
4. H2H: Do NOT include any H2H content in the paragraphs. It is handled separately.
5. All numbers (W/L, PPG, seed, H2H scores) must match Sections 1A, 1B, and 2 exactly.
   RECORD FORMAT: Always write records as "W-L" with a dash and numerals (e.g., "40-36").
   Never write records in words ("40 wins and 36 losses") or any other format.
6. PLAYER PARAGRAPH OWNERSHIP — HARD WALL:
   PARAGRAPH 1 contains ONLY {home_full} players. PARAGRAPH 2 contains ONLY {away_full} players.
   ZERO exceptions. Never name an opposing player for any reason.
7. BANNED PHRASES — see the list at the top of this prompt. Every one of them is forbidden.
   Reread your output word by word before finalizing. Any sentence containing a banned phrase
   must be discarded and rewritten from scratch.
8. STREAK ACCURACY: The streak/last-10 text is pre-written in each paragraph instruction — copy it verbatim.
   Never infer streak from recent form or from ANALYSIS — use only the pre-written text.
   NOTE: the ban on "in their last N games" applies to scoring ranges only — a W-L record like "8-2 in their last ten" is allowed.
9. ACTIVE ROSTER: Never state or infer how many players are available. Do not mention
   roster size unless Section 1 explicitly shows "Active roster: N players" in the record line.
10. Tone: dry and factual — a stats sheet read aloud, not a feature article.
11. NO FILLER SENTENCES. Every sentence must contain at least one specific stat, player name, score, or date.
    Sentences like "The Spurs have demonstrated a consistent offensive rhythm" contain zero facts — delete them.

Write ONLY the 2 paragraphs. Do not reproduce any tables."""

    # ── LLM call with retry loop for banned phrases ──────────────────────────
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    prose = response.content.strip()

    for attempt in range(2):
        violations = _find_violations(prose)
        if not violations:
            break
        print(f"[ReportComposer]    Banned phrases detected (attempt {attempt + 1}): {violations}")
        messages = [
            HumanMessage(content=prompt),
            AIMessage(content=prose),
            HumanMessage(content=(
                f"Your output contains these banned phrases: {', '.join(violations)}. "
                f"Rewrite ONLY the sentences that contain them. "
                f"Keep every other sentence identical. "
                f"Return the full 2-paragraph output."
            )),
        ]
        response = llm.invoke(messages)
        prose = response.content.strip()

    violations = _find_violations(prose)
    if violations:
        print(f"[ReportComposer]    WARNING — banned phrases remain after retries: {violations}")

    # ── Force correct opening sentence in Python — LLM cannot be trusted ────
    first_period = prose.find(".")
    if first_period != -1:
        prose = opening_sentence + prose[first_period + 1:]

    print(f"[ReportComposer]    Prose ready ({len(prose)} chars)")
    return {"prose_section": prose}


