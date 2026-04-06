"""
nodes/narrative_composer.py
───────────────────────────
Node 2 — Narrative Composer.
Replaces report_composer + storylines_composer.
Writes a 3-paragraph unified narrative using ESPN data + ChromaDB RAG context.
Runs after both data_specialist and context_extractor complete.
"""

import re as _re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from core.state import GraphState, extract_teams, parse_home_away, extract_roster_names
from db.chroma import get_collection
from nodes.utils import BANNED_PHRASES, BANNED_PATTERNS, find_violations

_llm = None

_N_RESULTS = 4


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    return _llm


# Banned phrases imported from nodes.utils — single source of truth
_BANNED_PHRASES  = BANNED_PHRASES
_BANNED_PATTERNS = BANNED_PATTERNS
_find_violations = find_violations


# ---------------------------------------------------------------------------
# Standings + signals helpers
# ---------------------------------------------------------------------------

_NUM_WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
    12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
}


def _record_seed_str(table: str, team_name: str) -> str:
    """Return 'W-L, #X East/West' for use in the opener sentence."""
    m = _re.search(
        rf"### {_re.escape(team_name)}.*?Record: W (\d+) / L (\d+) \| Seed: ([^|]+)\|",
        table, _re.DOTALL
    )
    if not m:
        return ""
    wins, losses, seed = m.groups()
    seed = seed.strip().replace(" Conference", "").replace(" Eastern", " East").replace(" Western", " West")
    return f"{wins}-{losses}, {seed}"


def _momentum_summary(table: str, team_name: str) -> str:
    m = _re.search(
        rf"### {_re.escape(team_name)}.*?Record: W (\d+) / L (\d+) \| Seed: ([^|]+)\| Streak: (\S+) \| Last 10: ([\d-]+)",
        table, _re.DOTALL
    )
    if not m:
        return f"{team_name}: record unavailable"
    wins, losses, seed, streak, last10 = m.groups()
    seed = seed.strip().replace(" Conference", "").replace(" Eastern", " East").replace(" Western", " West")
    direction = "winning" if streak.startswith("W") else "losing"
    n = streak[1:]
    games_remaining = 82 - int(wins) - int(losses)
    seed_num_m = _re.match(r'#(\d+)', seed)
    seed_num = int(seed_num_m.group(1)) if seed_num_m else None
    if seed_num and 7 <= seed_num <= 10:
        playoff_status = "play-in (seeds 7–10)"
    elif seed_num and 1 <= seed_num <= 6:
        playoff_status = "direct playoff (seeds 1–6)"
    else:
        playoff_status = "playoff status unknown"
    return (
        f"{team_name}: {wins}-{losses}, {seed}, {games_remaining} games remaining, "
        f"{playoff_status}, {direction} streak of {n}, {last10} last 10"
    )


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


def _active_roster(table: str, team_name: str) -> list[str]:
    m = _re.search(rf"### {_re.escape(team_name)}\n.*?(?=\n###|\Z)", table, _re.DOTALL)
    if not m:
        return []
    names = []
    for line in m.group(0).splitlines():
        if line.startswith("| ") and not line.startswith("| Player") and "---" not in line:
            parts = [p.strip() for p in line.split("|")]
            name = parts[1] if len(parts) > 1 else ""
            if name and name not in ("Stats unavailable", "—", ""):
                names.append(name)
    return names


def _game_signals(home_full: str, away_full: str, table: str, h2h_summary: str, injury_summary: str) -> str:
    signals = []

    streaks: dict[str, str] = {}
    for team in [home_full, away_full]:
        sm = _re.search(rf"### {_re.escape(team)}.*?Streak: (\S+)", table, _re.DOTALL)
        if sm:
            streaks[team] = sm.group(1)

    if len(streaks) == 2:
        h_s = streaks.get(home_full, "")
        a_s = streaks.get(away_full, "")
        if h_s and a_s and h_s[0] != a_s[0]:
            signals.append(
                f"STREAK COLLISION: {away_full} ({a_s}) visits {home_full} ({h_s}) — teams going in opposite directions."
            )

    a_s = streaks.get(away_full, "")
    if a_s.startswith("W") and int(a_s[1:]) >= 2:
        signals.append(f"ROAD TEST: {away_full} brings a {a_s} winning streak into an away game.")

    if h2h_summary and h2h_summary.strip() not in ("H2H data unavailable.", ""):
        for team in [home_full, away_full]:
            if _re.search(rf"{_re.escape(team)} won", h2h_summary, _re.IGNORECASE):
                loser = away_full if team == home_full else home_full
                signals.append(
                    f"H2H REMATCH: {team} won the last meeting. {loser} plays tonight looking to even the season series."
                )
                break

    ppg_lookup: dict[str, float] = {}
    for line in table.splitlines():
        if line.startswith("| ") and not line.startswith("| Player") and "---" not in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                try:
                    ppg_lookup[parts[1]] = float(parts[3])
                except (ValueError, IndexError):
                    pass

    _MIN_INJURY_PPG = 10.0
    for team in [home_full, away_full]:
        im = _re.search(rf"### {_re.escape(team)}\n(.*?)(?=\n###|\Z)", injury_summary, _re.DOTALL)
        if not im:
            continue
        significant = []
        for entry in im.group(1).splitlines():
            entry = entry.strip().lstrip("•").strip()
            if not entry or "No injuries" in entry or "unavailable" in entry:
                continue
            player_name = _re.split(r"\s*[—\-\(]", entry)[0].strip()
            if ppg_lookup.get(player_name, 0.0) >= _MIN_INJURY_PPG or player_name not in ppg_lookup:
                significant.append(entry)
        if significant:
            signals.append(f"INJURY ALERT ({team}): " + " | ".join(significant))

    return "\n".join(f"  • {s}" for s in signals) if signals else "  (none)"


def _extract_team_section(table: str, full_name: str) -> str:
    marker = f"### {full_name}"
    if marker not in table:
        return f"### {full_name}\nStats unavailable."
    after = table.split(marker)[1]
    next_section = after.find("\n###")
    return marker + (after[:next_section] if next_section != -1 else after)


def _emergency_note(section: str, team: str) -> str:
    for line in section.splitlines():
        if line.startswith("EMERGENCY ROSTER"):
            players = line.split(": ", 1)[1] if ": " in line else line
            return (
                f'INCLUDE THIS SENTENCE VERBATIM: "{team} are operating with a depleted '
                f'roster due to injuries and are relying on G League and short-term '
                f'players: {players}."'
            )
    return f"SKIP — {team} has no emergency roster situation."


def _dtd_stars(stats_section: str, inj_summary: str, team: str) -> str:
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
    top3 = {name for name, _ in sorted(player_ppg, key=lambda x: -x[1])[:2]}
    qualified_names = {name for name, _ in player_ppg}

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
                if name in top3:
                    dtd.append(name)

    if not dtd:
        return ""
    names = ", ".join(dtd)
    verb = "is" if len(dtd) == 1 else "are"
    return (
        f"NOTE: {names} {verb} listed as Day-To-Day. "
        f"Write exactly one sentence stating this. "
        f"Day-To-Day already implies uncertainty — do NOT add 'if he suits up', "
        f"'adds uncertainty', 'pending availability', or any qualifier. State the status only."
    )



def _narrative_milestone(ctx: str, roster: list[str]) -> str:
    """
    Scan retrieved chunks for a concrete milestone sentence tied to an active player.
    Looks for: game performances with numbers, ROY race, records, historical comparisons.
    Returns a pre-computed signal so the LLM can't miss it.
    """
    if not ctx or not roster:
        return ""

    last_names = {p.split()[-1].lower(): p for p in roster}

    MILESTONE_KEYWORDS = [
        "roy", "rookie of the year", "record", "surpassed", "most", "franchise",
        "historic", "history", "all-time", "ladder", "mvp", "surpass",
    ]

    best_sentence = ""
    best_player   = ""
    best_score    = 0

    for sent in _re.split(r'(?<=[.!?])\s+', ctx):
        if len(sent) < 60 or len(sent) > 400:
            continue
        # Skip markdown/stats headers — not prose
        if _re.search(r'\*\*|#{1,6}\s|^\d+\.|ladder:|stats:|draft pick:', sent, _re.IGNORECASE):
            continue
        # Must have at least 6 words to be a real sentence
        if len(sent.split()) < 6:
            continue
        lower = sent.lower()
        player_hit = next(
            (full for last, full in last_names.items() if last in lower),
            None
        )
        if not player_hit:
            continue
        score = sum(1 for kw in MILESTONE_KEYWORDS if kw in lower)
        if _re.search(r'\d+\s*(points?|pts)', lower):
            score += 1
        # Require a number for low-score sentences, but franchise/record sentences
        # often describe milestones in words — allow them if score is high enough
        if score == 0:
            continue
        if score == 1 and not _re.search(r'\d', sent):
            continue
        if score > best_score:
            best_score    = score
            best_sentence = sent.strip()
            best_player   = player_hit

    if best_sentence and best_score >= 1:
        return (
            f"MILESTONE ({best_player}): include this specific detail — "
            f'"{best_sentence}"'
        )
    return ""


def _absent_star_signal(team_name: str, recent_form: str) -> str:
    """Extract the ABSENT STAR bullet from the team's recent form section, if present."""
    import re as re2
    pattern = rf"### {re2.escape(team_name)}.*?(?=\n###|\Z)"
    m = re2.search(pattern, recent_form, re2.DOTALL)
    if not m:
        return ""
    for line in m.group(0).splitlines():
        if "ABSENT STAR:" in line:
            return line.strip()
    return ""


def _extract_verbatim_sentence(team_name: str, recent_form: str) -> str:
    """
    Extract the exact quoted sentence from the VERBATIM OPENING SENTENCE or
    INCLUDE THIS SENTENCE VERBATIM line in the team's recent form block.
    Returns the raw quoted text (without surrounding quotes), or "" if not found.
    """
    import re as re2
    pattern = rf"### {re2.escape(team_name)}.*?(?=\n###|\Z)"
    m = re2.search(pattern, recent_form, re2.DOTALL)
    if not m:
        return ""
    for line in m.group(0).splitlines():
        if "VERBATIM OPENING SENTENCE" in line or "INCLUDE THIS SENTENCE VERBATIM" in line:
            q = re2.search(r"'([^']+)'", line)
            if q:
                return q.group(1)
    return ""


def _retrieve(team_name: str, today: str) -> str:
    try:
        results = get_collection().query(
            query_texts=[f"{team_name} NBA season storyline arc key players recent games"],
            n_results=_N_RESULTS,
            where={"$and": [{"team_name": team_name}, {"date": today}]},
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            print(f"[NarrativeComposer] No chunks for {team_name} on {today} — trying without date filter")
            results = get_collection().query(
                query_texts=[f"{team_name} NBA season storyline arc key players recent games"],
                n_results=_N_RESULTS,
                where={"team_name": team_name},
            )
            docs = results.get("documents", [[]])[0]
        return "\n\n".join(docs) if docs else ""
    except Exception as exc:
        print(f"[NarrativeComposer] ChromaDB query failed for {team_name}: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def narrative_composer_node(state: GraphState) -> dict:
    query          = state.get("query", "")
    table          = state.get("player_stats_table", "")
    h2h_summary    = state.get("h2h_summary", "No H2H data available.")
    injury_summary = state.get("injury_summary", "")
    recent_form    = state.get("recent_form", "")
    stakes_context = state.get("stakes_context", "")

    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    teams = extract_teams(query)
    if len(teams) < 2:
        return {"narrative_section": ""}
    away_team, home_team = parse_home_away(query, teams)
    away_full, home_full = away_team[0], home_team[0]

    home_stats = _extract_team_section(table, home_full)
    away_stats  = _extract_team_section(table, away_full)

    home_momentum    = _momentum_summary(table, home_full)
    away_momentum    = _momentum_summary(table, away_full)
    signals          = _game_signals(home_full, away_full, table, h2h_summary, injury_summary)
    home_roster      = _active_roster(table, home_full)
    away_roster      = _active_roster(table, away_full)
    home_streak      = _streak_text(home_stats)
    away_streak      = _streak_text(away_stats)
    home_record_seed = _record_seed_str(table, home_full) or home_full
    away_record_seed = _record_seed_str(table, away_full) or away_full
    home_short       = home_full.split()[-1]
    away_short       = away_full.split()[-1]
    print(f"[NarrativeComposer] home_record_seed: {home_record_seed}")
    print(f"[NarrativeComposer] away_record_seed: {away_record_seed}")
    home_emergency = _emergency_note(home_stats, home_full)
    away_emergency = _emergency_note(away_stats, away_full)
    home_dtd       = _dtd_stars(home_stats, injury_summary, home_full)
    away_dtd       = _dtd_stars(away_stats, injury_summary, away_full)

    authorized_names = extract_roster_names(table)
    roster_allowlist = "\n".join(f"  - {n}" for n in authorized_names) or "  (none parsed)"

    print(f"[NarrativeComposer] Retrieving ChromaDB context for {home_full} + {away_full} ...")
    home_ctx = _retrieve(home_full, today)
    away_ctx = _retrieve(away_full, today)

    home_milestone   = _narrative_milestone(home_ctx, home_roster)
    away_milestone   = _narrative_milestone(away_ctx, away_roster)
    home_absent_star = _absent_star_signal(home_full, recent_form)
    away_absent_star = _absent_star_signal(away_full, recent_form)
    home_verbatim    = _extract_verbatim_sentence(home_full, recent_form)
    away_verbatim    = _extract_verbatim_sentence(away_full, recent_form)
    print(f"[NarrativeComposer] Home milestone: {home_milestone or 'none'}")
    print(f"[NarrativeComposer] Away milestone: {away_milestone or 'none'}")
    print(f"[NarrativeComposer] Home verbatim: {home_verbatim or 'none'}")
    print(f"[NarrativeComposer] Away verbatim: {away_verbatim or 'none'}")

    prompt = f"""You are a professional NBA analyst writing a pre-game briefing for fans.
Write in clear, factual prose — specific and grounded, not hype.

BEFORE WRITING — memorize the banned phrase list. Any banned phrase means your output is rejected:
  {" | ".join(_BANNED_PHRASES)}
  Also forbidden: any phrase matching "N of their last N", "N of N games", "in their last N games",
  "over the last N games", "all N of their", "won all N", "winning N of".

═══════════ MATCHUP ═══════════
{away_full} (AWAY) @ {home_full} (HOME)

─── STANDINGS (verified ESPN — use these numbers only, do not invent) ───
{home_momentum}
{away_momentum}

─── STAKES CONTEXT (pre-computed from full conference standings) ───
{stakes_context if stakes_context else "Stakes data unavailable."}

─── GAME CONTEXT SIGNALS (pre-computed from ESPN data) ───
{signals}

─── HOME TEAM STATS — {home_full} ───
{home_stats}

─── AWAY TEAM STATS — {away_full} ───
{away_stats}

─── ACTIVE ROSTERS (live ESPN — tonight's available players only) ───
{home_full}: {", ".join(home_roster) if home_roster else "unavailable"}
{away_full}: {", ".join(away_roster) if away_roster else "unavailable"}

─── AUTHORIZED STATS ROSTER (performance/stats mentions only) ───
{roster_allowlist}

CRITICAL: Do NOT mention any player by name unless they appear in the lists above.
If your training data says a player is on one of these teams but they are NOT in these lists,
they have been traded, waived, or released — do NOT mention them under any circumstances.

─── INJURY REPORT (ESPN API) ───
{injury_summary}

─── RECENT FORM (pre-analyzed — do NOT use raw game lines) ───
{recent_form}

─── PRE-EXTRACTED SIGNALS (Python-verified from retrieved content) ───
{home_full} MILESTONE: {home_milestone if home_milestone else "None."}
{away_full} MILESTONE: {away_milestone if away_milestone else "None."}

═══════════ OUTPUT ═══════════

Write EXACTLY 3 paragraphs, blank line between them. No headers. Do NOT write an injury section.

PARAGRAPH 1 — CONTEXT AND STAKES:
Write exactly 4 sentences, in this order — no more, no fewer:
  Sentence 1 (opener): Use this exact format —
    "The {home_full} ({home_record_seed}) host the {away_full} ({away_record_seed})."
    Nothing else in this sentence. No stakes summary here — records and seeds only.
  Sentence 2 ({home_full} stakes): Start with "{home_short}" (not "They", not the full name).
    Include: streak, games remaining, and the most urgent games-back number from STAKES CONTEXT.
    PRIORITY: if STAKES CONTEXT lists a RISK number for {home_full}, use that — it is more
    urgent than UPSIDE. Use UPSIDE only if no RISK exists.
    One sentence — do NOT mention {home_full}'s games-back situation again after this.
  Sentence 3 ({away_full} stakes): Start with "{away_short}" (not "They", not the full name).
    Same structure as sentence 2 but for {away_full}.
    PRIORITY: use the RISK number from STAKES CONTEXT if one exists for {away_full}.
    One sentence — do NOT mention {away_full}'s games-back situation again after this.
  Sentence 4 (game frame): one sentence using GAME CONTEXT SIGNALS — streak collision, H2H
    rematch, or playoff pressure angle. No new stakes numbers.

Rules for all 4 sentences:
- STREAK: {home_full} — {home_streak} | {away_full} — {away_streak}
- All records and seeds must match STANDINGS exactly.
- Use STAKES CONTEXT for all games-back numbers — do NOT invent any.
- Do NOT use effort/desire framing ("striving to", "vying for", "looking to", etc.).
- POSITION LABELS: use the exact label from STAKES CONTEXT — "play-in #7", "play-in #8",
  "direct playoff #6", etc. Do NOT simplify play-in to "playoff spot" or "playoffs".
  Only say "clinched" if the STAKES CONTEXT label says "clinched".
- This is the only paragraph that may reference both teams together.

PARAGRAPH 2 — {home_full}:
- About {home_full} ONLY. Stats from HOME TEAM STATS section.
- ABSENT STAR: {home_absent_star if home_absent_star else f"NONE — no OUT player for {home_full} is significant enough to receive an absence mention. Do NOT open this paragraph with any injury context. Do NOT write 'With [player] sidelined' for any OUT player. The absence-opening sentence applies ONLY when this signal is explicitly provided."}
- DAY-TO-DAY: {home_dtd if home_dtd else f"None — write nothing about injury status for {home_full}. Do NOT state that no players are injured or list any availability. Silence = healthy."}
  If a DAY-TO-DAY note exists, it must be the FIRST sentence of this paragraph — it is the most
  consequential fact about this team tonight. Do not bury it at the end.
- Name the top 2-3 scorers with season PPG. The player with the highest season PPG in HOME TEAM STATS must be named — do not skip the leading scorer.
- LEADING SCORER LABEL: the phrase "leading scorer" or "team's top scorer" must refer to the player with the HIGHEST PPG in HOME TEAM STATS — never to the player in the opening sentence unless they are also the PPG leader in the table.
- REQUIRED OPENING SENTENCE — copy this word-for-word as the very first sentence of this paragraph:
  "{home_verbatim if home_verbatim else f'[No verbatim sentence — open with the top scorer from {home_full} RECENT FORM block and their recent scores.]'}"
  Do NOT change the player name. Do NOT rephrase. Do NOT substitute any other player's name.
  This sentence was computed by Python from live ESPN data and is authoritative.
- RECENT FORM ATTRIBUTION:
  - The player named in the required opening sentence above is the ONLY player who may be
    credited with recent game scores in this paragraph.
  - If the form scorer and the season star are different people, name them separately (form
    scorer for recent games, season star for season average).
  - SINGLE ENGINE RULE: there is exactly ONE offensive engine per team. Do not describe any
    other player as the engine, primary scorer, or driving force of recent scoring.
  - DO NOT invent or add "averaging X PPG this season" for any player unless the RECENT FORM
    bullet explicitly includes a PPG figure (e.g., "Offensive engine: OG Anunoby (26 PPG season)").
    If no PPG appears in the bullet, do not add a season average for that player.
  - COLLAPSE/SURGE ARC: if present in RECENT FORM, incorporate it — use the quoted phrase or close paraphrase.
  - Never write "between X and Y points" — name each individual score.
  - Never write "over/in the last N games" with a point range.
- CLOSING SENTENCE: the final sentence of this paragraph must contain a player name AND a specific number (stat, score, or percentage). A sentence with only descriptions and no facts is rejected.
- MILESTONE: if a MILESTONE signal for {home_full} is present in PRE-EXTRACTED SIGNALS, include it in this paragraph with the specific detail verbatim or close paraphrase.
- EMERGENCY ROSTER: {home_emergency}
- Use ONLY players in {home_full}'s AUTHORIZED STATS ROSTER. Any player from {away_full} or any other team appearing in this paragraph is a critical error.

PARAGRAPH 3 — {away_full}:
- About {away_full} ONLY. Stats from AWAY TEAM STATS section.
- Write at least 3 sentences.
- Name at least 2 players with their season PPG from AWAY TEAM STATS. The player with the highest season PPG must appear in the first or second sentence with their full name and PPG figure — do not skip or bury the leading scorer.
- ABSENT STAR: {away_absent_star if away_absent_star else f"NONE — no OUT player for {away_full} is significant enough to receive an absence mention. Do NOT open this paragraph with any injury context. Do NOT write 'With [player] sidelined' for any OUT player. The absence-opening sentence applies ONLY when this signal is explicitly provided."}
- DAY-TO-DAY: {away_dtd if away_dtd else f"None — write nothing about injury status for {away_full}. Do NOT state that no players are injured or list any availability. Silence = healthy."}
  If a DAY-TO-DAY note exists, it must be the FIRST sentence of this paragraph.
- Name the top 2-3 scorers with season PPG.
- LEADING SCORER LABEL: the phrase "leading scorer" or "team's top scorer" must refer to the player with the HIGHEST PPG in AWAY TEAM STATS — never to the player in the opening sentence unless they are also the PPG leader in the table.
- REQUIRED OPENING SENTENCE — copy this word-for-word as the very first sentence of this paragraph:
  "{away_verbatim if away_verbatim else f'[No verbatim sentence — open with the top scorer from {away_full} RECENT FORM block and their recent scores.]'}"
  Do NOT change the player name. Do NOT rephrase. Do NOT substitute any other player's name.
  This sentence was computed by Python from live ESPN data and is authoritative.
- RECENT FORM ATTRIBUTION:
  - The player named in the required opening sentence above is the ONLY player who may be
    credited with recent game scores in this paragraph.
  - If the form scorer and the season star are different people, name them separately (form
    scorer for recent games, season star for season average).
  - SINGLE ENGINE RULE: there is exactly ONE offensive engine per team. Do not describe any
    other player as the engine, primary scorer, or driving force of recent scoring.
  - DO NOT invent or add "averaging X PPG this season" for any player unless the RECENT FORM
    bullet explicitly includes a PPG figure (e.g., "Offensive engine: OG Anunoby (26 PPG season)").
    If no PPG appears in the bullet, do not add a season average for that player.
- CLOSING SENTENCE: the final sentence of this paragraph must contain a player name AND a specific number (stat, score, or percentage). A sentence with only descriptions and no facts is rejected.
- MILESTONE: if a MILESTONE signal for {away_full} is present in PRE-EXTRACTED SIGNALS, include it in this paragraph with the specific detail verbatim or close paraphrase.
- EMERGENCY ROSTER: {away_emergency}
- Use ONLY players from {away_full}'s AUTHORIZED STATS ROSTER. Any player from {home_full} or any other team appearing in this paragraph is a critical error.

ABSOLUTE RULES:
1. ACTIVE ROSTER: Only name players listed in ACTIVE ROSTERS. Any player not listed has been traded or is unavailable.
2. OUT PLAYERS: Players listed as OUT in INJURY REPORT are not playing tonight. If an OUT player has an ABSENT STAR signal in RECENT FORM, open that team's paragraph with exactly one sentence acknowledging their absence (e.g., "With [name] sidelined, [active scorer] has stepped up."). That is the only sentence permitted about an OUT player. Do NOT describe any OUT player as the offensive engine, a recent scorer, or a factor in tonight's game — even as context. Any OUT player not flagged as ABSENT STAR must not be mentioned at all.
3. All numbers must match STANDINGS and team stats sections exactly.
4. Record format: always "W-L" numerals (e.g., "46-31"). Never write records in words.
5. Every sentence must contain at least one specific fact: player name, stat, score, or date.
6. Extraordinary claims (records, franchise milestones, awards) require a MILESTONE signal to be present. Do not invent facts not grounded in the data sections above.
7. Name players explicitly — never "a veteran" or "a rookie" without a name.
8. Zero citations, zero URLs. Pure prose.
9. PARAGRAPH OWNERSHIP: paragraph 2 = {home_full} only, paragraph 3 = {away_full} only. No cross-team player mentions in these paragraphs.
10. BANNED PHRASES: reread your output before finalizing. Any sentence with a banned phrase must be rewritten.

Write ONLY the 3 paragraphs."""

    print(f"[NarrativeComposer] Generating narrative ...")
    response = _get_llm().invoke([HumanMessage(content=prompt)])
    narrative = response.content.strip()

    violations = _find_violations(narrative)
    if violations:
        print(f"[NarrativeComposer] WARNING — banned phrases detected (reviewer will handle): {violations}")

    print(f"[NarrativeComposer] Done ({len(narrative)} chars)")
    return {"narrative_section": narrative}
