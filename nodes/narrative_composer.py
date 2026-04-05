"""
nodes/narrative_composer.py
───────────────────────────
Node 2 — Narrative Composer.
Replaces report_composer + storylines_composer.
Writes a 4-paragraph unified narrative using ESPN data + ChromaDB RAG context.
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
    top3 = {name for name, _ in sorted(player_ppg, key=lambda x: -x[1])[:3]}
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
                if name in top3 or name not in qualified_names:
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


def _matchup_signal(home_full: str, away_full: str, table: str) -> str:
    """
    Pre-compute the key tactical tension for para 4.
    Columns: Player | MIN | PPG | FG% | 3P% | FT% | RPG | APG | STL | BLK | TOV
    Indices after split by |: 1=Player 2=MIN 3=PPG 4=FG% 5=3P% 6=FT% 7=RPG 8=APG 9=STL 10=BLK 11=TOV
    """
    def _parse_players(team_name: str) -> list[dict]:
        # New table columns: Player(1) MIN(2) PPG(3) FGA(4) FG%(5) 3PA(6) 3P%(7) FTA(8) FT%(9) RPG(10) APG(11) STL(12) BLK(13) TOV(14)
        section = _extract_team_section(table, team_name)
        players = []
        in_table = False
        for line in section.splitlines():
            if line.startswith("| Player"):
                in_table = True
                continue
            if in_table and line.startswith("| ") and "---" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 15:
                    try:
                        players.append({
                            "name": parts[1],
                            "mins": float(parts[2]),
                            "ppg":  float(parts[3]),
                            "tpa":  float(parts[6]),
                            "tpp":  float(parts[7]),
                            "rpg":  float(parts[10]),
                            "apg":  float(parts[11]),
                            "blk":  float(parts[13]),
                        })
                    except (ValueError, IndexError):
                        pass
        return players

    away_players = _parse_players(away_full)
    home_players = _parse_players(home_full)
    if not away_players or not home_players:
        return ""

    # Away team's best 3P threat (20+ MIN, 35%+ from three, 1.5+ 3PA as volume filter)
    shooters = [p for p in away_players if p["mins"] >= 20 and p["tpp"] >= 35.0 and p["tpa"] >= 1.5]
    best_shooter = max(shooters, key=lambda p: p["tpp"]) if shooters else None

    # Home team's paint anchor (highest BLK, 20+ MIN)
    anchors = [p for p in home_players if p["mins"] >= 20]
    paint_anchor = max(anchors, key=lambda p: p["blk"]) if anchors else None

    if best_shooter and paint_anchor and paint_anchor["blk"] >= 1.5:
        return (
            f"MATCHUP ANCHOR: {best_shooter['name']} ({away_full}) shoots {best_shooter['tpp']}% "
            f"from three ({best_shooter['ppg']} PPG). {paint_anchor['name']} ({home_full}) averages "
            f"{paint_anchor['blk']} blocks per game. Use this perimeter-vs-paint tension as the "
            f"foundation for paragraph 4: whether {away_full}'s shooters can pull {paint_anchor['name']} "
            f"from the paint determines spacing for both offenses."
        )

    # Fallback: top scorer vs top scorer — enrich with multiple stats to prevent repetition
    away_star = max(away_players, key=lambda p: p["ppg"]) if away_players else None
    home_star = max(home_players, key=lambda p: p["ppg"]) if home_players else None
    if away_star and home_star:
        return (
            f"MATCHUP ANCHOR: {away_star['name']} ({away_full}) — {away_star['ppg']} PPG, "
            f"{away_star['tpp']}% from three, {away_star['rpg']} RPG. "
            f"{home_star['name']} ({home_full}) — {home_star['ppg']} PPG, "
            f"{home_star['tpp']}% from three, {home_star['rpg']} RPG. "
            f"Frame paragraph 4 around this duel. Use a DIFFERENT stat in each sentence — "
            f"do not repeat any number already used in this paragraph."
        )
    return ""


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
    matchup_signal   = _matchup_signal(home_full, away_full, table)
    home_roster      = _active_roster(table, home_full)
    away_roster      = _active_roster(table, away_full)
    home_streak    = _streak_text(home_stats)
    away_streak    = _streak_text(away_stats)
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
    print(f"[NarrativeComposer] Home milestone: {home_milestone or 'none'}")
    print(f"[NarrativeComposer] Away milestone: {away_milestone or 'none'}")

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

─── INJURY REPORT (ESPN API) ───
{injury_summary}

─── RECENT FORM (pre-analyzed — do NOT use raw game lines) ───
{recent_form}

─── PRE-EXTRACTED SIGNALS (Python-verified from retrieved content) ───
{home_full} MILESTONE: {home_milestone if home_milestone else "None."}
{away_full} MILESTONE: {away_milestone if away_milestone else "None."}

═══════════ OUTPUT ═══════════

Write EXACTLY 4 paragraphs, blank line between them. No headers. Do NOT write an injury section.

PARAGRAPH 1 — CONTEXT AND STAKES:
- Open with {home_full}'s record, their recent stretch, and where they stand with games remaining.
- Bring in {away_full}: their record, streak, and what's at stake for them (play-in pressure, games left).
- Use GAME CONTEXT SIGNALS where relevant: streak collision, H2H rematch frame, playoff implications.
- STREAK: {home_full} — {home_streak} | {away_full} — {away_streak}
- All records and seeds must match STANDINGS exactly.
- CRITICAL: use STAKES CONTEXT to state specific consequences — games back, what each position
  means, what is and isn't within reach. Do NOT invent any games-back numbers — use only what
  STAKES CONTEXT provides. Do NOT use effort/desire framing ("striving to", "vying for", etc.).
- REQUIRED: include at least one specific games-back number for EACH team from STAKES CONTEXT.
  A team's playoff situation described with no games-back number is an automatic error.
- POSITION LABELS: use the exact label from STAKES CONTEXT — "play-in #7", "play-in #8",
  "direct playoff #6", etc. Do NOT simplify play-in positions to "playoff spot" or "playoffs" —
  play-in is not a guaranteed playoff berth. Only say "direct playoff berth" if the label says
  "direct playoff".
- NO PREAMBLE: Do NOT open with a generic setup sentence ("The stakes are high...", "In a pivotal
  matchup...", "Tonight's game...", "As the season winds down..."). Start directly with {home_full}'s
  record and situation — that IS the opening.
- This is the only paragraph that may reference both teams together.

PARAGRAPH 2 — {home_full}:
- About {home_full} ONLY. Stats from HOME TEAM STATS section.
- ABSENT STAR: {home_absent_star if home_absent_star else f"NONE — no OUT player for {home_full} is significant enough to receive an absence mention. Do NOT open this paragraph with any injury context. Do NOT write 'With [player] sidelined' for any OUT player. The absence-opening sentence applies ONLY when this signal is explicitly provided."}
- DAY-TO-DAY: {home_dtd if home_dtd else f"None — write nothing about injury status for {home_full}. Do NOT state that no players are injured or list any availability. Silence = healthy."}
  If a DAY-TO-DAY note exists, it must be the FIRST sentence of this paragraph — it is the most
  consequential fact about this team tonight. Do not bury it at the end.
- Name the top 2-3 scorers with season PPG. The player with the highest season PPG in HOME TEAM STATS must be named — do not skip the leading scorer.
- RECENT FORM: weave in 1-2 sentences from the ANALYSIS block in RECENT FORM for {home_full}.
  - ATTRIBUTION RULE: before writing, identify the player named under "Offensive engine" or
    "Go-to scorer" in the {home_full} RECENT FORM block. That is the player who produced those
    scores. Use THAT name — even if a different player leads season PPG. If the form scorer and
    the season star are different people, name them separately (form scorer for recent games,
    season star for season average). Never transfer recent game scores to the season PPG leader.
  - SINGLE ENGINE RULE: there is exactly ONE offensive engine per team — the player named under
    "Offensive engine" or "Go-to scorer" in RECENT FORM. Do not describe any other player as
    the engine, primary scorer, or driving force of recent scoring.
  - COLLAPSE/SURGE ARC: if present, incorporate it — use the quoted phrase or close paraphrase.
  - Open directly with "[Player] scored X against Y and Z against W." Never characterize first.
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
- RECENT FORM: same attribution and formatting rules as paragraph 2. Before writing, identify
  the player named under "Offensive engine" or "Go-to scorer" in the {away_full} RECENT FORM
  block. Use THAT name for the scoring credit — not the season PPG leader.
  - SINGLE ENGINE RULE: there is exactly ONE offensive engine per team — the player named under
    "Offensive engine" or "Go-to scorer" in RECENT FORM. Do not describe any other player as
    the engine, primary scorer, or driving force of recent scoring.
- CLOSING SENTENCE: the final sentence of this paragraph must contain a player name AND a specific number (stat, score, or percentage). A sentence with only descriptions and no facts is rejected.
- MILESTONE: if a MILESTONE signal for {away_full} is present in PRE-EXTRACTED SIGNALS, include it in this paragraph with the specific detail verbatim or close paraphrase.
- EMERGENCY ROSTER: {away_emergency}
- Use ONLY players from {away_full}'s AUTHORIZED STATS ROSTER. Any player from {home_full} or any other team appearing in this paragraph is a critical error.

PARAGRAPH 4 — MATCHUP ANGLE:
- Use this pre-computed signal as your foundation — do not invent a different matchup:
  {matchup_signal if matchup_signal else "No signal computed — identify the key tension from the stats tables yourself."}
- Build 2-3 sentences around that specific tension using player names and numbers from the tables.
- Do NOT reference the H2H result — it has its own dedicated section below the narrative.
- May name players from both teams.
- No generic language: no "key factor in determining the outcome", no abstract style descriptions.
- NO REPETITION: each sentence must introduce a different statistic. Do not repeat any number or player name already used earlier in this paragraph.
- CLOSING SENTENCE: the final sentence of this paragraph must contain a player name AND a specific number (stat, score, or percentage). A sentence with only descriptions and no facts is rejected.

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

Write ONLY the 4 paragraphs."""

    print(f"[NarrativeComposer] Generating narrative ...")
    response = _get_llm().invoke([HumanMessage(content=prompt)])
    narrative = response.content.strip()

    violations = _find_violations(narrative)
    if violations:
        print(f"[NarrativeComposer] WARNING — banned phrases detected (reviewer will handle): {violations}")

    print(f"[NarrativeComposer] Done ({len(narrative)} chars)")
    return {"narrative_section": narrative}
