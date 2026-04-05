"""
nodes/storylines_composer.py
────────────────────────────
Node 3 — Storylines Composer.
Queries ChromaDB for retrieved team context, generates the Storylines
section via LLM. Runs in parallel with report_composer (stage 2).
"""

import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from core.state import GraphState, extract_teams, parse_home_away
from db.chroma import get_collection

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    return _llm

_N_RESULTS = 4


def _momentum_summary(table: str, team_name: str) -> str:
    """Extract record, seed, streak, last-10, games remaining, and playoff status for a team."""
    m = re.search(
        rf"### {re.escape(team_name)}.*?Record: W (\d+) / L (\d+) \| Seed: ([^|]+)\| Streak: (\S+) \| Last 10: ([\d-]+)",
        table, re.DOTALL
    )
    if not m:
        return f"{team_name}: record unavailable"
    wins, losses, seed, streak, last10 = m.groups()
    seed = seed.strip().replace(" Conference", "").replace(" Eastern", " East").replace(" Western", " West")
    direction = "winning" if streak.startswith("W") else "losing"
    n = streak[1:]
    games_remaining = 82 - int(wins) - int(losses)
    seed_num_m = re.match(r'#(\d+)', seed)
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


def _game_signals(
    home_full: str,
    away_full: str,
    table: str,
    h2h_summary: str,
    injury_summary: str,
) -> str:
    """
    Compute grounded game-context signals from ESPN data.
    Python detects the conditions; the LLM narrates them.
    Returns a newline-separated list of signal strings.
    """
    signals = []

    # Streak direction per team
    streaks: dict[str, str] = {}
    for team in [home_full, away_full]:
        sm = re.search(rf"### {re.escape(team)}.*?Streak: (\S+)", table, re.DOTALL)
        if sm:
            streaks[team] = sm.group(1)

    # 1. Streak collision — teams going in opposite directions
    if len(streaks) == 2:
        h_s = streaks.get(home_full, "")
        a_s = streaks.get(away_full, "")
        if h_s and a_s and h_s[0] != a_s[0]:
            signals.append(
                f"STREAK COLLISION: {away_full} ({a_s}) visits {home_full} ({h_s}) — teams going in opposite directions."
            )

    # 2. Road winning streak — away team trying to extend on the road
    a_s = streaks.get(away_full, "")
    if a_s.startswith("W") and int(a_s[1:]) >= 2:
        signals.append(
            f"ROAD TEST: {away_full} brings a {a_s} winning streak into an away game."
        )

    # 3. H2H rematch — who won the last meeting, other team plays tonight
    if h2h_summary and h2h_summary.strip() not in ("H2H data unavailable.", ""):
        for team in [home_full, away_full]:
            if re.search(rf"{re.escape(team)} won", h2h_summary, re.IGNORECASE):
                loser = away_full if team == home_full else home_full
                signals.append(
                    f"H2H REMATCH: {team} won the last meeting. {loser} plays tonight looking to even the season series."
                )
                break

    # Build PPG lookup from stats table — used to filter out bench players
    ppg_lookup: dict[str, float] = {}
    for line in table.splitlines():
        if line.startswith("| ") and not line.startswith("| Player") and "---" not in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                try:
                    ppg_lookup[parts[1]] = float(parts[3])
                except (ValueError, IndexError):
                    pass

    # 4. Key injury alert — only for players averaging 10+ PPG (filters out bench players)
    _MIN_INJURY_PPG = 10.0
    for team in [home_full, away_full]:
        im = re.search(
            rf"### {re.escape(team)}\n(.*?)(?=\n###|\Z)",
            injury_summary, re.DOTALL
        )
        if not im:
            continue
        significant = []
        for entry in im.group(1).splitlines():
            entry = entry.strip().lstrip("•").strip()
            if not entry or "No injuries" in entry or "unavailable" in entry:
                continue
            player_name = re.split(r"\s*[—\-\(]", entry)[0].strip()
            if ppg_lookup.get(player_name, 0.0) >= _MIN_INJURY_PPG or player_name not in ppg_lookup:
                significant.append(entry)
        if significant:
            signals.append(f"INJURY ALERT ({team}): " + " | ".join(significant))

    return "\n".join(f"  • {s}" for s in signals) if signals else "  (none)"


def _active_roster(table: str, team_name: str) -> list[str]:
    """Return player names from the live ESPN stats table for a given team."""
    m = re.search(
        rf"### {re.escape(team_name)}\n.*?(?=\n###|\Z)",
        table, re.DOTALL
    )
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


def _retrieve(team_name: str) -> str:
    try:
        results = get_collection().query(
            query_texts=[f"{team_name} NBA season storyline arc key players"],
            n_results=_N_RESULTS,
            where={"team_name": team_name},
        )
        docs = results.get("documents", [[]])[0]
        return "\n\n".join(docs) if docs else ""
    except Exception as exc:
        print(f"[StorylinesComposer] ChromaDB query failed for {team_name}: {exc}")
        return ""


def storylines_composer_node(state: GraphState) -> dict:
    query = state["query"]

    teams = extract_teams(query)
    if len(teams) < 2:
        return {"storylines_section": ""}

    away_team, home_team = parse_home_away(query, teams)
    home_full, away_full = home_team[0], away_team[0]

    table         = state.get("player_stats_table", "")
    h2h_summary   = state.get("h2h_summary", "")
    injury_summary = state.get("injury_summary", "")

    home_momentum = _momentum_summary(table, home_full)
    away_momentum = _momentum_summary(table, away_full)
    signals       = _game_signals(home_full, away_full, table, h2h_summary, injury_summary)
    home_roster   = _active_roster(table, home_full)
    away_roster   = _active_roster(table, away_full)

    print(f"[StorylinesComposer] Retrieving context for {home_full} + {away_full} ...")
    home_ctx = _retrieve(home_full)
    away_ctx = _retrieve(away_full)

    print(f"[StorylinesComposer] === Retrieved chunks for {home_full} ===")
    print(home_ctx[:800] if home_ctx else "  (empty)")
    print(f"[StorylinesComposer] === Retrieved chunks for {away_full} ===")
    print(away_ctx[:800] if away_ctx else "  (empty)")

    if not home_ctx and not away_ctx:
        print("[StorylinesComposer] No ChromaDB content — skipping storylines.")
        return {"storylines_section": ""}

    prompt = f"""You are an NBA analyst writing for fans. Write a Storylines section for tonight's game.

MATCHUP: {away_full} (AWAY) @ {home_full} (HOME)

─── CURRENT STANDINGS (verified ESPN data — use these numbers, do not invent others) ───
{home_momentum}
{away_momentum}

─── ACTIVE ROSTERS (live ESPN data — tonight's available players only) ───
{home_full}: {", ".join(home_roster) if home_roster else "unavailable"}
{away_full}: {", ".join(away_roster) if away_roster else "unavailable"}

─── GAME CONTEXT SIGNALS (pre-computed from ESPN data — weave relevant ones into your bullets) ───
{signals}

─── {home_full} NARRATIVE CONTEXT (retrieved) ───
{home_ctx or "No context available."}

─── {away_full} NARRATIVE CONTEXT (retrieved) ───
{away_ctx or "No context available."}

Write EXACTLY 3 bullets:
1. **{home_full}** — 2-3 sentences.
2. **{away_full}** — 2-3 sentences.
3. **Matchup Angle** — 1-2 sentences. Name the specific players or factors that make this game interesting. No generic contrasts.

RULES:
- OPENING SENTENCE: Each team bullet must open with a specific player name or a concrete event — not "The [Team] are/is..." or any sentence describing what the team wants, needs, or is trying to do.
- ACTIVE ROSTER: Only name players who appear in ACTIVE ROSTERS. Any player not on that list has been traded, waived, or is otherwise no longer on this team — do not mention them.
- OUT PLAYERS: Players listed in INJURY ALERT as OUT are not playing tonight. Do not mention them as contributors, defenders, or key figures.
- Do NOT restate records, streaks, or last-10 — those are already in the briefing above. The reader already knows them.
- Where GAME CONTEXT SIGNALS apply (play-in stakes, road streak, H2H rematch, injury to a key player), weave them naturally into the relevant bullet. Ignore signals that don't apply tonight.
- Every sentence must contain at least one specific fact: a player name, a stat, a milestone, a result, or a stakes consequence.
- When the narrative context contains specific numbers (PPG, win percentages, etc.), use them — do not replace with vague phrases.
- Records and streaks from CURRENT STANDINGS may be used only for framing momentum — not as the main point of a sentence.
- Narrative facts must come from NARRATIVE CONTEXT only. Do not invent.
- Name players explicitly — never "a veteran" or "a rookie" without a name.
- MATCHUP ANGLE: Must name a specific player-vs-player or unit-vs-unit matchup grounded in a stat or context fact. Do not describe the matchup in abstract terms like "offensive firepower vs. shooting prowess" or "defense against three-point shooting."
- Dry, factual tone. No hype, no clichés.
- Forbidden: "looking to", "hoping to", "aiming to", "bounce back", "capitalize",
  "impressive", "dominant", "make a statement", "showcasing", "potential",
  "will look", "contributions", "struggling", "recent successes", "intrigue",
  "formidable", "remarkable", "contrasting momentum", "pivotal",
  "seeking to", "need to find", "will need", "must find", "establish consistency",
  "key factor in determining".
- Output only the 3 bullets. No headers, no intro line."""

    print(f"[StorylinesComposer] Generating storylines ...")
    response = _get_llm().invoke([HumanMessage(content=prompt)])
    storylines = response.content.strip()
    print(f"[StorylinesComposer] Done ({len(storylines)} chars)")
    return {"storylines_section": storylines}
