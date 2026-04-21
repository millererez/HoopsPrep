"""
nodes/context_extractor.py
──────────────────────────
Node 2 — Context Extractor.
Searches Tavily for team narrative content, chunks, embeds via OpenAI,
and upserts into ChromaDB for RAG retrieval by storylines_composer.
"""

import hashlib
import os
import re
from datetime import datetime

from tavily import TavilyClient

from core.state import GraphState, CURRENT_SEASON, extract_teams, parse_home_away
from db.chroma import get_collection

_CHUNK_CHARS = 500


def _clean(text: str) -> str:
    """Strip HTML/markdown noise, keeping only prose sentences."""
    # markdown links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # bare URLs
    text = re.sub(r'https?://\S+', '', text)
    # markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # inline source tags like "www.si.com•20h" or "site.com•1d"
    text = re.sub(r'\S+\.\S+•\S+', '', text)
    # lines that are pure nav/UI (short lines with | or all-caps or no letters)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) < 30 and not re.search(r'[a-z]{4,}', stripped):
            continue
        if stripped.count('|') >= 3:
            continue
        lines.append(stripped)
    text = " ".join(lines)

    # Strip all icon/logo navigation (e.g., "!Boston Celtics Logo", "!NBA Store Icon")
    text = re.sub(r'![\w\s]+(Logo|Icon)\s*', '', text)
    # Strip "Last Ladder: No. N" and similar stats-table artifacts
    text = re.sub(r'Last Ladder:.*', '', text)

    # Strip sentences containing article source-isms that the LLM tends to echo verbatim
    _SOURCE_ISMS = re.compile(
        r'\b(here at|click here|subscribe|sign up|follow us|read more|'
        r'more on this|earlier this week|last week on|our staff|we asked|'
        r'you can find|check out|tune in|stay tuned)\b'
        r'|^(this summer|last summer|this offseason|last offseason|in the offseason)',
        re.IGNORECASE | re.MULTILINE,
    )
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s for s in sentences if not _SOURCE_ISMS.search(s)]
    return " ".join(sentences)


def _chunk(text: str) -> list[str]:
    words = text.split()
    chunks, current, length = [], [], 0
    overlap = 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= _CHUNK_CHARS:
            chunks.append(" ".join(current))
            overlap = max(1, len(current) // 5)
            current = current[-overlap:]
            length = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _upsert(team_name: str, content: str, today: str) -> int:
    chunks = _chunk(content)
    ids, docs, metas = [], [], []
    for i, chunk in enumerate(chunks):
        doc_id = hashlib.md5(f"{team_name}:{today}:{i}:{chunk[:40]}".encode()).hexdigest()
        ids.append(doc_id)
        docs.append(chunk)
        metas.append({"team_name": team_name, "date": today, "chunk_index": i})
    if ids:
        get_collection().upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def context_extractor_node(state: GraphState) -> dict:
    query = state["query"]
    today = datetime.now().strftime("%Y-%m-%d")

    teams = extract_teams(query)
    if len(teams) >= 2:
        away_team, home_team = parse_home_away(query, teams)
        home_full, away_full = home_team[0], away_team[0]
    else:
        home_full = teams[0][0] if teams else "Team 1"
        away_full = "Team 2"

    try:
        tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    except Exception as e:
        print(f"[ContextExtractor] ❌ Failed to initialize Tavily client: {e}")
        return {"team_narrative_bullets": "[RAG: 0 chunks ingested due to missing API key]"}

    total_chunks = 0
    is_playoff_window = datetime.now().month in {4, 5, 6}

    for team_name in [home_full, away_full]:
        print(f"[ContextExtractor]  Searching: {team_name} ...")

        if is_playoff_window:
            q1 = f'"{team_name}" NBA 2026 playoffs first round series preview players'
        else:
            q1 = f'"{team_name}" NBA 2026 recent games season players performance standings'

        try:
            r1 = tavily.search(query=q1, max_results=3,
                               include_domains=["espn.com", "nba.com", "theathletic.com",
                                                "cbssports.com", "bleacherreport.com", "si.com"],
                               days=45, include_raw_content=True)
        except Exception as e:
            print(f"[ContextExtractor] ❌ Tavily API Error (main search): {e}")
            r1 = {}

        seen: set[str] = set()
        combined = ""
        team_lower  = team_name.lower()
        short_name  = team_name.split()[-1].lower()
        
        for result in r1.get("results", []):
            url = result.get("url", "")
            if url in seen:
                continue
            seen.add(url)
            title = result.get("title", "")
            raw = result.get("raw_content") or result.get("content", "")
            content = raw[:8000] if raw else ""

            mentions = content.lower().count(team_lower) + content.lower().count(short_name)
            if mentions < 3:
                print(f"  → SKIPPED ({mentions} mentions): {title[:70]}")
                continue

            combined += f"\n{title}\n{content}\n"
            print(f"  → {title[:80]} ({len(content)} chars, {mentions} mentions)")

        if combined.strip():
            combined = _clean(combined)
            if len(combined) < 1500:
                print(f"[ContextExtractor]  Thin content for {team_name} ({len(combined)} chars) — retrying ...")
                if is_playoff_window:
                    q_fallback = f"{team_name} NBA 2026 playoffs series players preview"
                else:
                    q_fallback = f"{team_name} NBA 2026 season players games"
                try:
                    r_fallback = tavily.search(
                        query=q_fallback, max_results=3,
                        include_domains=["espn.com", "nba.com", "cbssports.com",
                                         "bleacherreport.com", "si.com", "theathletic.com"],
                        days=45, include_raw_content=True,
                    )
                except Exception as e:
                    print(f"[ContextExtractor] ❌ Tavily API Error (fallback search): {e}")
                    r_fallback = {}
                    
                for result in r_fallback.get("results", []):
                    url = result.get("url", "")
                    if url in seen:
                        continue
                    seen.add(url)
                    title = result.get("title", "")
                    raw = result.get("raw_content") or result.get("content", "")
                    content = raw[:8000] if raw else ""
                    mentions = content.lower().count(team_lower) + content.lower().count(short_name)
                    if mentions < 3:
                        print(f"  → [fallback] SKIPPED ({mentions} mentions): {title[:70]}")
                        continue
                    combined += f"\n{title}\n{content}\n"
                    print(f"  → [fallback] {title[:80]} ({len(content)} chars, {mentions} mentions)")
                combined = _clean(combined)

            try:
                # Catching OpenAI quota errors during embedding creation
                n = _upsert(team_name, combined.strip(), today)
                total_chunks += n
                print(f"[ContextExtractor]  Upserted {n} chunks for {team_name}")
            except Exception as e:
                print(f"[ContextExtractor] ❌ Upsert/Embedding Error (Check OpenAI Quota): {e}")
        else:
            print(f"[ContextExtractor]  No Tavily content for {team_name}")

    return {"team_narrative_bullets": f"[RAG: {total_chunks} chunks ingested for {home_full} + {away_full}]"}