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
    return " ".join(lines)


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

    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    total_chunks = 0

    for team_name in [home_full, away_full]:
        print(f"[ContextExtractor]  Searching: {team_name} ...")

        # Query 1: Season arc and narrative
        q1 = f'"{team_name}" NBA {CURRENT_SEASON} season storyline arc key players analysis'
        r1 = tavily.search(query=q1, max_results=3,
                           include_domains=["espn.com", "nba.com", "theathletic.com",
                                            "cbssports.com", "bleacherreport.com", "si.com"],
                           days=120, include_raw_content=True)

        # Query 2: What to watch / preview angle
        q2 = f'"{team_name}" NBA 2026 season preview what to watch storyline'
        r2 = tavily.search(query=q2, max_results=2,
                           include_domains=["espn.com", "nba.com", "cbssports.com",
                                            "bleacherreport.com", "si.com", "theathletic.com"],
                           days=120, include_raw_content=True)

        seen: set[str] = set()
        combined = ""
        for result in r1.get("results", []) + r2.get("results", []):
            url = result.get("url", "")
            if url in seen:
                continue
            seen.add(url)
            title = result.get("title", "")
            # prefer raw_content (full page) over snippet
            content = result.get("raw_content") or result.get("content", "")
            combined += f"\n{title}\n{content}\n"
            print(f"  → {title[:80]} ({len(content)} chars)")

        if combined.strip():
            combined = _clean(combined)
            n = _upsert(team_name, combined.strip(), today)
            total_chunks += n
            print(f"[ContextExtractor]  Upserted {n} chunks for {team_name}")
        else:
            print(f"[ContextExtractor]  No Tavily content for {team_name}")

    return {"team_narrative_bullets": f"[RAG: {total_chunks} chunks ingested for {home_full} + {away_full}]"}
