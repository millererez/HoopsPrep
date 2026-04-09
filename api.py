"""
api.py
──────
HoopsPrep FastAPI application.

Endpoints:
  GET  /games/tonight  — tonight's NBA games from ESPN scoreboard
  POST /briefing       — generate a full pre-game report for a selected game

Security:
  - X-API-Key header required on every request
  - Rate limiting via slowapi (10/min on games, 3/min on briefing)
"""

import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

from main import build_graph
from core.state import GraphState, EST
from db.cache import get_cached, set_cached

# ---------------------------------------------------------------------------
# App + rate limiter setup
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="HoopsPrep API",
    description="NBA pre-game broadcast briefing generator.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(key: str = Depends(api_key_header)) -> str:
    expected = os.environ.get("HOOPSPREP_API_KEY", "")
    if not key or key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return key

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Game(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    tip_off_utc: str          # ISO-8601 UTC string from ESPN
    tip_off_est: str          # Human-readable Eastern time

class TonightGamesResponse(BaseModel):
    date: str                 # YYYY-MM-DD Eastern date
    games: list[Game]

class BriefingRequest(BaseModel):
    game_id: str

class BriefingResponse(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    report: str

# ---------------------------------------------------------------------------
# ESPN scoreboard helper
# ---------------------------------------------------------------------------

_ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)

def _fetch_tonight_games() -> list[Game]:
    """Fetch today's NBA games from ESPN scoreboard API."""
    today_date = datetime.now(tz=EST).strftime("%Y%m%d")
    resp = requests.get(_ESPN_SCOREBOARD_URL, params={"dates": today_date}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games: list[Game] = []
    for event in data.get("events", []):
        comp        = event["competitions"][0]
        game_id     = event["id"]
        date_utc    = event["date"]                     # e.g. "2026-04-03T23:30Z"

        utc_dt      = datetime.fromisoformat(date_utc.replace("Z", "+00:00"))
        est_dt      = utc_dt.astimezone(EST)
        tip_est_str = est_dt.strftime("%I:%M %p ET").lstrip("0")

        home_team = away_team = ""
        for competitor in comp["competitors"]:
            if competitor["homeAway"] == "home":
                home_team = competitor["team"]["displayName"]
            else:
                away_team = competitor["team"]["displayName"]

        games.append(Game(
            game_id     = game_id,
            home_team   = home_team,
            away_team   = away_team,
            tip_off_utc = date_utc,
            tip_off_est = tip_est_str,
        ))

    return games


def _get_game_by_id(game_id: str):
    """Return the Game object for a given ID if it's in tonight's schedule."""
    games = _fetch_tonight_games()
    for g in games:
        if g.game_id == game_id:
            return g
    return None

# ---------------------------------------------------------------------------
# Graph (built once at startup)
# ---------------------------------------------------------------------------

graph = build_graph()

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.get(
    "/games/tonight",
    response_model=TonightGamesResponse,
    summary="List tonight's NBA games",
)
@limiter.limit("10/minute")
def get_games_tonight(
    request: Request,
    _key: str = Depends(verify_api_key),
):
    """
    Returns all NBA games scheduled for tonight (Eastern date).
    Use the returned `game_id` to request a briefing.
    """
    try:
        games = _fetch_tonight_games()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ESPN scoreboard unavailable: {exc}")

    today_est = datetime.now(tz=EST).strftime("%Y-%m-%d")
    return TonightGamesResponse(date=today_est, games=games)


@app.post(
    "/briefing",
    response_model=BriefingResponse,
    summary="Generate a pre-game briefing",
)
@limiter.limit("3/minute")
def post_briefing(
    request: Request,
    body: BriefingRequest,
    _key: str = Depends(verify_api_key),
):
    """
    Generate a full broadcast briefing for a game from tonight's schedule.
    Provide a `game_id` from `GET /games/tonight`.
    """
    game = _get_game_by_id(body.game_id)
    if game is None:
        raise HTTPException(
            status_code=404,
            detail="Game ID not found in tonight's schedule. Use GET /games/tonight for valid IDs.",
        )

    query = (
        f"Prepare a pre-game briefing for "
        f"{game.away_team} at {game.home_team}"
    )

    cached = get_cached(body.game_id)
    if cached:
        print(f"[API]   Cache hit for game {body.game_id}")
        return BriefingResponse(
            game_id   = game.game_id,
            home_team = game.home_team,
            away_team = game.away_team,
            report    = cached,
        )

    initial_state: GraphState = {
        "query":                  query,
        "player_stats_table":     "",
        "h2h_summary":            "",
        "injury_summary":         "",
        "recent_form":            "",
        "team_narrative_bullets": "",
        "stakes_context":         "",
        "narrative_section":      "",
        "review_issues":          "",
        "final_report":           "",
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")

    report = result["final_report"]
    set_cached(body.game_id, report)

    return BriefingResponse(
        game_id   = game.game_id,
        home_team = game.home_team,
        away_team = game.away_team,
        report    = report,
    )
