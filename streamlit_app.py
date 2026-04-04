import os
import requests
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

API_KEY  = os.environ.get("HOOPSPREP_API_KEY", "")
BASE_URL = "http://127.0.0.1:8001"
HEADERS  = {"X-API-Key": API_KEY}


def fetch_games() -> list[dict]:
    resp = requests.get(f"{BASE_URL}/games/tonight", headers=HEADERS, timeout=15)
    if resp.status_code == 401:
        st.error("Invalid API key — check your .env file")
        return []
    resp.raise_for_status()
    return resp.json().get("games", [])


def fetch_briefing(game_id: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/briefing",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"game_id": game_id},
        timeout=120,
    )
    if resp.status_code == 401:
        raise ValueError("Invalid API key — check your .env file")
    if resp.status_code == 404:
        raise ValueError("Game not found in tonight's schedule")
    if resp.status_code == 500:
        raise ValueError("Report generation failed — check server logs")
    resp.raise_for_status()
    return resp.json()


def parse_report(report: str) -> tuple[str, str, str, str, str]:
    """Split report into (prose, storylines, injury_block, h2h_block, stats_block)."""
    # Storylines block (optional)
    if "\n\nStorylines:\n\n" in report:
        prose, rest = report.split("\n\nStorylines:\n\n", 1)
    else:
        prose, rest = report, ""

    if "\n\nInjury Report:\n\n" in rest:
        storylines, rest = rest.split("\n\nInjury Report:\n\n", 1)
    elif "\n\nInjury Report:\n\n" in prose:
        # No storylines — prose contains everything up to injury
        prose, rest = prose.split("\n\nInjury Report:\n\n", 1)
        storylines = ""
    else:
        return prose, rest, "", "", ""

    if "\n\nH2H This Season:\n\n" in rest:
        injury_block, rest = rest.split("\n\nH2H This Season:\n\n", 1)
    else:
        injury_block, rest = rest, ""

    parts = rest.split("\n\n### ", 1)
    h2h_block   = parts[0]
    stats_block = ("### " + parts[1]) if len(parts) > 1 else ""
    return prose, storylines, injury_block, h2h_block, stats_block


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="HoopsPrep", page_icon="🏀", layout="centered")

st.title("HoopsPrep")
st.caption("NBA Pre-Game Broadcast Briefing")
st.divider()

# ── Load games (once per session) ────────────────────────────────────────────

if "games" not in st.session_state:
    with st.spinner("Loading tonight's games..."):
        try:
            st.session_state.games = fetch_games()
        except Exception as e:
            st.error(f"Could not load games: {e}")
            st.session_state.games = []

games = st.session_state.games

# ── Game selector ─────────────────────────────────────────────────────────────

st.subheader("Tonight's Games")

if not games:
    st.info("No games scheduled for tonight.")
    st.stop()

options = {
    f"{g['away_team']} @ {g['home_team']} — {g['tip_off_est']}": g["game_id"]
    for g in games
}

selected_label = st.selectbox("Select a game", list(options.keys()))
selected_game_id = options[selected_label]

# ── Generate button ───────────────────────────────────────────────────────────

if st.button("Generate Briefing", type="primary"):
    st.session_state.pop("report_data", None)  # clear previous result
    with st.spinner("Generating report... this takes ~45 seconds"):
        try:
            st.session_state.report_data = fetch_briefing(selected_game_id)
        except ValueError as e:
            st.error(str(e))
        except requests.exceptions.Timeout:
            st.error("Request timed out — the server may be overloaded.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── Report display ────────────────────────────────────────────────────────────

if "report_data" in st.session_state:
    data = st.session_state.report_data
    prose, storylines, injury_block, h2h_block, stats_block = parse_report(data["report"])

    st.divider()
    st.subheader(f"{data['away_team']} @ {data['home_team']}")

    # Prose paragraphs
    st.markdown(prose)

    # Storylines
    if storylines:
        st.divider()
        st.markdown("**Storylines**")
        st.markdown(storylines)

    # Injury report
    if injury_block:
        st.divider()
        st.markdown("**Injury Report**")
        st.markdown(injury_block)

    # H2H
    if h2h_block:
        st.divider()
        st.markdown("**H2H This Season**")
        st.markdown(h2h_block)

    # Stats tables
    if stats_block:
        st.divider()
        st.markdown(stats_block)
