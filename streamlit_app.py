import os
import requests
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

API_KEY  = os.environ.get("HOOPSPREP_API_KEY", "")
BASE_URL = os.environ.get("API_URL", "http://localhost:8000")
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


def parse_report(report: str) -> tuple[str, str, str, str]:
    """Split report into (narrative, injury_block, h2h_block, stats_block)."""
    if "\n\nInjury Report:\n\n" in report:
        narrative, rest = report.split("\n\nInjury Report:\n\n", 1)
    else:
        return report, "", "", ""

    if "\n\nH2H This Season:\n\n" in rest:
        injury_block, rest = rest.split("\n\nH2H This Season:\n\n", 1)
    else:
        injury_block, rest = rest, ""

    parts = rest.split("\n\n### ", 1)
    h2h_block   = parts[0]
    stats_block = ("### " + parts[1]) if len(parts) > 1 else ""
    return narrative, injury_block, h2h_block, stats_block


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="HoopsPrep", page_icon="🏀", layout="centered")

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f1923 !important;
    color: #e8eaf0 !important;
}
[data-testid="stSidebar"] { background-color: #0a1118 !important; }
[data-testid="stHeader"]  { background-color: #0f1923 !important; border-bottom: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding-top: 1rem !important; }

/* ── Typography ── */
h1, h2, h3, h4 { color: #ffffff !important; }
p, li, span     { color: #e8eaf0; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stSelectbox"] > div > div {
    background-color: #1a2535 !important;
    border: 1px solid #2d3f55 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

/* ── Primary button (Generate) ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #F5821E, #e06b10) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.03em !important;
    width: 100%;
    transition: opacity 0.2s;
}
[data-testid="stButton"] > button[kind="primary"]:hover:not(:disabled) {
    opacity: 0.88 !important;
}
[data-testid="stButton"] > button[kind="primary"]:disabled {
    background: linear-gradient(135deg, #3a3a3a, #2a2a2a) !important;
    color: #666666 !important;
    opacity: 1 !important;
    cursor: not-allowed !important;
}

/* ── Secondary button (Copy) ── */
[data-testid="stButton"] > button[kind="secondary"] {
    background-color: #1a2535 !important;
    color: #94a3b8 !important;
    border: 1px solid #2d3f55 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #F5821E !important;
    color: #F5821E !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p { color: #94a3b8 !important; }

/* ── Divider ── */
hr { border-color: #2d3f55 !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── Code / monospace ── */
code, pre { background-color: #1a2535 !important; color: #e8eaf0 !important; }

/* ── Markdown tables ── */
[data-testid="stMarkdownContainer"] table {
    border-collapse: collapse !important;
    width: 100%;
}
[data-testid="stMarkdownContainer"] th,
[data-testid="stMarkdownContainer"] td {
    border: 1px solid #4a6080 !important;
    padding: 0.4rem 0.7rem !important;
    color: #e8eaf0 !important;
}
[data-testid="stMarkdownContainer"] th {
    background-color: #1e3248 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
[data-testid="stMarkdownContainer"] tr:nth-child(even) td {
    background-color: #162030 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 1.5rem 0 0.5rem 0; text-align: center;">
  <div style="font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em; color: #ffffff;">
    🏀 HoopsPrep
  </div>
  <div style="font-size: 0.9rem; color: #94a3b8; margin-top: 0.35rem;
              letter-spacing: 0.1em; text-transform: uppercase;">
    NBA Pre-Game Broadcast Intelligence
  </div>
</div>
<div style="height: 3px; background: linear-gradient(90deg, transparent, #F5821E, transparent);
            margin: 0.8rem 0 1.8rem 0; border-radius: 2px;"></div>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────

if "games" not in st.session_state: 
    st.session_state.games = None
if "is_generating" not in st.session_state: 
    st.session_state.is_generating = False


# ── Load games (once per session) ────────────────────────────────────────────

if st.session_state.games is None:
    with st.spinner("Loading tonight's games..."):
        try:
            st.session_state.games = fetch_games()
        except Exception as e:
            st.error(f"Could not load games: {e}")
            st.session_state.games = []

games = st.session_state.games

if not games:
    st.markdown("""
    <div style="text-align:center; padding: 2rem; color: #94a3b8;">
      No games scheduled for tonight.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Game selector + Generate ──────────────────────────────────────────────────

options = {
    f"{g['away_team']}  @  {g['home_team']}  —  {g['tip_off_est']}": g["game_id"]
    for g in games
}
def lock_generate_button():
    st.session_state.is_generating = True

with st.form("generate_form"):
    selected_label = st.selectbox("Tonight's Games", list(options.keys()))
    st.markdown("<div style='height: 0.4rem'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "▶  Generate Briefing",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_generating,  
        on_click=lock_generate_button            
    )

if submitted:
    selected_game_id = options[selected_label]
    st.session_state.pop("report_data", None)
    with st.spinner("Preparing your report, please wait"):
        try:
            st.session_state.report_data = fetch_briefing(selected_game_id)
        except ValueError as e:
            st.error(str(e))
        except requests.exceptions.Timeout:
            st.error("Request timed out — the server may be overloaded.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
        finally:
            st.session_state.is_generating = False
            
    st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

def section_card(title: str, content: str, accent: str, icon: str = ""):
    """Render a styled card with a colored left-border accent."""
    # escape any < > in the content for safe HTML injection, but keep newlines
    import html as _html
    safe_content = _html.escape(content)
    st.markdown(f"""
<div style="
    background: #1a2535;
    border-left: 4px solid {accent};
    border-radius: 0 10px 10px 0;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
">
  <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
              text-transform: uppercase; color: {accent}; margin-bottom: 0.8rem;">
    {icon}&nbsp; {title}
  </div>
  <div style="color: #e8eaf0; line-height: 1.75; white-space: pre-wrap; font-size: 0.97rem;">{safe_content}</div>
</div>
""", unsafe_allow_html=True)


# ── Report display ────────────────────────────────────────────────────────────

if "report_data" in st.session_state:
    data = st.session_state.report_data
    narrative, injury_block, h2h_block, stats_block = parse_report(data["report"])

    # Matchup header
    st.markdown(f"""
<div style="text-align: center; padding: 1.6rem 0 0.8rem 0; margin-top: 1.2rem;
            border-top: 1px solid #2d3f55;">
  <div style="font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.12em;
              text-transform: uppercase; margin-bottom: 0.5rem;">
    Broadcast Briefing
  </div>
  <div style="font-size: 1.65rem; font-weight: 800; color: #ffffff; letter-spacing: -0.01em;">
    {data['away_team']}&nbsp;&nbsp;<span style="color: #F5821E;">@</span>&nbsp;&nbsp;{data['home_team']}
  </div>
</div>
""", unsafe_allow_html=True)

    # Narrative card
    section_card("Narrative", narrative, accent="#F5821E", icon="📋")

    # Injury card
    if injury_block:
        section_card("Injury Report", injury_block, accent="#F59E0B", icon="🩹")

    # H2H card
    if h2h_block:
        section_card("H2H This Season", h2h_block, accent="#3B82F6", icon="📊")

    # Stats — label then markdown (can't nest st.markdown inside an HTML div)
    if stats_block:
        st.markdown("""
<div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase; color: #10B981; margin: 1.4rem 0 0.4rem 0;">
  📈&nbsp; Player Stats
</div>
<hr style="border-color: #10B981; border-width: 1px; margin: 0 0 0.8rem 0;">
""", unsafe_allow_html=True)

        # 1. Split the block by team (using the markdown header "### ")
        team_sections = stats_block.split("### ")
        
        for section in team_sections:
            if not section.strip():
                continue
                
            # Re-attach the header we split by
            full_section = "### " + section
            lines = full_section.split('\n')
            
            clean_lines = []
            emergency_alerts = []
            
            # 2. Extract alerts for THIS specific team
            for line in lines:
                lower_line = line.lower()
                if ("emergency roster" in lower_line or "g league" in lower_line or "10-day" in lower_line) and "|" not in line:
                    clean_text = line.replace("🚨", "").replace("**", "").strip()
                    if clean_text:
                        emergency_alerts.append(clean_text)
                else:
                    clean_lines.append(line)
            
            # 3. Find exactly where the table starts so we can inject the warning right before it
            table_start_idx = 0
            for i, line in enumerate(clean_lines):
                if line.startswith("|"):
                    table_start_idx = i
                    break
            
            # 4. Render the team section
            if emergency_alerts:
                # Print team info (Record, Team Stats)
                st.markdown("\n".join(clean_lines[:table_start_idx]))
                
                # Print the warning
                for alert in emergency_alerts:
                    st.warning(alert, icon="🚨")
                    
                # Print the table
                st.markdown("\n".join(clean_lines[table_start_idx:]))
            else:
                # No emergency for this team, just print normally
                st.markdown("\n".join(clean_lines))
    
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding-top: 1.5rem; border-top: 1px dashed #2d3f55;">
      <span style="font-size: 1.15rem; font-weight: 700; color: #F5821E; letter-spacing: 0.05em; text-transform: uppercase;">
        Enjoy the game! 🏀
      </span>
    </div>
    """, unsafe_allow_html=True)