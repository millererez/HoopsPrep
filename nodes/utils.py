"""
nodes/utils.py
──────────────
Shared utilities used by narrative_composer and reviewer_node.
"""

import re as _re

# ---------------------------------------------------------------------------
# Banned phrases — single source of truth
# ---------------------------------------------------------------------------

BANNED_PHRASES = [
    "looking to", "look to", "aiming to", "hoping to", "bounce back", "capitalize",
    "will look", "seek to", "stand out", "standout", "impressive", "dominant",
    "come into", "comes into", "turn around", "highlighted by", "highlighted", "highlighting",
    "struggles", "struggled", "find their footing", "found their footing", "firing on all cylinders",
    "showcasing", "potential", "when healthy", "make a statement",
    "of their last", "of their past",
    "leading the offense in", "leading the scoring in", "led scoring in", "led the offense in",
    "averaging between", "posting between", "scoring between", "between n and",
    "bringing an average", "emerged", "has emerged", "have emerged",
    "a strong showing", "strong showing", "collective offensive", "offensive rhythm",
    "offensive output", "won all", "all ten of", "notable performances",
    "recent success", "recent form", "recent stretch", "performing well", "performed well",
    "their ability to", "has seen them", "contributing to their",
    "winning four", "winning n of", "consistent", "surge in", "effective in",
    "demonstrating", "generating points", "strong collective", "in their recent",
    "adds uncertainty", "if he suits up", "pending availability",
    "at the bottom", "which ranks", "which is nth",
    "notable performance", "contributing",
    "has shown improvement", "shown improvement", "offensive presence",
    "demonstrating", "showcasing", "displaying", "exhibiting",
    "contributions", "has been a", "relying on", "peaking at",
    "seek to", "seeking to", "need to find", "will need", "must find", "establish consistency",
    "key factor in determining", "formidable", "shooting prowess", "offensive firepower",
    "striving to", "strive to", "vie for", "vying for", "trying to maintain",
    "look to regain", "regain their", "looking to regain",
    "aim for", "aims for", "aimed for", "aim to", "aims to",
    "the stakes are high", "stakes are high for both",
    "enter this matchup", "enters this matchup", "entering this matchup",
    "enter this game", "enters this game", "entering this game",
    "comes into this", "come into this", "going into this",
    "crucial in determining", "will be crucial", "been crucial", "is crucial", "are crucial",
    "maintain their momentum", "needing to maintain",
    "faced challenges", "crucial part of their offense",
    "trending in opposite directions", "playoff implications for both",
    "moving in different directions", "significant playoff stakes",
    "in different directions", "significant implications",
    "contributes significantly", "contribute significantly",
    "establishing", "made significant strides", "significant strides",
    "transforming their season", "challenging start",
]

BANNED_PATTERNS = [
    r"\d+ of their last \d+",
    r"\d+ of \d+ games",
    r"in their last \d+ games",
    r"over the last \d+ games",
    r"in his last \d+ games",
    r"in the last \d+ games",
    r"all \d+ of their",
    r"won all \d+",
    r"winning \d+ of",
    # word-number scoring range ban only — W-L records like "8-2 in their last ten" are allowed
    r"\d+\.?\d*\s*(points?|pts).*in their last (ten|nine|eight|seven|six|five|four|three|two)",
    r"scored.*in their last (ten|nine|eight|seven|six|five|four|three|two)",
]


def fmt_num(v) -> str:
    """Format a number stripping trailing .0 from whole numbers. Accepts str or float."""
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:.1f}"
    except (ValueError, TypeError):
        return str(v)


def find_violations(text: str) -> list[str]:
    found = []
    lower = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lower:
            found.append(f'"{phrase}"')
    for pattern in BANNED_PATTERNS:
        if _re.search(pattern, lower):
            found.append(f"/{pattern}/")
    return found
