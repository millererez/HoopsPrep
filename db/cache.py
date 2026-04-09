"""
db/cache.py
───────────
SQLite report cache — keyed by (game_id, date).
Same game requested twice on the same day returns the stored report
without re-running the pipeline.
"""

import os
import sqlite3
from datetime import date

_DB_PATH = os.environ.get(
    "CACHE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "cache.db"),
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS report_cache (
            game_id TEXT NOT NULL,
            date    TEXT NOT NULL,
            report  TEXT NOT NULL,
            PRIMARY KEY (game_id, date)
        )
    """)
    conn.commit()
    return conn


def get_cached(game_id: str) -> str | None:
    today = date.today().isoformat()
    with _conn() as conn:
        row = conn.execute(
            "SELECT report FROM report_cache WHERE game_id = ? AND date = ?",
            (game_id, today),
        ).fetchone()
    return row[0] if row else None


def set_cached(game_id: str, report: str) -> None:
    today = date.today().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO report_cache (game_id, date, report) VALUES (?, ?, ?)",
            (game_id, today, report),
        )
        conn.commit()
