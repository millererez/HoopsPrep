#!/bin/bash

# ── Auto-clear today's cache on new deployment ───────────────────────────────
# Render sets RENDER_GIT_COMMIT on every deploy. Compare against the last
# seen commit stored on the persistent disk. If different → new deploy →
# delete only today's rows (yesterday's are never served; the file stays).
VERSION_FILE="/data/.last_deploy_commit"
CURRENT_COMMIT="${RENDER_GIT_COMMIT:-unknown}"

if [ -f "$VERSION_FILE" ] && [ "$(cat "$VERSION_FILE")" = "$CURRENT_COMMIT" ]; then
    echo "Same deployment ($CURRENT_COMMIT) — keeping cache"
else
    echo "New deployment detected ($CURRENT_COMMIT) — clearing today's cache entries"
    python3 - <<'EOF'
import sqlite3, datetime, os
db = os.environ.get("CACHE_DB_PATH", "/data/cache.db")
if os.path.exists(db):
    today = datetime.date.today().isoformat()
    conn = sqlite3.connect(db)
    deleted = conn.execute("DELETE FROM report_cache WHERE date = ?", (today,)).rowcount
    conn.commit()
    conn.close()
    print(f"  Removed {deleted} cached report(s) for {today}")
else:
    print("  No cache file yet — nothing to clear")
EOF
    echo "$CURRENT_COMMIT" > "$VERSION_FILE"
fi
# ─────────────────────────────────────────────────────────────────────────────

uvicorn api:app --host 0.0.0.0 --port 8000 &

echo "Waiting for FastAPI to start..."
timeout=60
while ! curl -s http://localhost:8000/health > /dev/null; do
  sleep 1
  ((timeout--))
  if [ "$timeout" -eq 0 ]; then
    echo "Error: FastAPI took too long to start!"
    exit 1
  fi
done
echo "FastAPI is up!"

export API_URL="http://localhost:8000"

streamlit run streamlit_app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true
