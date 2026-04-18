#!/bin/bash

# ── Auto-clear today's cache on new deployment ───────────────────────────────
VERSION_FILE="/data/.last_deploy_commit"
CURRENT_COMMIT="${RENDER_GIT_COMMIT:-unknown}"

echo "[Boot] Checking deployment version..."
echo "[Boot] Current commit: $CURRENT_COMMIT"

if [ -f "$VERSION_FILE" ]; then
    LAST_COMMIT=$(cat "$VERSION_FILE")
    echo "[Boot] Last seen commit: $LAST_COMMIT"
else
    LAST_COMMIT="none"
    echo "[Boot] No previous commit found on disk."
fi

# Force clear if commits don't match OR if the commit is unknown
if [ "$LAST_COMMIT" = "$CURRENT_COMMIT" ] && [ "$CURRENT_COMMIT" != "unknown" ]; then
    echo "[Boot] Same deployment detected ($CURRENT_COMMIT) — keeping cache."
else
    echo "[Boot] New deployment detected — clearing cache!"
    
    DB_PATH=${CACHE_DB_PATH:-"/data/cache.db"}
    
    if [ -f "$DB_PATH" ]; then
        sqlite3 "$DB_PATH" "DELETE FROM report_cache;"
        echo "[Boot] Cache cleared successfully via sqlite3."
    else
        echo "[Boot] Database file not found at $DB_PATH — nothing to clear yet."
    fi
    
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