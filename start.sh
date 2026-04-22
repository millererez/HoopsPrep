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
        MAX_RETRIES=3
        RETRY_COUNT=0
        SUCCESS=false

        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            ((RETRY_COUNT++))
            echo "[Boot] Attempting to clear cache (Attempt $RETRY_COUNT/$MAX_RETRIES)..."
            
            # Try to delete. We use .timeout 2000 inside for extra safety.
            SQL_OUTPUT=$(sqlite3 "$DB_PATH" ".timeout 2000" "DELETE FROM report_cache; SELECT changes();" 2>&1)
            EXIT_CODE=$?

            if [ $EXIT_CODE -eq 0 ]; then
                echo "[Boot] SUCCESS: Cache cleared ($SQL_OUTPUT rows deleted)."
                SUCCESS=true
                break
            else
                echo "[Boot] Attempt $RETRY_COUNT failed: $SQL_OUTPUT"
                if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
                    echo "[Boot] Waiting 2 seconds before next attempt..."
                    sleep 2
                fi
            fi
        done

        if [ "$SUCCESS" = false ]; then
            echo "[Boot] FATAL: Could not clear cache after $MAX_RETRIES attempts. Moving on with existing cache."
        fi
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