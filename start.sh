#!/bin/bash

uvicorn api:app --host 0.0.0.0 --port 8000 &

echo "Waiting for FastAPI to start..."
timeout=15
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
