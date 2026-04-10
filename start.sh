#!/usr/bin/env bash
# ~/qdrant/start.sh
set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
PIDS="$BASE/pids"
mkdir -p "$PIDS"

echo "==> Starting Qdrant..."
"$BASE/qdrant" --config-path "$BASE/config.yaml" &>/tmp/qdrant.log &
echo $! > "$PIDS/qdrant.pid"
until curl -sf http://localhost:6333/healthz &>/dev/null; do sleep 0.5; done
echo "    Qdrant up."

echo "==> Warming up Ollama model..."
curl -sf -X POST http://localhost:11434/api/pull \
  -d "{\"name\": \"$(cd "$BASE" && python -c 'import config; print(config.EMBEDDING_MODEL)')\"}" \
  -H "Content-Type: application/json" | tail -1
echo "    Ollama ready."

echo "==> Running init_schema..."
cd "$BASE" && python init_schema.py

echo "==> Starting FastAPI backend..."
uvicorn backend.main:app --workers 1 --port 8000 --host 127.0.0.1 \
  &>/tmp/brain-backend.log &
echo $! > "$PIDS/backend.pid"
until curl -sf http://localhost:8000/status &>/dev/null; do sleep 0.5; done
echo "    Backend up."

echo "==> Starting watcher daemon..."
python "$BASE/ingestion/watcher.py" &>/tmp/brain-watcher.log &
echo $! > "$PIDS/watcher.pid"

echo ""
echo "Second Brain running:"
echo "  Search UI: http://localhost:8000/app"
echo "  API:       http://localhost:8000"
echo "  Logs:      /tmp/qdrant.log  /tmp/brain-backend.log  /tmp/brain-watcher.log"
