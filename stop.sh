#!/usr/bin/env bash
BASE="$(cd "$(dirname "$0")" && pwd)"
PIDS="$BASE/pids"
for f in "$PIDS"/*.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f")
  kill "$pid" 2>/dev/null && echo "Stopped PID $pid ($(basename "$f" .pid))" || true
  rm "$f"
done
echo "All processes stopped."
