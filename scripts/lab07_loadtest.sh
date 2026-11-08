#!/usr/bin/env bash
# Lab 7 load test with oha: def vs async def vs to_thread, 1 vs 4 workers.
#   bash scripts/lab07_loadtest.sh data/index.pkl            (needs `oha` on PATH)
#   DEPLOYED=https://findex-xxxx.onrender.com bash scripts/lab07_loadtest.sh data/index.pkl
set -euo pipefail
INDEX=${1:-data/index.pkl}
OHA=${OHA:-oha}
DURATION=${DURATION:-15s}
Q="q=python%20async%20await&k=10"
PORT=8811

run() {  # label url concurrency  -- plus a 1-client /health probe running alongside
  local base="${2%%/search*}"; base="${base%%/_bench*}"
  "$OHA" -z "$DURATION" -c 1 -q 10 --no-tui --output-format json "$base/health" > /tmp/oha-health.json &
  local probe=$!
  "$OHA" -z "$DURATION" -c "$3" --no-tui --output-format json "$2" > /tmp/oha.json
  wait $probe
  python3 - "$1" "$3" <<'PY'
import json, sys
d = json.load(open("/tmp/oha.json"))
h = json.load(open("/tmp/oha-health.json"))["latencyPercentiles"]
s, p = d["summary"], d["latencyPercentiles"]
print(f"| {sys.argv[1]} | /search | {sys.argv[2]} | {s['requestsPerSec']:.0f} | "
      f"{p['p50']*1000:.0f} | {p['p95']*1000:.0f} | {p['p99']*1000:.0f} | "
      f"{h['p50']*1000:.0f} / {h['p99']*1000:.0f} |")
PY
}

start() {  # workers
  ENABLE_BENCH_ROUTES=1 LOG_LEVEL=WARNING findex serve --index "$INDEX" --port $PORT \
    --workers "$1" >/dev/null 2>&1 &
  SERVER=$!
  for _ in $(seq 50); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 0.3; done
  for _ in $(seq 20); do curl -sf "http://127.0.0.1:$PORT/search?$Q" >/dev/null; done  # warm up
}

echo "| Target | Route | Concurrency | RPS | p50 (ms) | p95 | p99 | /health p50 / p99 under load (ms) |"
echo "|---|---|---|---|---|---|---|---|"
start 1
run "local, 1 worker, \`async def\` (CPU in the loop)" "http://127.0.0.1:$PORT/_bench/search-async?$Q" 50
run "local, 1 worker, \`async def\` + \`to_thread\`" "http://127.0.0.1:$PORT/_bench/search-to-thread?$Q" 50
run "local, 1 worker, \`def\` (thread pool)" "http://127.0.0.1:$PORT/search?$Q" 50
kill $SERVER; wait $SERVER 2>/dev/null || true
start 4
run "local, 4 workers, \`def\`" "http://127.0.0.1:$PORT/search?$Q" 50
kill $SERVER; wait $SERVER 2>/dev/null || true
if [[ -n "${DEPLOYED:-}" ]]; then
  run "deployed ($DEPLOYED)" "$DEPLOYED/search?$Q" 20
fi
