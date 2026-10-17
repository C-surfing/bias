#!/usr/bin/env bash
set -euo pipefail
BASE=${RSI_BASE:-/data/cjh/rsi}
: "${RSI_GPU_UUID:?Choose one idle GPU before starting.}"
mkdir -p "$BASE/logs" "$BASE/run"
if [[ -f "$BASE/run/vllm.pid" ]] && kill -0 "$(cat "$BASE/run/vllm.pid")" 2>/dev/null; then
  echo 'Recorded service process is still running; inspect it instead of starting another.'; exit 1
fi
nohup bash "$BASE/deployment/qwen3/serve.sh" > "$BASE/logs/vllm.log" 2>&1 < /dev/null &
echo "$!" > "$BASE/run/vllm.pid"
echo "Service starting. PID $(cat "$BASE/run/vllm.pid"). Inspect $BASE/logs/vllm.log and /health."
