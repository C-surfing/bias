#!/usr/bin/env bash
# Stop only the recorded server process owned by this user.
set -euo pipefail
BASE=${RSI_BASE:-/data/cjh/rsi}
PID=$(cat "$BASE/run/vllm.pid")
[[ "$PID" =~ ^[0-9]+$ ]] || exit 1
[[ $(stat -c %U "/proc/$PID") == "$(id -un)" ]] || exit 1
CMD=$(tr '\0' ' ' < "/proc/$PID/cmdline")
[[ "$CMD" == *"$BASE/envs/inference/bin/"* && "$CMD" == *"$BASE/models/Qwen3-8B"* ]] || { echo 'PID no longer identifies this model service.'; exit 1; }
kill -TERM "$PID"
echo 'Termination requested for this account model service only.'
