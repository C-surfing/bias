#!/usr/bin/env bash
# Run a DeepSeek scientific episode without printing or storing the API key.
set -euo pipefail
cd /data/cjh/rsi/project
SECRET_FILE=/data/cjh/rsi/.secrets/deepseek.key
if [[ -z "${DEEPSEEK_API_KEY:-}" && -f "$SECRET_FILE" ]]; then
  DEEPSEEK_API_KEY="$(cat "$SECRET_FILE")"
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  if [[ -t 0 ]]; then
    read -r -s -p 'DeepSeek API key: ' DEEPSEEK_API_KEY
    printf '\n' >&2
  else
    IFS= read -r DEEPSEEK_API_KEY
    DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY%$'\r'}"
  fi
fi
# PowerShell-to-SSH transfers can prefix a Unicode BOM (U+FEFF).
# Remove it before constructing the Authorization request header.
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY#$'\xef\xbb\xbf'}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY%$'\r'}"
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo 'No DeepSeek API key available in this Linux process' >&2
  exit 2
fi
export DEEPSEEK_API_KEY
trap 'unset DEEPSEEK_API_KEY' EXIT
OUTPUT="${1:-runs/discovery-bias-relevance-20261026/deepseek-relevance-v1}"
/data/cjh/rsi/discovery_private_20261024/env/bin/python -B \
  -m rsi_researcher.research.discovery explore \
  --source-db runs/kuairand27k-science-frozen-v1-20261024/seq.duckdb \
  --development-db runs/kuairand27k-gpt-qwen-parity-v1-20261024/shared_explore.duckdb \
  --output "$OUTPUT" \
  --config configs/discovery_deepseek_20261024.json \
  --max-rounds 8 --min-queries 2
