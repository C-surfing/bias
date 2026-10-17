#!/usr/bin/env bash
set -euo pipefail
umask 077
BASE=${RSI_BASE:-/data/cjh/rsi}
: "${RSI_GPU_UUID:?Set RSI_GPU_UUID to one currently idle GPU UUID.}"
[[ "$RSI_GPU_UUID" == GPU-* && "$RSI_GPU_UUID" != *,* ]] || exit 1
read -r USED UTIL < <(nvidia-smi -i "$RSI_GPU_UUID" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
(( USED < 512 && UTIL == 0 )) || { echo 'Selected GPU is occupied. Do not stop other users; select an idle one.'; exit 1; }
export CUDA_VISIBLE_DEVICES="$RSI_GPU_UUID"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PATH="$BASE/envs/inference/bin:$PATH"
export HF_HOME="$BASE/cache/huggingface" XDG_CACHE_HOME="$BASE/cache" TMPDIR="$BASE/tmp"
export CUDA_CACHE_PATH="$BASE/cache/cuda" VLLM_CACHE_ROOT="$BASE/cache/vllm"
export VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4
exec "$BASE/envs/inference/bin/vllm" serve "$BASE/models/Qwen3-8B" \
  --served-model-name qwen3-8b --host 127.0.0.1 --port "${RSI_MODEL_PORT:-8000}" \
  --dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.75 \
  --max-model-len 32768 --max-num-seqs 2 --generation-config vllm
