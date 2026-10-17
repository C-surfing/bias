#!/usr/bin/env bash
# All installs and caches belong to this account. Never uses sudo or conda init.
set -euo pipefail
umask 077
BASE=${RSI_BASE:-/data/cjh/rsi}
[[ $(realpath -m "$BASE") == /data/cjh/rsi ]] || { echo 'Use /data/cjh/rsi for this shared server.'; exit 1; }
[[ $(stat -c %U /data/cjh) == "$(id -un)" ]] || exit 1
mkdir -p "$BASE"/{tools,envs,cache/conda,cache/pip,cache/huggingface,tmp,logs,models,run}
export CONDA_PKGS_DIRS="$BASE/cache/conda" PIP_CACHE_DIR="$BASE/cache/pip"
export HF_HOME="$BASE/cache/huggingface" TMPDIR="$BASE/tmp"
export XDG_CACHE_HOME="$BASE/cache" HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
INSTALLER=Miniforge3-Linux-x86_64.sh
MINIFORGE_VERSION=26.7.2-0
ASSET="Miniforge3-$MINIFORGE_VERSION-Linux-x86_64.sh"
if [[ ! -x "$BASE/tools/miniforge/bin/conda" ]]; then
  if [[ ! -s "$BASE/tools/$INSTALLER" ]]; then
    curl -fsSL --connect-timeout 10 --max-time 600 --retry 2 "https://github.com/conda-forge/miniforge/releases/download/$MINIFORGE_VERSION/$ASSET" -o "$BASE/tools/$INSTALLER"
  fi
  curl -fsSL --connect-timeout 10 --max-time 60 --retry 2 "https://github.com/conda-forge/miniforge/releases/download/$MINIFORGE_VERSION/$ASSET.sha256" -o "$BASE/tools/$INSTALLER.sha256"
  EXPECTED=$(awk '{print $1}' "$BASE/tools/$INSTALLER.sha256")
  (cd "$BASE/tools" && printf '%s  %s\n' "$EXPECTED" "$INSTALLER" | sha256sum -c -)
  bash "$BASE/tools/$INSTALLER" -b -p "$BASE/tools/miniforge"
fi
CONDA="$BASE/tools/miniforge/bin/conda"
for name in inference research; do
  if [[ ! -x "$BASE/envs/$name/bin/python" ]]; then
    "$CONDA" create -y -p "$BASE/envs/$name" --override-channels -c conda-forge python=3.12 pip
  fi
done
"$BASE/envs/inference/bin/python" -m pip install 'vllm==0.31.0' modelscope
"$BASE/envs/research/bin/python" -m pip install -r "$BASE/project/requirements.txt"
"$BASE/envs/inference/bin/python" -m pip freeze > "$BASE/logs/inference-requirements.lock.txt"
"$BASE/envs/research/bin/python" -m pip freeze > "$BASE/logs/research-requirements.lock.txt"
"$BASE/envs/inference/bin/python" "$BASE/deployment/qwen3/download_model.py" --base "$BASE" --source "${RSI_MODEL_SOURCE:-huggingface}"
echo 'Installation completed. Recheck an idle GPU before starting serve.sh.'
