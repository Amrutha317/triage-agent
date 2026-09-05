#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh -- one-shot setup for a fresh RunPod (or any Linux+NVIDIA) pod.
#
#   1. creates an isolated .venv and installs every pinned dep the project needs
#   2. starts vLLM as an OpenAI-compatible server on :8000 (background)
#   3. waits until it answers, prints a test curl
#
# Usage (from the repo root, on the pod):
#   bash bootstrap.sh                 # install + serve base model
#   MODEL=meta-llama/Llama-3.1-8B-Instruct bash bootstrap.sh
#   SKIP_INSTALL=1 bash bootstrap.sh  # just (re)start the server
#   SERVE=0 bash bootstrap.sh         # install only, don't start vLLM
#   LORA_DIR=adapters/triage-lora bash bootstrap.sh   # serve base + LoRA adapter
#
# Versions below are frozen from a confirmed-working run (2026-09-05) --
# do NOT `pip install -U` anything in this stack ad hoc. If you need a newer
# package, update the pin here, re-verify end to end, and re-freeze.
#
# If you get "bad interpreter": run  sed -i 's/\r$//' bootstrap.sh  first
# (Windows line endings).
# =============================================================================
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
SERVE="${SERVE:-1}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
LORA_DIR="${LORA_DIR:-}"
LOG="${LOG:-vllm.log}"

echo "=============================================================="
echo " model      : $MODEL"
echo " port       : $PORT"
echo " workdir    : $(pwd)"
[ -n "$LORA_DIR" ] && echo " lora       : $LORA_DIR"
echo "=============================================================="

case "$(pwd)" in
  /workspace*) : ;;
  *) echo "WARNING: not under /workspace -- files here are lost when the pod stops." ;;
esac

command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || \
  echo "WARNING: nvidia-smi not found -- no GPU?"

# --- 0. clear any leftover GPU-resident process from a prior crashed run ----
# vLLM's engine runs as a separate multiprocessing.spawn subprocess. If a
# previous `vllm serve` died uncleanly, `pkill -f "vllm serve"` alone won't
# catch it (its cmdline is the spawn bootstrap, not "vllm serve"), and it
# will silently hold GPU memory, causing a CUDA OOM on the next start.
echo "--- clearing any stale vLLM / GPU processes ---"
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
sleep 2
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | \
    while IFS=, read -r pid mem; do
      pid="$(echo "$pid" | tr -d ' ')"
      [ -n "$pid" ] && { echo "  killing stale GPU process $pid ($mem)"; kill -9 "$pid" 2>/dev/null || true; }
    done
  sleep 1
fi

# --- 1. venv + pinned deps ---------------------------------------------------
# NOTE: everything installs into an isolated .venv, not the system/base
# Python. Do not `pip install -U vllm` (or anything else) outside this venv
# and expect it to stay compatible -- numpy/torch/outlines have narrow
# mutually-compatible version windows and an out-of-band upgrade WILL break
# this the same way it did before (numpy>=2.0 breaks outlines==0.0.46;
# torch!=2.4.0 breaks vllm==0.6.3 / xformers==0.0.27.post2).
if [ "$SKIP_INSTALL" != "1" ]; then
  echo "--- creating venv ---"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate

  echo "--- installing python deps (a few minutes) ---"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  pip install -q --upgrade pip

  # numpy pinned FIRST and explicitly -- outlines==0.0.46 requires numpy<2.0
  # and nothing later in this list will force it back down if something
  # upstream already pulled numpy>=2.0 into the environment.
  pip install -q "numpy==1.26.4"

  pip install -q \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "vllm==0.6.3" \
    "outlines==0.0.46" \
    "xformers==0.0.27.post2" \
    "openai==1.54.3" \
    "gradio==4.44.1" \
    "transformers==4.45.2" \
    "tokenizers==0.20.3" \
    "huggingface_hub==0.36.2" \
    "peft==0.13.2" \
    "trl==0.11.4" \
    "accelerate==1.0.1" \
    "bitsandbytes==0.44.1" \
    "datasets==3.0.1" \
    "pyyaml==6.0.2" \
    "pytest==8.3.3" \
    "huggingface_hub[hf_transfer]"

  # pyairports is NOT on PyPI under a compatible name -- it must come from
  # this git ref, or outlines' guided-decoding import path breaks with
  # "ModuleNotFoundError: No module named 'pyairports'" at server startup.
  pip install -q "pyairports @ git+https://github.com/ozeliger/pyairports.git"

  echo "--- verifying environment ---"
  python -c "
import numpy, torch, outlines, vllm
assert numpy.__version__.startswith('1.26'), f'numpy drifted: {numpy.__version__}'
assert torch.__version__.startswith('2.4.0'), f'torch drifted: {torch.__version__}'
print(f'numpy {numpy.__version__}  torch {torch.__version__}  outlines {outlines.__version__}  vllm {vllm.__version__}  OK')
"
  echo "--- deps installed and verified ---"
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Optional: gated models (Llama). Export HF_TOKEN before running, or run
#   huggingface-cli login
if [ -n "${HF_TOKEN:-}" ]; then
  python - <<'PY'
from huggingface_hub import login
import os
login(os.environ["HF_TOKEN"])
print("HF: logged in")
PY
fi

# --- 2. serve -------------------------------------------------------------
if [ "$SERVE" != "1" ]; then
  echo "SERVE=0 -- skipping vLLM start. Done."
  exit 0
fi

# kill any previous server on this port (belt-and-suspenders with step 0)
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
sleep 2

CMD=(vllm serve "$MODEL"
     --host 0.0.0.0 --port "$PORT"
     --served-model-name "$MODEL"
     --dtype auto
     --max-model-len "$MAX_LEN"
     --gpu-memory-utilization "$GPU_UTIL")

if [ -n "$LORA_DIR" ]; then
  CMD+=(--enable-lora --lora-modules "triage-lora=$LORA_DIR" --max-lora-rank 16)
fi

echo "--- starting vLLM (logs -> $LOG) ---"
echo "${CMD[*]}"
export HF_HUB_ENABLE_HF_TRANSFER=1
nohup "${CMD[@]}" > "$LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM pid $VLLM_PID"

# --- 3. wait for ready --------------------------------------------------
echo "--- waiting for the server (first run also downloads the model) ---"
for i in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo
    echo "=============================================================="
    echo " vLLM is up:  http://localhost:$PORT/v1"
    echo " model name:  $MODEL"
    [ -n "$LORA_DIR" ] && echo " lora name :  triage-lora"
    echo "=============================================================="
    echo "test it:"
    echo "  curl -s http://localhost:$PORT/v1/chat/completions \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}]}'"
    echo
    echo "point the code at:  BASE_URL = \"http://localhost:$PORT/v1\""
    echo "tail logs        :  tail -f $LOG"
    exit 0
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM process died. Last 40 log lines:"
    tail -n 40 "$LOG"
    exit 1
  fi
  sleep 5
done

echo "ERROR: server not ready after 15 min. Last 40 log lines:"
tail -n 40 "$LOG"
exit 1
