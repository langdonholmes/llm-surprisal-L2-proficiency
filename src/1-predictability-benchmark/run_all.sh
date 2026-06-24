#!/usr/bin/env bash
#
# run_all.sh — run the Study 1 surprisal matrix across every available GPU.
#
# The work is embarrassingly parallel across models (each loads independently
# and writes its own output dir, and every model fits on one card), so this
# cost-balances the model list across the detected GPUs and launches one
# process per GPU, each pinned via CUDA_VISIBLE_DEVICES. The pipeline is
# idempotent, so re-running resumes wherever it left off.
#
# Usage:
#   bash src/1-predictability-benchmark/run_all.sh [extra run_surprisal.py args]
#
# Examples:
#   bash src/1-predictability-benchmark/run_all.sh
#   bash src/1-predictability-benchmark/run_all.sh --windows 64 --corpora ellipse
#   MODELS="qwen2.5-7b olmo2-7b llama3.1-8b" bash src/1-predictability-benchmark/run_all.sh
#   GPUS="0,1" bash src/1-predictability-benchmark/run_all.sh        # force GPU set
#
# Env overrides:
#   MODELS  space-separated model keys to run (default: the whole registry)
#   GPUS    comma-separated GPU indices    (default: all from nvidia-smi)
#
# For a long detached run, wrap it: nohup bash .../run_all.sh & disown
#
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
SCRIPT=src/1-predictability-benchmark/run_surprisal.py
LOGDIR=logs
mkdir -p "$LOGDIR"

# --- detect GPUs ----------------------------------------------------------
if [[ -n "${GPUS:-}" ]]; then
  IFS=',' read -ra GPU_IDS <<< "$GPUS"
else
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
NGPU=${#GPU_IDS[@]}
[[ "$NGPU" -eq 0 ]] && { echo "No GPUs found (set GPUS=... to override)."; exit 1; }
echo "Using $NGPU GPU(s): ${GPU_IDS[*]}"

# --- cost-balanced model -> bin assignment (one bin per GPU) --------------
# Model list comes from the registry (so newly-added models are picked up);
# weights below are rough relative GPU-cost, with a default for unlisted keys.
mapfile -t ASSIGN < <(MODELS="${MODELS:-}" "$PY" - "$NGPU" <<'PYEOF'
import os, sys
sys.path.insert(0, "src/1-predictability-benchmark")
sys.path.insert(0, "src")
from models import MODEL_REGISTRY

n = int(sys.argv[1])
weights = {
    "llama3.1-8b": 8.0, "llama3.1-8b-instruct": 8.0,
    "qwen2.5-7b": 7.0, "qwen2.5-7b-instruct": 7.0, "olmo2-7b": 7.0,
    "gpt2-xl": 2.0, "olmo2-1b": 1.5, "modernbert-large": 1.0,
    "modernbert-base": 0.6, "gpt2": 0.5, "bert-base": 0.5,
}
DEFAULT_WEIGHT = 1.0

requested = os.environ.get("MODELS", "").split()
models = requested or list(MODEL_REGISTRY)
unknown = [m for m in models if m not in MODEL_REGISTRY]
if unknown:
    sys.exit(f"unknown model(s): {unknown}")

# Longest-processing-time greedy: heaviest first, assign to least-loaded bin.
order = sorted(models, key=lambda m: weights.get(m, DEFAULT_WEIGHT), reverse=True)
load = [0.0] * n
bins = [[] for _ in range(n)]
for m in order:
    i = min(range(n), key=lambda k: load[k])
    bins[i].append(m)
    load[i] += weights.get(m, DEFAULT_WEIGHT)
for b in bins:
    print(" ".join(b))
PYEOF
)

# --- launch one process per GPU -------------------------------------------
pids=()
for i in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$i]}"
  models="${ASSIGN[$i]:-}"
  [[ -z "$models" ]] && continue   # more GPUs than models
  log="$LOGDIR/surprisal_gpu${gpu}.log"
  echo "GPU $gpu <- $models"
  echo "          log: $log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" --models $models "$@" \
      > "$log" 2>&1 &
  pids+=("$!")
done

echo
echo "Launched ${#pids[@]} process(es). Monitor with:"
echo "    tail -f $LOGDIR/surprisal_gpu*.log"
echo

# --- wait, surface failures -----------------------------------------------
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
if [[ "$fail" -ne 0 ]]; then
  echo "FAILED: a GPU process exited non-zero — check $LOGDIR/surprisal_gpu*.log"
  exit 1
fi
echo "All GPU processes finished."
