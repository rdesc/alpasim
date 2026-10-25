#!/usr/bin/env bash
# Detached launcher for Alpamayo 1.5 on nurec_train_sample400 (400 scenes,
# 1 rollout). Waits for the scene download, then for free GPUs, then runs.
#
# Config is identical to the completed curated_val run except for the suite and
# parse_unstructured_debug_info=true (the val run had it off; see
# NOTES_val_eval.md), so the two runs are directly comparable.
set -uo pipefail

REPO=/mnt/efs/users/rod/repos/alpasim-e2e
OUT=/mnt/efs/users/rod/repos/alpasim
CACHE=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts
LOG="$OUT/launch_train.log"

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export HF_HOME=/opt/dlami/nvme/rod/cache/hf
export HF_TOKEN="$(cat /opt/dlami/nvme/rod/cache/hf/token)"

say "=== train launcher started (pid $$) ==="
cd "$REPO" || { say "ABORT: no repo"; exit 1; }

# --- 1. wait for the train scene download ---------------------------------
# NOTE: do not gate on `pgrep -f warm_cache_*.py`. pgrep -f matches full command
# lines, so any shell whose command line mentions the script (including the one
# that launched it) is a false positive and the gate never opens. Gate on the
# completion sentinel the download writes instead.
for i in $(seq 1 240); do   # up to 2h
  grep -q "sceneset ready" "$OUT/warm_cache_mix200.log" 2>/dev/null && break
  [[ $((i % 20)) -eq 0 ]] && say "waiting on download ($(ls "$CACHE/all-usdzs" | wc -l) artifacts cached)"
  sleep 30
done
if ! grep -q "sceneset ready" "$OUT/warm_cache_mix200.log" 2>/dev/null; then
  say "ABORT: train scene download did not complete -- see warm_cache_mix200.log"
  exit 1
fi
say "train scene cache ready ($(ls "$CACHE/all-usdzs" | wc -l) artifacts total)"

# --- 2. wait for a free box ------------------------------------------------
for i in $(seq 1 720); do   # up to 6h
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000' | wc -l)
  [[ "$busy" -eq 0 ]] && { say "all 8 GPUs free"; break; }
  [[ $((i % 20)) -eq 0 ]] && say "waiting on GPUs: $busy busy"
  sleep 30
done
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000' | wc -l)
if [[ "$busy" -ne 0 ]]; then say "ABORT: GPUs still busy after 6h"; exit 1; fi

# --- 3. launch -------------------------------------------------------------
source setup_local_env.sh >> "$LOG" 2>&1
RESULTS="$OUT/alpamayo_eval_results"
mkdir -p "$RESULTS"
RUN_DIR="$RESULTS/$(date +%Y-%m-%d)_train_mix200"
mkdir -p "$RUN_DIR"   # the wizard log is redirected into it below
say "LAUNCHING: $RUN_DIR"

uv run --no-sync alpasim_wizard \
  deploy=local topology=8gpu_12rollouts driver=alpamayo1_5 \
  +nurec_scenes=train_mix200 \
  runtime.simulation_config.n_rollouts=1 \
  runtime.simulation_config.route_start_offset_m=40.0 \
  eval.allow_aggregation_with_failed_rollouts=true \
  eval.parse_unstructured_debug_info=true \
  '+runtime.endpoints.startup_timeout_s=900' \
  defines.filesystem=/mnt/efs/users/rod/repos/alpasim/data \
  defines.hf_cache=/opt/dlami/nvme/rod/cache/hf \
  "defines.sensordata=$CACHE" \
  "scenes.scene_cache=$CACHE" \
  wizard.log_dir="$RUN_DIR" >> "$RUN_DIR/wizard.log" 2>&1
rc=$?
say "wizard exited rc=$rc ; results: $RUN_DIR/aggregate/"
