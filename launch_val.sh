#!/usr/bin/env bash
# Self-contained overnight launcher for Alpamayo 1.5 closed-loop eval on
# nurec_curated_val (441 scenes, 1 rollout). Detached: survives chat restarts.
#
# Waits for: scene cache complete -> a passing smoke test -> all 8 GPUs free.
set -uo pipefail

REPO=/mnt/efs/users/rod/repos/alpasim-e2e
OUT=/mnt/efs/users/rod/repos/alpasim
CACHE=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts
LOG="$OUT/launch_val.log"

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export HF_HOME=/opt/dlami/nvme/rod/cache/hf
export HF_TOKEN="$(cat /opt/dlami/nvme/rod/cache/hf/token)"

# NOTE: no +cameras=6cam_1080. The in-repo A1.5 driver raises
# "Camera camera_rear_right_70fov not in desired cameras" for any delivered
# camera outside inference.use_cameras, so it cannot take the 6-cam set.
# driver=alpamayo1_5 brings its native 4-camera config, which is exactly the
# set the model consumes.
COMMON=(
  deploy=local driver=alpamayo1_5
  +nurec_scenes=curated_val
  runtime.simulation_config.n_rollouts=1
  runtime.simulation_config.route_start_offset_m=40.0
  eval.allow_aggregation_with_failed_rollouts=true
  eval.parse_unstructured_debug_info=false
  +runtime.endpoints.startup_timeout_s=900
  defines.filesystem=/mnt/efs/users/rod/repos/alpasim/data
  defines.hf_cache=/opt/dlami/nvme/rod/cache/hf
  "defines.sensordata=$CACHE"
  "scenes.scene_cache=$CACHE"
)

say "=== launcher started (pid $$) ==="
cd "$REPO" || { say "ABORT: no repo"; exit 1; }
source setup_local_env.sh >> "$LOG" 2>&1

# --- 1. scene cache: resume download until all 441 artifacts are present ---
# Another warm_cache may already be running (started from the chat session);
# wait it out rather than racing a second downloader against it.
for i in $(seq 1 240); do
  pgrep -f warm_cache.py >/dev/null || break
  [[ $((i % 20)) -eq 0 ]] && say "existing warm_cache still running ($(ls "$CACHE/all-usdzs" 2>/dev/null | wc -l)/441)"
  sleep 30
done

for attempt in 1 2 3; do
  n=$(ls "$CACHE/all-usdzs" 2>/dev/null | wc -l)
  [[ "$n" -ge 441 ]] && break
  say "scene cache $n/441 -- running warm_cache (attempt $attempt)"
  uv run --no-sync python "$OUT/warm_cache.py" >> "$OUT/warm_cache.log" 2>&1
done
n=$(ls "$CACHE/all-usdzs" 2>/dev/null | wc -l)
if [[ "$n" -lt 441 ]]; then say "ABORT: scene cache incomplete ($n/441)"; exit 1; fi
say "scene cache complete ($n artifacts)"

# --- 2. smoke test: reuse a passing one, otherwise run our own -------------
# A smoke may already be running (started from the chat session); wait for it
# instead of racing a second wizard onto the same GPU.
for i in $(seq 1 120); do
  pgrep -f alpasim_wizard >/dev/null || break
  [[ $((i % 10)) -eq 0 ]] && say "a wizard is already running; waiting before smoke check"
  sleep 30
done

passing_smoke() { ls -d "$OUT"/smoke_val_*/aggregate/metrics_results.txt 2>/dev/null | head -1; }
if [[ -n "$(passing_smoke)" ]]; then
  say "smoke already PASSED: $(passing_smoke)"
else
  SMOKE="$OUT/smoke_val_$(hostname -s)_$(date +%m%d_%H%M)"
  say "running smoke (3 scenes, 1 GPU): $SMOKE"
  uv run --no-sync alpasim_wizard "${COMMON[@]}" \
    topology=1gpu scenes.limit_to_first_n=3 \
    wizard.log_dir="$SMOKE" >> "$OUT/smoke_val.log" 2>&1
  if [[ ! -f "$SMOKE/aggregate/metrics_results.txt" ]]; then
    say "ABORT: smoke produced no aggregate -- see $OUT/smoke_val.log"; exit 1
  fi
  say "smoke PASSED: $SMOKE"
fi

# --- 3. wait for all 8 GPUs to be free ------------------------------------
for i in $(seq 1 720); do   # up to 6h
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000' | wc -l)
  [[ "$busy" -eq 0 ]] && { say "all 8 GPUs free"; break; }
  [[ $((i % 20)) -eq 0 ]] && say "waiting on GPUs: $busy busy"
  sleep 30
done
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000' | wc -l)
if [[ "$busy" -ne 0 ]]; then say "ABORT: GPUs still busy after 6h"; exit 1; fi

# --- 4. launch the full val run -------------------------------------------
RUN_DIR="$OUT/eval_alpamayo1_5_curated_val_$(hostname -s)_$(date +%m%d_%H%M)"
say "LAUNCHING: $RUN_DIR"
uv run --no-sync alpasim_wizard "${COMMON[@]}" \
  topology=8gpu_12rollouts \
  wizard.log_dir="$RUN_DIR" >> "$OUT/val_run.log" 2>&1
say "wizard exited rc=$? ; results: $RUN_DIR/aggregate/"
