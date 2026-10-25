#!/usr/bin/env bash
# Alpamayo 1.5 closed-loop eval on nurec_curated_val (441 scenes, 1 rollout),
# conditioned on navigation text from the served route.
#
# Differs from launch_val_nav.sh in two ways:
#
#  1. route_start_offset_m=0.0 instead of 40.0. At 40.0 the first ~42 m of
#     route are trimmed while route_to_nav_text only searches [5, 40] m, so
#     nav_text degenerates to 99.3% "Continue straight" (measured over all 441
#     scenes). The flag reaches only RouteGenerator._trim_route_start, and no
#     scorer reads the route, so this does not affect comparison against the
#     2026-10-20 baseline.
#
#  2. Four GPUs (4-7) instead of eight, because GPUs 0-3 are running a BC
#     collection job. 8gpu_12rollouts is reshaped to 6 concurrent rollouts:
#     renderer on 4-5, driver and physics on 6-7. Per-GPU load matches the
#     8-GPU run; only the rollout count halves.
#
# Usage: ./launch_val_nav_off0.sh
set -uo pipefail

REPO=/mnt/efs/users/rod/repos/alpasim-e2e
OUT=/mnt/efs/users/rod/repos/alpasim
CACHE="${CACHE:-/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts}"
LOG="$OUT/launch_val_nav_off0.log"

say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export UV_CACHE_DIR=/opt/dlami/nvme/rod/cache/uv
export HF_HOME=/opt/dlami/nvme/rod/cache/hf
export HF_TOKEN="$(cat /opt/dlami/nvme/rod/cache/hf/token)"

# Refuse to start if another job already holds one of our GPUs.
for g in 4 5 6 7; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  if [ "$used" -gt 2000 ]; then
    say "ABORT: GPU $g already has ${used} MiB in use"
    exit 1
  fi
done

cd "$REPO" || { say "ABORT: no repo"; exit 1; }
RUN_DIR="$OUT/eval_alpamayo1_5_curated_val_nav_off0_$(hostname -s)_$(date +%m%d_%H%M)"
say "=== started on $(hostname -s) (pid $$, cache $CACHE, $(git -c safe.directory='*' rev-parse --short HEAD)) ==="
say "LAUNCHING: $RUN_DIR"

uv run --no-sync alpasim_wizard \
  deploy=local driver=alpamayo1_5 \
  +nurec_scenes=curated_val \
  runtime.simulation_config.n_rollouts=1 \
  runtime.simulation_config.route_start_offset_m=0.0 \
  eval.allow_aggregation_with_failed_rollouts=true \
  eval.parse_unstructured_debug_info=false \
  +runtime.endpoints.startup_timeout_s=900 \
  defines.filesystem=/mnt/efs/users/rod/repos/alpasim/data \
  defines.hf_cache=/opt/dlami/nvme/rod/cache/hf \
  "defines.sensordata=$CACHE" \
  "scenes.scene_cache=$CACHE" \
  topology=8gpu_12rollouts \
  "services.renderer.gpus=[4,5]" \
  "services.driver.gpus=[6,7]" \
  "services.physics.gpus=[6,7]" \
  "services.trafficsim.gpus=[4]" \
  runtime.endpoints.renderer.n_concurrent_rollouts=3 \
  runtime.endpoints.driver.n_concurrent_rollouts=3 \
  runtime.endpoints.physics.n_concurrent_rollouts=3 \
  wizard.log_dir="$RUN_DIR" >> "$OUT/val_nav_off0_run.log" 2>&1

say "wizard exited rc=$? ; results: $RUN_DIR/aggregate/"
