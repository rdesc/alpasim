#!/usr/bin/env bash
# Alpamayo 1.5 closed-loop eval on nurec_curated_val (441 scenes, 1 rollout),
# conditioned on navigation text from the served route ("Turn left in 30m").
# Same settings as launch_val.sh (the 2026-10-20 run without navigation text);
# the only difference is the driver code: rod/e2e-train-subsets 9ec3682 derives
# nav_text from the route and passes it to the model.
#
# Usage:
#   ./launch_val_nav.sh smoke   # 3 scenes on one GPU (topology=1gpu; SMOKE_GPU=N, default 0)
#   ./launch_val_nav.sh full    # all 441 scenes, topology=8gpu_12rollouts
set -uo pipefail

MODE="${1:?usage: $0 smoke|full}"
REPO=/mnt/efs/users/rod/repos/alpasim-e2e
OUT=/mnt/efs/users/rod/repos/alpasim
# Scene cache (must hold the 441 val USDZs). Default: the garage's Alpasim cache
# on ip-10-225-150-13. On other hosts set CACHE, e.g. the NVMe cache that
# warm_cache.py fills: /opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts
CACHE="${CACHE:-/mnt/ebs/py123d_garage/lib/alpasim/alpasim/data/nre-artifacts}"
LOG="$OUT/launch_val_nav.log"

say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export UV_CACHE_DIR=/opt/dlami/nvme/rod/cache/uv
export HF_HOME=/opt/dlami/nvme/rod/cache/hf
export HF_TOKEN="$(cat /opt/dlami/nvme/rod/cache/hf/token)"

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

cd "$REPO" || { say "ABORT: no repo"; exit 1; }
say "=== $MODE started on $(hostname -s) (pid $$, cache $CACHE, $(git -c safe.directory='*' rev-parse --short HEAD)) ==="

case "$MODE" in
  smoke)
    RUN_DIR="$OUT/smoke_val_nav_$(hostname -s)_$(date +%m%d_%H%M)"
    G="${SMOKE_GPU:-0}"
    say "smoke (3 scenes, GPU $G): $RUN_DIR"
    uv run --no-sync alpasim_wizard "${COMMON[@]}" \
      topology=1gpu scenes.limit_to_first_n=3 \
      "services.renderer.gpus=[$G]" "services.driver.gpus=[$G]" \
      "services.physics.gpus=[$G]" "services.trafficsim.gpus=[$G]" \
      wizard.log_dir="$RUN_DIR" >> "$OUT/smoke_val_nav.log" 2>&1
    ;;
  full)
    RUN_DIR="$OUT/eval_alpamayo1_5_curated_val_nav_$(hostname -s)_$(date +%m%d_%H%M)"
    say "LAUNCHING: $RUN_DIR"
    uv run --no-sync alpasim_wizard "${COMMON[@]}" \
      topology=8gpu_12rollouts \
      wizard.log_dir="$RUN_DIR" >> "$OUT/val_nav_run.log" 2>&1
    ;;
  *) say "unknown mode $MODE"; exit 1 ;;
esac
say "wizard exited rc=$? ; results: $RUN_DIR/aggregate/"
