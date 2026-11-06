#!/usr/bin/env bash
# Alpamayo 1.5 closed-loop eval on AgenticDriver's Alpasim dev split (alpasim-curated-dev100: 100 random
# nurec_curated_train scenes, seed 0), 1 rollout per scene. Purpose: is dev100 representative of curated_val?
# Compare with the 2026-10-20 curated_val run (scene score 56.1).
#
# Settings are exactly those of launch_val.sh (the 2026-10-20 curated_val run); only the scene suite differs
# (+nurec_scenes=dev100, added 2026-11-06). Code: alpasim-e2e rod/e2e-main 3414640; the commits since the val
# run's 9ec3682 do not change the Alpamayo driver's behaviour (model revisions pinned to the installed ones,
# rectification moved off the event loop, nav_text logging). Defaults confirmed unchanged:
# force_gt_duration_us 1.7 s, control_timestep_us 0.1 s; route_start_offset_m 40 as in the val run.
#
# Autoresume is on: re-running with the same RUN_DIR skips completed rollouts.
#   ./launch_alpamayo_dev100.sh                  new dated run dir
#   RUN_DIR=<dir> ./launch_alpamayo_dev100.sh    resume <dir>
set -uo pipefail

N=100
OUT=/mnt/efs/users/rod/repos/alpasim
REPO=/mnt/efs/users/rod/repos/alpasim-e2e
CACHE=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts
RUN_DIR="${RUN_DIR:-$OUT/alpamayo_eval_results/$(date +%Y-%m-%d)_dev100}"
LOG="$OUT/launch_dev100.log"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export HF_HOME=/opt/dlami/nvme/rod/cache/hf
export HF_TOKEN="$(cat /opt/dlami/nvme/rod/cache/hf/token)"

say "=== dev100 launcher started (pid $$) on $(hostname -s) ==="
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000' | wc -l)
if [[ "$busy" -ne 0 ]]; then say "ABORT: $busy GPUs busy"; exit 1; fi
root_avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
if [[ "$root_avail" -lt 40 ]]; then say "ABORT: / below 40G free (${root_avail}G)"; exit 1; fi

cd "$REPO" || { say "ABORT: no repo"; exit 1; }
source setup_local_env.sh >> "$LOG" 2>&1
mkdir -p "$RUN_DIR"
resume_from=$(find "$RUN_DIR/rollouts" -name _complete 2>/dev/null | wc -l)
say "LAUNCHING: $RUN_DIR ($resume_from/$N already complete)"
start=$SECONDS

(
  while sleep 900; do
    n=$(find "$RUN_DIR/rollouts" -name _complete 2>/dev/null | wc -l)
    say "progress: $n/$N after $(((SECONDS - start) / 60)) min"
  done
) &
reporter=$!
trap 'kill $reporter 2>/dev/null' EXIT

uv run --no-sync alpasim_wizard \
  deploy=local topology=8gpu_12rollouts driver=alpamayo1_5 \
  +nurec_scenes=dev100 \
  runtime.simulation_config.n_rollouts=1 \
  runtime.simulation_config.route_start_offset_m=40.0 \
  runtime.enable_autoresume=true \
  eval.allow_aggregation_with_failed_rollouts=true \
  eval.parse_unstructured_debug_info=false \
  '+runtime.endpoints.startup_timeout_s=900' \
  defines.filesystem=/mnt/efs/users/rod/repos/alpasim/data \
  defines.hf_cache=/opt/dlami/nvme/rod/cache/hf \
  "defines.sensordata=$CACHE" \
  "scenes.scene_cache=$CACHE" \
  wizard.log_dir="$RUN_DIR" >> "$RUN_DIR/wizard.log" 2>&1
rc=$?
kill $reporter 2>/dev/null
say "wizard exited rc=$rc after $(((SECONDS - start) / 60)) min"
say "completed rollouts: $(find "$RUN_DIR/rollouts" -name _complete 2>/dev/null | wc -l)/$N"
say "results: $RUN_DIR/aggregate/"
