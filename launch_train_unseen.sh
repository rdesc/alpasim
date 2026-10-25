#!/usr/bin/env bash
# Alpamayo 1.5 closed-loop eval on every train scene not yet evaluated:
# 1179 of the 1761 nurec_curated_train scenes, conditioned on navigation text
# from the served route.
#
# route_start_offset_m is 0.0, not the 40.0 the earlier train runs used. At
# 40.0 the first ~42 m of route are trimmed while route_to_nav_text only
# searches [5, 40] m, so nav_text degenerates to "Continue straight" on 99.3%
# of scenes (measured over all 441 val scenes) and the run would not test nav
# conditioning at all. The flag reaches only RouteGenerator._trim_route_start
# and no scorer reads the route, so the metric definition is unchanged against
# the mix200/rand200b/val baselines -- but the policy is conditioned on
# different text, so scores are expected to move. That difference is the
# measurement, not an artifact.
#
# parse_unstructured_debug_info is on so each rollout records the nav_text the
# model actually saw; without it there is no way to tell afterwards what it was
# conditioned on.
#
# All 1179 artifacts are already cached on NVMe, so there is no download step.
# Autoresume is on: at ~35.6 scenes/h this is ~33 h, so expect to stop and
# resume it. A re-run with the same log_dir drops partial rollouts and skips
# completed ones.
set -uo pipefail

OUT=/mnt/efs/users/rod/repos/alpasim
REPO=/mnt/efs/users/rod/repos/alpasim-e2e
CACHE=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts
RUN_DIR="${RUN_DIR:-$OUT/alpamayo_eval_results/$(date +%Y-%m-%d)_train_unseen_nav}"
LOG="$OUT/launch_train_unseen.log"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export HF_HOME=/opt/dlami/nvme/rod/cache/hf
export HF_TOKEN="$(cat /opt/dlami/nvme/rod/cache/hf/token)"

say "=== train_unseen nav-text launcher started (pid $$) ==="

root_avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
say "disk: / has ${root_avail}G free"
if [[ "$root_avail" -lt 40 ]]; then say "ABORT: / below 40G free"; exit 1; fi

for i in $(seq 1 720); do   # up to 6h
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000' | wc -l)
  [[ "$busy" -eq 0 ]] && { say "all 8 GPUs free"; break; }
  [[ $((i % 20)) -eq 0 ]] && say "waiting on GPUs: $busy busy"
  sleep 30
done
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000' | wc -l)
if [[ "$busy" -ne 0 ]]; then say "ABORT: GPUs still busy after 6h"; exit 1; fi

cd "$REPO" || { say "ABORT: no repo"; exit 1; }
source setup_local_env.sh >> "$LOG" 2>&1
mkdir -p "$RUN_DIR"
resume_from=$(find "$RUN_DIR/rollouts" -name _complete 2>/dev/null | wc -l)
say "LAUNCHING: $RUN_DIR ($resume_from/1179 already complete)"
start=$SECONDS

(
  while sleep 1800; do
    n=$(find "$RUN_DIR/rollouts" -name _complete 2>/dev/null | wc -l)
    e=$((SECONDS - start)); d=$((n - resume_from))
    [[ "$d" -le 0 ]] && { say "progress: $n/1179, none added yet after ${e}s"; continue; }
    say "progress: $(python3 -c "
e, n, d = $e, $n, $d
print(f'{n}/1179 (+{d} this run) in {e/3600:.1f}h, {d/e*3600:.1f} scenes/h, '
      f'ETA {(1179-n)*e/d/3600:.1f}h')")"
  done
) &
reporter=$!
trap 'kill $reporter 2>/dev/null' EXIT

uv run --no-sync alpasim_wizard \
  deploy=local topology=8gpu_12rollouts driver=alpamayo1_5 \
  +nurec_scenes=train_unseen \
  runtime.simulation_config.n_rollouts=1 \
  runtime.simulation_config.route_start_offset_m=0.0 \
  runtime.enable_autoresume=true \
  eval.allow_aggregation_with_failed_rollouts=true \
  eval.parse_unstructured_debug_info=true \
  '+runtime.endpoints.startup_timeout_s=900' \
  defines.filesystem=/mnt/efs/users/rod/repos/alpasim/data \
  defines.hf_cache=/opt/dlami/nvme/rod/cache/hf \
  "defines.sensordata=$CACHE" \
  "scenes.scene_cache=$CACHE" \
  wizard.log_dir="$RUN_DIR" >> "$RUN_DIR/wizard.log" 2>&1
rc=$?
kill $reporter 2>/dev/null
say "wizard exited rc=$rc after $(((SECONDS - start) / 3600))h"
say "completed rollouts: $(find "$RUN_DIR/rollouts" -name _complete 2>/dev/null | wc -l)/1179"
say "results: $RUN_DIR/aggregate/"
