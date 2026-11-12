#!/usr/bin/env bash
# Alpamayo 2 Super closed-loop eval on AgenticDriver's Alpasim scene sets, 1 rollout per scene, to compare
# with the Claude teacher and Alpamayo 1.5 on the same scenes and the same scorer (Alpasim's scene score).
#
# Code: alpasim-a2 rod/alpamayo2 = rod/e2e-main (the Alpamayo 1.5 dev100 run's simulator, controller and eval)
# + upstream NVlabs/alpasim main's driver package (affc2ea), driver=alpamayo2 as upstream ships it: the
# 4 cameras of Alpamayo 1.5 at 10 Hz, 4 context frames, reasoning (CoC) on, no route / navigation input,
# 1 trajectory sample, no trajectory optimiser. Scene settings as launch_alpamayo_dev100.sh
# (route_start_offset_m 40, defaults force_gt_duration_us 1.7 s, control_timestep_us 0.1 s).
#
#   [DRIVER=alpamayo2_nav] SUITE=dev100|val100 ./launch_alpamayo2.sh   new dated run dir
#   SUITE=... RUN_DIR=<dir> ./launch_alpamayo2.sh        resume <dir> (autoresume skips completed rollouts)
#   SUITE=... SMOKE=<n> ./launch_alpamayo2.sh            only the first n scenes of the suite
set -uo pipefail

SUITE="${SUITE:?set SUITE=dev100 or val100}"
DRIVER="${DRIVER:-alpamayo2}"   # alpamayo2 (upstream, no navigation) or alpamayo2_nav
OUT=/mnt/efs/users/rod/repos/alpasim
REPO=/mnt/efs/users/rod/repos/alpasim-a2
CACHE=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts
TAG="${SMOKE:+smoke${SMOKE}_}"
RUN_DIR="${RUN_DIR:-$OUT/alpamayo_eval_results/$(date +%Y-%m-%d)_${DRIVER/alpamayo2/a2super}_${TAG}${SUITE}}"
LOG="$OUT/launch_a2super.log"
say() { echo "$(date '+%F %T') [$SUITE] $*" | tee -a "$LOG"; }

export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim-a2
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-a2
export HF_HOME=/opt/dlami/nvme/rod/cache/hf
export HF_TOKEN="$(cat /opt/dlami/nvme/rod/cache/hf/token)"

say "=== Alpamayo 2 Super launcher started (pid $$) on $(hostname -s) ==="
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1>=4 && $2>3000' | wc -l)
if [[ "$busy" -ne 0 ]]; then say "ABORT: $busy of the driver GPUs 4-7 busy"; exit 1; fi
root_avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
if [[ "$root_avail" -lt 40 ]]; then say "ABORT: / below 40G free (${root_avail}G)"; exit 1; fi

cd "$REPO" || { say "ABORT: no repo"; exit 1; }
SCENES=()
N=$(($(wc -l < "data/scenes/sim_suites_${SUITE}.csv") - 1))
if [[ -n "${SMOKE:-}" ]]; then
  ids=$(tail -n +2 "data/scenes/sim_suites_${SUITE}.csv" | head -n "$SMOKE" | cut -d, -f2 | paste -sd, -)
  SCENES=("scenes.scene_ids=[$ids]" "scenes.test_suite_id=null")
  N=$SMOKE
fi
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
  deploy=local topology=8gpu_a2super "driver=$DRIVER" \
  "+nurec_scenes=$SUITE" "${SCENES[@]}" \
  runtime.simulation_config.n_rollouts=1 \
  runtime.simulation_config.route_start_offset_m=40.0 \
  runtime.enable_autoresume=true \
  eval.allow_aggregation_with_failed_rollouts=true \
  eval.parse_unstructured_debug_info=false \
  '+runtime.endpoints.startup_timeout_s=1800' \
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
