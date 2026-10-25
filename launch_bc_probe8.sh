#!/usr/bin/env bash
# 8-scene throughput probe for BC data collection.
#
# Runs exactly what the 1761-scene sweep would run, on 8 already-cached scenes,
# and reports wall-clock scenes/hour plus what actually landed in each ASL.
# Every previous estimate for this sweep (50 h, 10 h, 3 h, 5 h, 4 h) was an
# extrapolation and all of them were wrong, so measure first.
#
# The driver is `static`: it answers immediately with a constant-velocity
# trajectory that force-GT discards. It exists only because rendered frames
# reach the ASL as driver_camera_image entries, i.e. only if a driver is
# queried. driver=manual cannot do this job -- it blocks until wall-clock time
# catches up with sim time, which is what deadlocked the earlier sweeps.
#
# The `static` plugin is registered in src/driver/pyproject.toml, but the
# container's venv metadata is baked at image build time while src/ is mounted
# live, so the entry point only exists in a rebuilt image -- hence the
# defines.base_image override below.
set -uo pipefail

OUT=/mnt/efs/users/rod/repos/alpasim
REPO=/mnt/efs/users/rod/repos/alpasim-e2e
CACHE=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts
IMAGE=alpasim-base:0.96.0-static
RUN_DIR="$OUT/alpamayo_eval_results/$(date +%Y-%m-%d_%H%M)_bc_probe8"
LOG="$OUT/launch_bc_probe8.log"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export HF_HOME=/opt/dlami/nvme/rod/cache/hf
export HF_TOKEN="$(cat /opt/dlami/nvme/rod/cache/hf/token)"

say "=== bc probe8 started (pid $$) ==="

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  say "ABORT: image $IMAGE not built"; exit 1
fi

root_avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
say "disk: / has ${root_avail}G free"
if [[ "$root_avail" -lt 40 ]]; then say "ABORT: / below 40G free"; exit 1; fi

busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>3000' | wc -l)
if [[ "$busy" -ne 0 ]]; then say "ABORT: $busy GPUs busy"; exit 1; fi

cd "$REPO" || { say "ABORT: no repo"; exit 1; }
source setup_local_env.sh >> "$LOG" 2>&1
mkdir -p "$RUN_DIR"
say "LAUNCHING: $RUN_DIR"
start=$SECONDS

uv run --no-sync alpasim_wizard \
  deploy=local topology=8gpu_bc_collect driver=static \
  +nurec_scenes=bc_probe8 \
  runtime.simulation_config.n_sim_steps=200 \
  runtime.simulation_config.force_gt_duration_us=20000000 \
  runtime.simulation_config.n_rollouts=1 \
  eval.allow_aggregation_with_failed_rollouts=true \
  '+runtime.endpoints.startup_timeout_s=900' \
  "defines.base_image=$IMAGE" \
  defines.filesystem=/mnt/efs/users/rod/repos/alpasim/data \
  defines.hf_cache=/opt/dlami/nvme/rod/cache/hf \
  "defines.sensordata=$CACHE" \
  "scenes.scene_cache=$CACHE" \
  wizard.log_dir="$RUN_DIR" >> "$RUN_DIR/wizard.log" 2>&1
rc=$?
elapsed=$((SECONDS - start))

done_n=$(find "$RUN_DIR/rollouts" -name _complete 2>/dev/null | wc -l)
say "wizard exited rc=$rc after ${elapsed}s; completed rollouts: $done_n"
if [[ "$done_n" -gt 0 ]]; then
  # The probe's 8 scenes occupy only 8 of the topology's 24 rollout slots, so
  # the amortized per-scene rate measured here is ~3x the rate the full sweep
  # sees. Project by scaling to the configured concurrency, not by multiplying.
  say "THROUGHPUT: $(python3 -c "
e, n, slots = $elapsed, $done_n, 24
print(f'{e/n:.0f}s/scene amortized at {n}-way, {n/e*3600:.1f} scenes/h; '
      f'scaled to {slots}-way: {e/n*1761/3600*n/slots:.1f} h for all 1761')")"
fi

# What actually landed in the logs: the BC inputs (camera frames) and the BC
# target (the recorded trajectory in rollout_metadata).
say "--- ASL contents ---"
find "$RUN_DIR/rollouts" -name "*.asl" 2>/dev/null | head -3 | while read -r asl; do
  say "$(du -h "$asl" | cut -f1)  $asl"
done
say "total ASL size: $(find "$RUN_DIR/rollouts" -name '*.asl' -printf '%s\n' 2>/dev/null \
  | awk '{t+=$1} END {printf "%.1f GB across %d files", t/1073741824, NR}')"
say "results: $RUN_DIR"
