#!/usr/bin/env bash
# BC data collection: replay the logged human trajectory through AlpaSim over
# all 1761 nurec_curated_train scenes and record the rendered observations.
#
# Usage:
#   ./launch_bc_collect.sh                      8 GPUs, new dated run dir
#   NGPU=4 ./launch_bc_collect.sh               GPUs 0-3 only, new run dir
#   NGPU=4 RUN_DIR=<dir> ./launch_bc_collect.sh resume <dir> on GPUs 0-3
#
# NGPU selects topology=<NGPU>gpu_bc_collect and restricts the GPU guard to
# indices below NGPU, so a 4-GPU run leaves 4-7 for other work. Four GPUs do
# roughly half the scenes per hour: the 8-GPU run held every GPU at 90-96%,
# so render compute is the limit and the higher per-GPU concurrency in
# 4gpu_bc_collect recovers only the artifact-load stalls. Measured: 190
# scenes/h on 8, 111 scenes/h on 4.
#
# This is the probe8 configuration (2026-10-23_0444_bc_probe8: 8/8 scenes,
# rc=0, every scene scoring 1.0 with dist_to_gt_trajectory max 0.00 m) widened
# to the whole train split. Two things differ from the probe, both deliberate:
#
#   n_sim_steps=165   The BC target is a 4 s future, and the GT trajectory
#                     covers 20 s, so a frame rendered after t=16.0 s has no
#                     full future and its sample is discarded. The probe
#                     rendered 200 steps and threw the last 40 away.
#                     _log_metadata writes gt_ego_trajectory at rollout start
#                     from the clip, not from the poses actually simulated, so
#                     shortening the rollout does not shorten the target --
#                     this drops ~18% of render time for the same ~29
#                     samples/scene.
#   autoresume        13 h is long enough that a re-run is likely; a re-run
#                     with the same log_dir drops partial rollouts and skips
#                     completed ones.
#
# The driver is `static`: it answers immediately with a constant-velocity
# trajectory that force-GT discards. It exists only because rendered frames
# reach the ASL as driver_camera_image entries, i.e. only if a driver is
# queried. driver=manual cannot do this job -- it blocks until wall-clock time
# catches up with sim time, which deadlocked two earlier sweeps.
#
# The `static` plugin is registered in src/driver/pyproject.toml, but the
# container's venv metadata is baked at image build time while src/ is mounted
# live, so the entry point only exists in a rebuilt image -- hence the
# defines.base_image override below.
#
# Camera set, skip_driver_during_force_gt=false and the CPU-only driver service
# all come from driver=static / static_configs.yaml; do not re-override them
# here.
set -uo pipefail

OUT=/mnt/efs/users/rod/repos/alpasim
REPO=/mnt/efs/users/rod/repos/alpasim-e2e
CACHE=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts
IMAGE=alpasim-base:0.96.0-static
NGPU="${NGPU:-8}"
if [[ "$NGPU" != 4 && "$NGPU" != 8 ]]; then
  echo "NGPU must be 4 or 8 (no other <n>gpu_bc_collect topology exists)" >&2
  exit 1
fi
# A RUN_DIR that already holds rollouts is resumed, not restarted: autoresume
# drops partial rollouts and skips the ones carrying a _complete marker.
RUN_DIR="${RUN_DIR:-$OUT/alpamayo_eval_results/$(date +%Y-%m-%d)_bc_collect_train1761}"
LOG="$OUT/launch_bc_collect.log"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export HF_HOME=/opt/dlami/nvme/rod/cache/hf
export HF_TOKEN="$(cat /opt/dlami/nvme/rod/cache/hf/token)"

say "=== BC collection launcher started on $NGPU GPUs (pid $$) ==="

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  say "ABORT: image $IMAGE not built"; exit 1
fi

# --- guard: disk. A full root filesystem killed an earlier sweep. -----------
root_avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
nvme_avail=$(df --output=avail -BG /opt/dlami/nvme | tail -1 | tr -dc '0-9')
efs_avail=$(df --output=avail -BG /mnt/efs | tail -1 | tr -dc '0-9')
say "disk: / ${root_avail}G, nvme ${nvme_avail}G, efs ${efs_avail}G free"
if [[ "$root_avail" -lt 40 ]]; then say "ABORT: / below 40G free"; exit 1; fi
# The ASL output lands on EFS (~440 GB at 165 steps), not on nvme; nvme only
# holds the already-populated scene cache, so a small floor is enough here.
if [[ "$nvme_avail" -lt 120 ]]; then say "ABORT: nvme below 120G free"; exit 1; fi

# --- the scene artifacts must all be cached --------------------------------
cd "$REPO" || { say "ABORT: no repo"; exit 1; }
say "verifying curated_train artifacts are cached"
uv run --no-sync python "$OUT/warm_cache_train_all.py" >> "$OUT/warm_cache_train_all.log" 2>&1
if ! grep -q "sceneset ready" "$OUT/warm_cache_train_all.log" 2>/dev/null; then
  say "ABORT: scene download did not complete"; exit 1
fi
say "scene cache ready ($(ls "$CACHE/all-usdzs" | wc -l) artifacts)"

# --- wait for a free box ---------------------------------------------------
gpus_busy() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v n="$NGPU" '$1 < n && $2 > 3000' | wc -l
}
for i in $(seq 1 480); do
  busy=$(gpus_busy)
  [[ "$busy" -eq 0 ]] && { say "GPUs 0-$((NGPU - 1)) free"; break; }
  [[ $((i % 20)) -eq 0 ]] && say "waiting on GPUs 0-$((NGPU - 1)): $busy busy"
  sleep 30
done
if [[ "$(gpus_busy)" -ne 0 ]]; then
  say "ABORT: GPUs 0-$((NGPU - 1)) still busy after 4h"; exit 1
fi

source setup_local_env.sh >> "$LOG" 2>&1
mkdir -p "$RUN_DIR"
resume_from=$(find "$RUN_DIR/rollouts" -name _complete 2>/dev/null | wc -l)
say "LAUNCHING on GPUs 0-$((NGPU - 1)): $RUN_DIR ($resume_from/1761 already complete)"
start=$SECONDS

# --- progress reporter: a 13 h run needs to be observable while it runs ----
(
  while sleep 900; do
    n=$(find "$RUN_DIR/rollouts" -name _complete 2>/dev/null | wc -l)
    e=$((SECONDS - start))
    d=$((n - resume_from))
    [[ "$d" -le 0 ]] && { say "progress: $n/1761, none added yet after ${e}s"; continue; }
    say "progress: $(python3 -c "
e, n, d = $e, $n, $d
print(f'{n}/1761 (+{d} this run) in {e/3600:.1f}h, {d/e*3600:.0f} scenes/h, '
      f'ETA {(1761-n)*e/d/3600:.1f}h')")  asl $(find "$RUN_DIR/rollouts" -name '*.asl' -printf '%s\n' 2>/dev/null | awk '{t+=$1} END {printf "%.0f GB", t/1073741824}')"
  done
) &
reporter=$!
trap 'kill $reporter 2>/dev/null' EXIT

uv run --no-sync alpasim_wizard \
  deploy=local "topology=${NGPU}gpu_bc_collect" driver=static \
  +nurec_scenes=curated_train \
  runtime.simulation_config.n_sim_steps=165 \
  runtime.simulation_config.force_gt_duration_us=20000000 \
  runtime.simulation_config.n_rollouts=1 \
  runtime.enable_autoresume=true \
  eval.allow_aggregation_with_failed_rollouts=true \
  '+runtime.endpoints.startup_timeout_s=900' \
  "defines.base_image=$IMAGE" \
  defines.filesystem=/mnt/efs/users/rod/repos/alpasim/data \
  defines.hf_cache=/opt/dlami/nvme/rod/cache/hf \
  "defines.sensordata=$CACHE" \
  "scenes.scene_cache=$CACHE" \
  wizard.log_dir="$RUN_DIR" >> "$RUN_DIR/wizard_${NGPU}gpu_$(date +%m%d_%H%M).log" 2>&1
rc=$?
elapsed=$((SECONDS - start))
kill $reporter 2>/dev/null

done_n=$(find "$RUN_DIR/rollouts" -name _complete 2>/dev/null | wc -l)
say "wizard exited rc=$rc after $((elapsed / 3600))h$(( (elapsed % 3600) / 60 ))m; completed rollouts: $done_n/1761"
say "total ASL size: $(find "$RUN_DIR/rollouts" -name '*.asl' -printf '%s\n' 2>/dev/null \
  | awk '{t+=$1} END {printf "%.1f GB across %d files", t/1073741824, NR}')"
say "results: $RUN_DIR"
say "next: uv run python $OUT/extract_bc_dataset.py --run $RUN_DIR --out /mnt/efs/users/rod/bc_nurec_render_curated_train1761"
