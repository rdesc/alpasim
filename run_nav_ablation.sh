#!/usr/bin/env bash
# Nav-text conditioning ablation for Alpamayo 1.5.
#
# Runs the same 3 scenes twice -- once nav-unconditioned (current behavior) and
# once with route-derived instructions ("Turn left in 30m" / "Continue straight")
# -- and prints a per-scene score comparison.
#
# Requires the rod-dev branch (nav_text plumbing + DEFAULT_QUAD video layout).
#
# Usage:
#   ./run_nav_ablation.sh                 # 5 rollouts/scene/condition (30 total)
#   ROLLOUTS=1 ./run_nav_ablation.sh      # quick smoke, 6 rollouts total
#   ONLY=on ./run_nav_ablation.sh         # only the nav-conditioned arm
#   GPUS=1,5,6 ./run_nav_ablation.sh      # pin to specific GPUs
#   PER_GPU=3 ./run_nav_ablation.sh       # concurrent Alpamayo rollouts per GPU
set -eo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO}"

ROLLOUTS="${ROLLOUTS:-5}"   # rollouts per scene per condition; >1 needed to see past sampling noise
ONLY="${ONLY:-both}"        # both | off | on

# --- GPUs -----------------------------------------------------------------
# Runs on however many GPUs are free -- all services share the given set, like
# the 1gpu topology but fanned out. Default: auto-detect idle GPUs (<1 GiB used).
if [[ -z "${GPUS:-}" ]]; then
  GPUS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          | awk -F', ' '$2 < 1000 {print $1}' | paste -sd, -)"
fi
if [[ -z "${GPUS}" ]]; then
  echo "❌ No idle GPUs found. Set GPUS=<comma list>, e.g. GPUS=1,5,6" >&2
  exit 1
fi
PER_GPU="${PER_GPU:-2}"                              # concurrent Alpamayo 10B rollouts per GPU (~20 GB each)
NGPU="$(tr ',' '\n' <<<"${GPUS}" | grep -c .)"
TOTAL=$(( PER_GPU * NGPU ))
GPU_LIST="[${GPUS}]"
NRE_CACHE=$(( (PER_GPU + 1) * 4 ))                   # (concurrent + buffer) x 4 cameras
echo "GPUs ${GPUS} : ${NGPU} x ${PER_GPU} = ${TOTAL} concurrent rollouts"

# Per-box venv on local NVMe, not on shared EFS (avoids cross-instance UID clashes).
# VIRTUAL_ENV is needed too: setup_local_env.sh calls `uv pip install` for utils_rs,
# and `uv pip` ignores UV_PROJECT_ENVIRONMENT -- it only honors VIRTUAL_ENV.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/dlami/nvme/rod/venvs/alpasim}"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"

# --- Scenes: one per nav command, chosen from the full A1.5 public_2601 eval ---
# Baseline scores below are that run's nav-unconditioned results.
#   LEFT     08990ec8  LEFT on all 197 route updates            score 0.820
#   RIGHT    0a18c5a4  route turns right, but the model narrates turning LEFT
#                      into a driveway (flip-flopping)            score 0.728
#   STRAIGHT 054b5901  clean straight                             score 1.000
#                      <- control, must not regress
SCENES='[clipgt-08990ec8-0ba9-4284-9919-65b71750a5fa,clipgt-0a18c5a4-9aca-4efd-b604-c75f3269c502,clipgt-054b5901-fdcb-4146-b125-eb2bb333cf02]'

# --- NVMe locations (per-instance, ephemeral) -----------------------------
NVME_ROOT="${NVME_ROOT:-/opt/dlami/nvme/rod}"
NVME_HF="${NVME_ROOT}/cache/hf"
NVME_DATA="${NVME_ROOT}/datasets/alpasim/data/nre-artifacts"
mkdir -p "${NVME_HF}" "${NVME_DATA}/all-usdzs"

# The renderer bind-mounts ${defines.sensordata}/ego-hoods. Scenes auto-download,
# but the hood masks ship in the repo, so stage them onto the ephemeral NVMe.
if [[ ! -d "${NVME_DATA}/ego-hoods" ]]; then
  cp -r data/nre-artifacts/ego-hoods "${NVME_DATA}/"
fi

# --- HuggingFace auth -----------------------------------------------------
export HF_HOME="${NVME_HF}"
if [[ ! -f "${NVME_HF}/token" && -f "${HOME}/.cache/huggingface/token" ]]; then
  cp "${HOME}/.cache/huggingface/token" "${NVME_HF}/token"
fi
if [[ -z "${HF_TOKEN:-}" && -f "${NVME_HF}/token" ]]; then
  export HF_TOKEN="$(cat "${NVME_HF}/token")"
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "❌ HF_TOKEN not set and no ${NVME_HF}/token file. Run 'huggingface-cli login' or export HF_TOKEN first." >&2
  exit 1
fi

source setup_local_env.sh

STAMP="$(hostname -s)_$(date +%m%d_%H%M)"

run_condition() {
  local name="$1" use_nav="$2"
  local log_dir="${PWD}/nav_${name}_${STAMP}"

  echo
  echo "=============================================================="
  echo " condition: ${name}   (driver.route.use_nav_text=${use_nav})"
  echo " rollouts : ${ROLLOUTS} per scene x 3 scenes"
  echo " log dir  : ${log_dir}"
  echo "=============================================================="

  uv run alpasim_wizard \
    deploy=local topology=1gpu driver=alpamayo1_5 \
    '+runtime.endpoints.startup_timeout_s=900' \
    "services.renderer.gpus=${GPU_LIST}" \
    "services.driver.gpus=${GPU_LIST}" \
    "services.physics.gpus=${GPU_LIST}" \
    "services.trafficsim.gpus=${GPU_LIST}" \
    "defines.nre_cache_size=${NRE_CACHE}" \
    "runtime.endpoints.renderer.n_concurrent_rollouts=${PER_GPU}" \
    "runtime.endpoints.driver.n_concurrent_rollouts=${PER_GPU}" \
    "runtime.endpoints.physics.n_concurrent_rollouts=${PER_GPU}" \
    "runtime.endpoints.controller.n_concurrent_rollouts=${TOTAL}" \
    "defines.hf_cache=${NVME_HF}" \
    "defines.sensordata=${NVME_DATA}" \
    "scenes.scene_cache=${NVME_DATA}" \
    'scenes.test_suite_id=null' \
    "scenes.scene_ids=${SCENES}" \
    "runtime.simulation_config.n_rollouts=${ROLLOUTS}" \
    "driver.route.use_nav_text=${use_nav}" \
    'driver.route.use_waypoint_commands=true' \
    'eval.allow_aggregation_with_failed_rollouts=true' \
    'eval.video.video_layouts=[DEFAULT_QUAD]' \
    "wizard.log_dir=${log_dir}"

  echo "${log_dir}" >> "${PWD}/.nav_ablation_dirs_${STAMP}"
}

# use_waypoint_commands is on in both arms so the video overlay reports the real
# route command instead of the STRAIGHT default. It does not reach the model
# (Alpamayo's _encode_command returns None), so it cannot confound the ablation.
# Only use_nav_text changes what the model sees.
if [[ "${ONLY}" == "both" || "${ONLY}" == "off" ]]; then
  run_condition off false
fi
if [[ "${ONLY}" == "both" || "${ONLY}" == "on" ]]; then
  run_condition on true
fi

echo
echo "Comparing results..."
uv run python compare_nav_ablation.py $(cat "${PWD}/.nav_ablation_dirs_${STAMP}" | tr '\n' ' ')
