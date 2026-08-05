#!/usr/bin/env bash
# Chain-of-Cognition (CoT) ablation for Alpamayo 1.5.
#
# Runs the same scenes twice -- once normally (the model autoregressively decodes
# its reasoning before the trajectory) and once with driver.model.skip_cot=true
# (an empty Chain-of-Cognition is injected, so the reasoning rollout is skipped).
# Then prints, per scene, the driving score, and per arm, the mean driver/drive
# latency. The question: how much faster is inference without CoT, and does the
# driving quality survive?
#
# Requires the rod-dev branch (skip_cot plumbing + DEFAULT_QUAD video layout).
#
# Usage:
#   ./run_cot_ablation.sh                  # 3 rollouts/scene/arm, both arms
#   ROLLOUTS=1 ./run_cot_ablation.sh       # quick smoke (drive latency is stable
#                                          #   even at 1 rollout -- thousands of calls)
#   ONLY=cot_off ./run_cot_ablation.sh     # only the skip-CoT arm
#   ONLY=cot_on  ./run_cot_ablation.sh     # only the baseline (CoT on) arm
#   GPUS=1,5,6 ./run_cot_ablation.sh       # pin to specific GPUs
#   PER_GPU=3 ./run_cot_ablation.sh        # concurrent Alpamayo rollouts per GPU
#   VIDEO=DEFAULT ./run_cot_ablation.sh    # lighter single-cam video (default DEFAULT_QUAD)
set -eo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO}"

ROLLOUTS="${ROLLOUTS:-3}"   # rollouts per scene per arm; scores need a few, latency needs 1
ONLY="${ONLY:-both}"        # both | cot_on | cot_off
VIDEO="${VIDEO:-DEFAULT_QUAD}"

# --- GPUs -----------------------------------------------------------------
# Default: auto-detect idle GPUs (<1 GiB used). All services share the set.
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

# --- Scenes: reuse the nav-ablation set (one per nav command) -------------
#   LEFT     08990ec8   RIGHT 0a18c5a4   STRAIGHT 054b5901 (control)
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

run_arm() {
  local name="$1" skip_cot="$2"
  local log_dir="${PWD}/${name}_${STAMP}"

  echo
  echo "=============================================================="
  echo " arm      : ${name}   (driver.model.skip_cot=${skip_cot})"
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
    "driver.model.skip_cot=${skip_cot}" \
    'eval.allow_aggregation_with_failed_rollouts=true' \
    "eval.video.video_layouts=[${VIDEO}]" \
    "wizard.log_dir=${log_dir}"

  echo "${log_dir}" >> "${PWD}/.cot_ablation_dirs_${STAMP}"
}

# Baseline first so it is arm 1 in the comparison (speedup is measured against it).
if [[ "${ONLY}" == "both" || "${ONLY}" == "cot_on" ]]; then
  run_arm cot_on  false   # reasoning rollout runs (baseline)
fi
if [[ "${ONLY}" == "both" || "${ONLY}" == "cot_off" ]]; then
  run_arm cot_off true    # reasoning rollout skipped (experimental)
fi

echo
echo "Comparing results (score + drive latency)..."
uv run python compare_cot_ablation.py $(cat "${PWD}/.cot_ablation_dirs_${STAMP}" | tr '\n' ' ')
