#!/usr/bin/env bash
# Full closed-loop eval over the public_2601 suite (916 scenes), all assets on NVMe.
# Weights + scenes auto-download from HuggingFace on first run (needs HF_TOKEN).
#
# Usage:
#   ./run_full_eval_nvme.sh alpamayo1        # Alpamayo-R1-10B   (run on box A)
#   ./run_full_eval_nvme.sh alpamayo1_5      # Alpamayo-1.5-10B  (run on box B)
#
# Smoke test on the first N scenes instead of all 916:
#   LIMIT=5 ./run_full_eval_nvme.sh alpamayo1_5
#
# Override the NVMe root per box if it differs:
#   NVME_ROOT=/opt/dlami/nvme/rod ./run_full_eval_nvme.sh alpamayo1_5
set -eo pipefail

# Per-box venv on local NVMe, not on shared EFS (avoids cross-instance UID clashes).
# VIRTUAL_ENV is needed too: setup_local_env.sh calls `uv pip install` for utils_rs,
# and `uv pip` ignores UV_PROJECT_ENVIRONMENT -- it only honors VIRTUAL_ENV.
export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim

DRIVER="${1:?usage: $0 <alpamayo1|alpamayo1_5>}"
LIMIT="${LIMIT:-0}"       # 0 = no limit (full suite)
SKIP_COT="${SKIP_COT:-false}"   # true = disable Alpamayo 1.5 Chain-of-Cognition (A1.5 only)
cd /mnt/efs/users/rod/repos/alpasim

# --- Optional: run Alpamayo 1.5 with reasoning disabled -------------------
# skip_cot injects an empty Chain-of-Cognition, so the trajectory is decoded
# without the autoregressive reasoning rollout. It only works with a coc_text-
# capable alpamayo1_5; the pinned NVLabs build lacks it, so guard against a
# silent no-op by requiring the fork overlay in base_config.yaml.
EXTRA=()
TAG=""
if [[ "${SKIP_COT}" == "true" ]]; then
  if [[ "${DRIVER}" != "alpamayo1_5" ]]; then
    echo "❌ SKIP_COT=true is only supported for driver=alpamayo1_5." >&2
    exit 1
  fi
  if ! grep -q "site-packages/alpamayo1_5" src/wizard/configs/base_config.yaml; then
    echo "❌ SKIP_COT=true needs the coc_text-capable alpamayo1_5 overlay in" >&2
    echo "   src/wizard/configs/base_config.yaml (driver volumes). Without it the" >&2
    echo "   pinned build ignores coc_text and skip_cot silently no-ops." >&2
    exit 1
  fi
  EXTRA+=( "driver.model.skip_cot=true" )
  TAG="_nocot"
fi

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
# Seed the NVMe token from the default HF cache if it isn't there yet. HF_HOME is
# repointed at NVMe, so ~/.cache/huggingface/token is otherwise invisible.
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

if [[ "${LIMIT}" -gt 0 ]]; then
  LOG_DIR="$PWD/smoke${LIMIT}_${DRIVER}${TAG}_$(hostname -s)_$(date +%m%d_%H%M)"
else
  LOG_DIR="$PWD/eval_${DRIVER}${TAG}_public_2601_$(hostname -s)_$(date +%m%d_%H%M)"
fi

uv run alpasim_wizard \
  deploy=local topology=8gpu_12rollouts driver="${DRIVER}" \
  '+runtime.endpoints.startup_timeout_s=900' \
  "defines.hf_cache=${NVME_HF}" \
  "defines.sensordata=${NVME_DATA}" \
  "scenes.scene_cache=${NVME_DATA}" \
  'scenes.test_suite_id=public_2601' \
  'scenes.scene_ids=null' \
  "scenes.limit_to_first_n=${LIMIT}" \
  "${EXTRA[@]}" \
  "wizard.log_dir=${LOG_DIR}"

echo "Done. Results: ${LOG_DIR}/aggregate/results-summary.json"
