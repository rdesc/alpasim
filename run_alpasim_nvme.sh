#!/usr/bin/env bash
set -eo pipefail
cd /mnt/efs/users/rod/repos/alpasim

NVME_DATA=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts
NVME_HF=/opt/dlami/nvme/rod/cache/hf

  # Point HF token/cache at the NVMe cache (has the token file + weights)
export HF_HOME="${NVME_HF}"
export HF_TOKEN="$(cat "${NVME_HF}/token")"

source setup_local_env.sh      # compiles protos, sets up env

uv run alpasim_wizard \
    deploy=local topology=1gpu driver=alpamayo1_5 \
    '+runtime.endpoints.startup_timeout_s=900' \
    'services.renderer.gpus=[6]' \
    'services.driver.gpus=[6]' \
    'services.physics.gpus=[6]' \
    "defines.hf_cache=${NVME_HF}" \
    "defines.sensordata=${NVME_DATA}" \
    "scenes.scene_cache=${NVME_DATA}" \
    'scenes.scene_ids=[clipgt-02eadd92-02f1-46d8-86fe-a9e338fed0b6]' \
    'scenes.test_suite_id=null' \
    wizard.log_dir=$PWD/tutorial_nvme_test \
    'eval.video.video_layouts=[DEFAULT]'
