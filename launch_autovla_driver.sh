#!/usr/bin/env bash
# External AutoVLA driver process for AlpaSim (driver_source=external_static).
#
# Runs on the host in the alpasim-e2e venv, so the wizard does not own it and
# the alpasim-base image does not need rebuilding to carry the plugin.
#
#   RUN_DIR=... GPU=7 PORT=6890 DRIVER_CFG=autovla_standalone_10hz ./launch_autovla_driver.sh
#
# DRIVER_CFG: autovla_standalone (2 Hz control) | autovla_standalone_10hz (10 Hz).
# Set AUTOVLA_DEBUG_DIR to dump per-step frames + plans.
set -uo pipefail
REPO=/mnt/efs/users/rod/repos/alpasim-e2e
VENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
GPU="${GPU:-7}"
PORT="${PORT:-6890}"
DRIVER_CFG="${DRIVER_CFG:-autovla_standalone_10hz}"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME=/opt/dlami/nvme/rod/cache/hf
# OpenCV defaults to one thread per core (96 here) and several lanes build their
# f-theta rectification maps at the same moment, so cap it. NOTE: tested on its
# own (rerun4_threadcap_only) this does NOT fix the first-step stall -- map builds
# still took 12-19 s and both rollouts died. What fixes that is doing the work off
# the driver's event loop. Kept because it costs ~0.3 s and bounds oversubscription
# now that three camera handlers rectify concurrently.
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-8}"
export PYTHONUNBUFFERED=1
mkdir -p "$RUN_DIR/driver"
cd "$REPO"
exec "$VENV/bin/python" -m alpasim_driver.main \
  --config-path="$REPO/plugins/autovla_driver/configs" \
  --config-name="$DRIVER_CFG" \
  port="$PORT" \
  output_dir="$RUN_DIR/driver"
