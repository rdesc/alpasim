#!/usr/bin/env bash
# Single-scene closed-loop AlpaSim run with AutoVLA as the ego driver.
#
# The driver runs as an external host process (launch_autovla_driver.sh), so
# only renderer/physics/trafficsim/controller/runtime come up in containers and
# the alpasim-base image needs no rebuild. debug_flags.use_localhost adds
# network_mode:host so those containers can reach the driver on localhost.
#
#   RUN_DIR=... WIZ_DRIVER=autovla_10hz RENDER_GPU=6 ./launch_autovla_sim.sh
#
# WIZ_DRIVER: autovla (2 Hz control) | autovla_10hz (10 Hz). Must match the
# DRIVER_CFG the driver process was started with — the subsample factor and the
# camera rate have to agree.
set -uo pipefail
REPO=/mnt/efs/users/rod/repos/alpasim-e2e
VENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e
CACHE=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts
SCENE="${SCENE:-clipgt-17bec9f8-aa14-4091-ae83-b05714fe6e81}"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
DRIVER_ADDR="${DRIVER_ADDR:-localhost:6890}"
WIZ_DRIVER="${WIZ_DRIVER:-autovla_10hz}"
N_SIM_STEPS="${N_SIM_STEPS:-200}"
RENDER_GPU="${RENDER_GPU:-6}"
BASEPORT="${BASEPORT:-7100}"

export HF_HOME=/opt/dlami/nvme/rod/cache/hf
export HF_TOKEN="$(cat /opt/dlami/nvme/rod/cache/hf/token 2>/dev/null)"
mkdir -p "$RUN_DIR"
cd "$REPO"

"$VENV/bin/alpasim_wizard" \
  deploy=local topology=2gpu "driver=$WIZ_DRIVER" driver_source=external_static \
  wizard.debug_flags.use_localhost=true \
  "+wizard.external_services.driver=[$DRIVER_ADDR]" \
  scenes.scene_ids="[$SCENE]" \
  scenes.scenes_csv="[$REPO/data/scenes/sim_scenes.csv,$REPO/data/scenes/sim_scenes_2604.csv]" \
  scenes.scene_cache="$CACHE" \
  "services.renderer.gpus=[$RENDER_GPU]" \
  "services.physics.gpus=[$RENDER_GPU]" \
  "services.trafficsim.gpus=[$RENDER_GPU]" \
  runtime.endpoints.renderer.n_concurrent_rollouts=1 \
  runtime.simulation_config.n_rollouts=1 \
  runtime.simulation_config.n_sim_steps=$N_SIM_STEPS \
  eval.parse_unstructured_debug_info=true \
  '+runtime.endpoints.startup_timeout_s=900' \
  defines.filesystem=/mnt/efs/users/rod/repos/alpasim/data \
  defines.hf_cache=/opt/dlami/nvme/rod/cache/hf \
  "defines.sensordata=$CACHE" \
  wizard.baseport=$BASEPORT \
  wizard.log_dir="$RUN_DIR"
echo "wizard exited rc=$?"
