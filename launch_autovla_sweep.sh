#!/usr/bin/env bash
# AutoVLA closed-loop sweep over an AlpaSim test suite, sharded across GPUs.
#
# Each lane is one GPU running its own external AutoVLA driver plus its own
# wizard over a disjoint slice of the scene list. AutoVLA's driver has no
# batched path (max_batch_size=1), so one driver per GPU beats one driver
# serving many concurrent rollouts.
#
#   SWEEP=/path/to/out N_LANES=8 N_SCENES=0 ./launch_autovla_sweep.sh
#
#   SWEEP      output root (required)
#   LANE_GPUS  comma-separated GPU ids, one lane each (default 0..7)
#   N_SCENES   limit to the first N scenes of the suite (0 = all 441)
#   SUITE      test suite id (default nurec_curated_val)
#   SCENES_FILE  explicit newline-separated scene ids to run instead of the
#              suite list -- used to re-run the scenes a previous sweep lost
#   WIZ_DRIVER autovla_10hz (10 Hz control, default) | autovla (2 Hz)
#   DRIVER_CFG must match WIZ_DRIVER: autovla_standalone_10hz | autovla_standalone
#
# Renderer and driver share each GPU: the renderer peaks around 30 GB and
# AutoVLA holds ~10 GB, well inside an 80 GB card.
set -uo pipefail

REPO=/mnt/efs/users/rod/repos/alpasim-e2e
VENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e
CACHE=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts

SWEEP="${SWEEP:?set SWEEP}"
LANE_GPUS="${LANE_GPUS:-0,1,2,3,4,5,6,7}"
N_SCENES="${N_SCENES:-0}"
SUITE="${SUITE:-nurec_curated_val}"
WIZ_DRIVER="${WIZ_DRIVER:-autovla_10hz}"
DRIVER_CFG="${DRIVER_CFG:-autovla_standalone_10hz}"
BASE_DRIVER_PORT="${BASE_DRIVER_PORT:-6900}"
BASE_WIZ_PORT="${BASE_WIZ_PORT:-7200}"
# route_start_offset_m=40.0 matches the Alpamayo 1.5 curated_val baseline runs,
# so the numbers are comparable.
ROUTE_OFFSET="${ROUTE_OFFSET:-40.0}"

mkdir -p "$SWEEP"
LOG="$SWEEP/sweep.log"
say() { echo "$(date '+%F %T') [sweep] $*" | tee -a "$LOG"; }

IFS=',' read -r -a GPUS <<< "$LANE_GPUS"
N_LANES=${#GPUS[@]}

# --- scene list for the suite -------------------------------------------------
SCENE_LIST="$SWEEP/scenes.txt"
if [[ -n "${SCENES_FILE:-}" ]]; then
  # Explicit list (e.g. the scenes a previous sweep lost). Strip CR and blanks;
  # a stray \r inside a scene id breaks Hydra's override grammar.
  [[ -r "$SCENES_FILE" ]] || { say "ABORT: cannot read SCENES_FILE=$SCENES_FILE"; exit 1; }
  tr -d '\r' < "$SCENES_FILE" | awk 'NF' | sort -u > "$SCENE_LIST"
else
  # The suite CSVs are CRLF; a stray \r inside a scene id breaks Hydra's override grammar.
  awk -F, -v s="$SUITE" '{sub(/\r$/, "")} $1==s {print $2}' \
    "$REPO/data/scenes/sim_suites.csv" "$REPO/data/scenes/sim_suites_curated.csv" \
    2>/dev/null | sort -u > "$SCENE_LIST"
fi
TOTAL=$(wc -l < "$SCENE_LIST")
if [[ "$N_SCENES" -gt 0 ]]; then
  head -n "$N_SCENES" "$SCENE_LIST" > "$SCENE_LIST.tmp" && mv "$SCENE_LIST.tmp" "$SCENE_LIST"
  TOTAL=$(wc -l < "$SCENE_LIST")
fi
if [[ "$TOTAL" -eq 0 ]]; then say "ABORT: no scenes for suite $SUITE"; exit 1; fi
say "suite=$SUITE scenes=$TOTAL lanes=$N_LANES gpus=$LANE_GPUS driver=$WIZ_DRIVER"

export HF_HOME=/opt/dlami/nvme/rod/cache/hf
export HF_TOKEN="$(cat /opt/dlami/nvme/rod/cache/hf/token 2>/dev/null)"
cd "$REPO" || { say "ABORT: no repo"; exit 1; }

PIDS=()
for i in "${!GPUS[@]}"; do
  GPU="${GPUS[$i]}"
  LANE_DIR="$SWEEP/lane$i"
  DRIVER_PORT=$((BASE_DRIVER_PORT + i))
  WIZ_PORT=$((BASE_WIZ_PORT + i * 40))
  mkdir -p "$LANE_DIR"

  # Round-robin shard so adjacent (often similar) scenes spread across lanes.
  awk -v n="$N_LANES" -v k="$i" 'NR % n == k' "$SCENE_LIST" > "$LANE_DIR/scenes.txt"
  LANE_N=$(wc -l < "$LANE_DIR/scenes.txt")
  if [[ "$LANE_N" -eq 0 ]]; then say "lane$i: no scenes, skipping"; continue; fi
  SCENE_IDS="[$(paste -sd, "$LANE_DIR/scenes.txt")]"

  (
    # --- driver -------------------------------------------------------------
    RUN_DIR="$LANE_DIR" GPU="$GPU" PORT="$DRIVER_PORT" DRIVER_CFG="$DRIVER_CFG" \
      "$REPO/launch_autovla_driver.sh" > "$LANE_DIR/driver.log" 2>&1 &
    DPID=$!
    trap 'kill $DPID 2>/dev/null' EXIT

    for _ in $(seq 1 60); do
      ss -ltn | grep -q ":$DRIVER_PORT " && break
      kill -0 $DPID 2>/dev/null || break
      sleep 5
    done
    if ! ss -ltn | grep -q ":$DRIVER_PORT "; then
      echo "lane$i: driver failed to bind :$DRIVER_PORT" >> "$LOG"; exit 1
    fi
    echo "$(date '+%F %T') [sweep] lane$i: gpu=$GPU driver=:$DRIVER_PORT scenes=$LANE_N" >> "$LOG"

    # --- wizard -------------------------------------------------------------
    "$VENV/bin/alpasim_wizard" \
      deploy=local topology=1gpu "driver=$WIZ_DRIVER" driver_source=external_static \
      +nurec_scenes=curated_val \
      wizard.debug_flags.use_localhost=true \
      "+wizard.external_services.driver=[localhost:$DRIVER_PORT]" \
      scenes.test_suite_id=null \
      "scenes.scene_ids=$SCENE_IDS" \
      "scenes.scene_cache=$CACHE" \
      "services.renderer.gpus=[$GPU]" \
      "services.physics.gpus=[$GPU]" \
      "services.trafficsim.gpus=[$GPU]" \
      runtime.endpoints.renderer.n_concurrent_rollouts=1 \
      runtime.simulation_config.n_rollouts=1 \
      "runtime.simulation_config.route_start_offset_m=$ROUTE_OFFSET" \
      eval.allow_aggregation_with_failed_rollouts=true \
      eval.parse_unstructured_debug_info=false \
      '+runtime.endpoints.startup_timeout_s=900' \
      defines.filesystem=/mnt/efs/users/rod/repos/alpasim/data \
      defines.hf_cache=/opt/dlami/nvme/rod/cache/hf \
      "defines.sensordata=$CACHE" \
      wizard.baseport=$WIZ_PORT \
      wizard.log_dir="$LANE_DIR" >> "$LANE_DIR/wizard.log" 2>&1
    rc=$?
    echo "$(date '+%F %T') [sweep] lane$i: wizard rc=$rc rollouts=$(find "$LANE_DIR/rollouts" -name _complete 2>/dev/null | wc -l)/$LANE_N" >> "$LOG"
  ) &
  PIDS+=($!)
  sleep 10   # stagger container startup so docker/NRE don't thrash
done

say "all lanes launched; waiting"
for p in "${PIDS[@]}"; do wait "$p"; done
say "lanes finished; rollouts total=$(find "$SWEEP" -name _complete 2>/dev/null | wc -l)/$TOTAL"

# --- combined aggregation across lanes ---------------------------------------
say "aggregating"
# alpasim-aggregation treats each lane dir as an array job; any lane's
# eval-config.yaml describes the (identical) metric pipeline.
EVAL_CFG="$(ls "$SWEEP"/lane*/eval-config.yaml 2>/dev/null | head -1)"
if [[ -n "$EVAL_CFG" ]]; then
  "$VENV/bin/alpasim-aggregation" --array_job_dir "$SWEEP" --config_path "$EVAL_CFG" >> "$LOG" 2>&1 \
    || say "combined aggregation failed; per-lane aggregates remain under $SWEEP/lane*/aggregate"
else
  say "no lane eval-config.yaml; skipping combined aggregation"
fi
say "done -> $SWEEP"
