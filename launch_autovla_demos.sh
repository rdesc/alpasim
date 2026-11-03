#!/usr/bin/env bash
# Render a handful of AutoVLA rollouts with full per-step introspection dumps,
# one scene per GPU, for the visual walkthrough. Uses AUTOVLA_DEBUG_DIR, which
# writes 3 JPEGs + a step.json per inference — fine for a few scenes, far too
# much for a sweep.
set -uo pipefail
REPO=/mnt/efs/users/rod/repos/alpasim-e2e
OUT="${OUT:-$REPO/autovla_eval_results/2026-10-30_autovla_demos}"
mkdir -p "$OUT"

# scene | gpu | driver port | wizard baseport
LANES=(
  "clipgt-032b6f21-6799-4ad8-8712-0405f087ac4a|1|6891|7500"
  "clipgt-0026708c-a386-4dd2-82b1-286da952c68f|2|6892|7540"
  "clipgt-000525f6-3999-4812-9924-8adff40ca514|3|6893|7580"
  "clipgt-17bec9f8-aa14-4091-ae83-b05714fe6e81|4|6894|7620"
)

for entry in "${LANES[@]}"; do
  IFS='|' read -r SCENE GPU PORT BASEPORT <<< "$entry"
  SHORT="${SCENE:7:8}"
  RUN_DIR="$OUT/$SHORT"
  mkdir -p "$RUN_DIR/debug"
  (
    RUN_DIR="$RUN_DIR" GPU="$GPU" PORT="$PORT" \
      DRIVER_CFG=autovla_standalone_10hz AUTOVLA_DEBUG_DIR="$RUN_DIR/debug" \
      "$REPO/launch_autovla_driver.sh" > "$RUN_DIR/driver.log" 2>&1 &
    DPID=$!
    trap 'kill $DPID 2>/dev/null' EXIT
    for _ in $(seq 1 60); do
      ss -ltn | grep -q ":$PORT " && break
      kill -0 $DPID 2>/dev/null || break
      sleep 5
    done
    ss -ltn | grep -q ":$PORT " || { echo "driver failed on :$PORT" >> "$OUT/demos.log"; exit 1; }
    echo "$(date '+%T') $SHORT gpu=$GPU driver=:$PORT" >> "$OUT/demos.log"

    RUN_DIR="$RUN_DIR" SCENE="$SCENE" WIZ_DRIVER=autovla_10hz RENDER_GPU="$GPU" \
      BASEPORT="$BASEPORT" N_SIM_STEPS=200 DRIVER_ADDR="localhost:$PORT" \
      "$REPO/launch_autovla_sim.sh" > "$RUN_DIR/wizard.log" 2>&1
    echo "$(date '+%T') $SHORT done rc=$?" >> "$OUT/demos.log"
  ) &
  sleep 8
done
wait
echo "$(date '+%T') all demos finished" >> "$OUT/demos.log"
