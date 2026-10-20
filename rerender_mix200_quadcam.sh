#!/usr/bin/env bash
# Wait for the mix200 eval to finish, then re-render its videos with the
# DEFAULT_QUAD layout (2x2 camera grid + BEV + metrics + reasoning caption).
#
# Runs from the alpasim-video worktree, which carries the quad-cam port. The
# live alpasim-e2e worktree is bind-mounted into the eval containers, so it is
# deliberately not used here.
#
# Re-rendering goes through alpasim-reeval: it recomputes metrics and videos
# from the existing ASLs, so no re-simulation and no GPU time.
set -uo pipefail

OUT=/mnt/efs/users/rod/repos/alpasim
RUN="$OUT/alpamayo_eval_results/2026-10-20_train_mix200"
VIDEO_REPO=/mnt/efs/users/rod/repos/alpasim-video
LOG="$OUT/rerender_mix200.log"

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "=== re-render waiter started (pid $$) ==="

# --- 1. wait for the eval to finish ---------------------------------------
# Gate on the launcher's completion line, not on pgrep: pgrep -f matches full
# command lines, so any shell mentioning the script is a false positive.
for i in $(seq 1 960); do   # up to 8h
  grep -q "wizard exited rc=" "$OUT/launch_train.log" 2>/dev/null && break
  [[ $((i % 40)) -eq 0 ]] && say "waiting for mix200 ($(grep -c 'Session COMPLETED' "$OUT/train_run.log" 2>/dev/null) scenes done)"
  sleep 30
done
if ! grep -q "wizard exited rc=" "$OUT/launch_train.log" 2>/dev/null; then
  say "ABORT: mix200 did not finish within 8h"; exit 1
fi
say "mix200 finished: $(grep 'wizard exited rc=' "$OUT/launch_train.log" | tail -1)"

if [[ ! -f "$RUN/aggregate/results-summary.json" ]]; then
  say "ABORT: no aggregate at $RUN -- run may have failed"; exit 1
fi

# --- 2. preserve the original aggregate -----------------------------------
if [[ ! -d "$RUN/aggregate_backup_singlecam" ]]; then
  cp -r "$RUN/aggregate" "$RUN/aggregate_backup_singlecam"
  say "backed up original aggregate -> aggregate_backup_singlecam"
fi

# --- 3. switch the layout --------------------------------------------------
cp "$RUN/eval-config.yaml" "$RUN/eval-config.yaml.bak"
python3 - "$RUN/eval-config.yaml" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s2 = re.sub(r"(video_layouts:\n)(\s+)- DEFAULT\b(?!_)", r"\1\2- DEFAULT_QUAD", s)
open(p, "w").write(s2)
print("layout set to DEFAULT_QUAD" if s2 != s else "WARNING: layout unchanged")
PY
say "eval-config video_layouts -> DEFAULT_QUAD"

# --- 4. re-render ----------------------------------------------------------
cd "$VIDEO_REPO" || { say "ABORT: no video worktree"; exit 1; }
export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim-video
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-video
say "re-rendering $RUN"
uv run --no-sync alpasim-reeval "$RUN" >> "$OUT/rerender_mix200_reeval.log" 2>&1
rc=$?
n=$(find "$RUN/aggregate/videos" -name '*default_quad.mp4' 2>/dev/null | wc -l)
say "reeval exited rc=$rc ; default_quad videos: $n"
