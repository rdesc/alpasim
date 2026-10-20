#!/usr/bin/env bash
# Fast-forward rod/e2e-train-sample400 onto the merged quad-cam branch, once
# the mix200 eval has finished. Deferred because alpasim-e2e/src is bind-mounted
# into the running containers, so changing eval code mid-run would hit the
# aggregation stage that runs after the last rollout.
set -uo pipefail
OUT=/mnt/efs/users/rod/repos/alpasim
LOG="$OUT/ff_e2e_branch.log"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "=== fast-forward waiter started (pid $$) ==="
for i in $(seq 1 960); do   # up to 8h
  grep -q "wizard exited rc=" "$OUT/launch_train.log" 2>/dev/null && break
  sleep 30
done
if ! grep -q "wizard exited rc=" "$OUT/launch_train.log" 2>/dev/null; then
  say "ABORT: mix200 did not finish within 8h"; exit 1
fi
# Let any in-flight aggregation finish before touching the bind-mounted tree.
sleep 120
cd /mnt/efs/users/rod/repos/alpasim-e2e || { say "ABORT: no worktree"; exit 1; }
out=$(git merge --ff-only rod/e2e-quadcam-video 2>&1)
say "git merge --ff-only: $out"
say "HEAD now: $(git log --oneline -1)"
