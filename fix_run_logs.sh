#!/usr/bin/env bash
# Post-run cleanup for the shared train_run.log problem.
#
# launch_train.sh appends the wizard's stdout to a single $OUT/train_run.log and
# copies it into the run directory at the end. Two runs therefore share one log:
# the aborted train_sample400 run and the train_mix200 run both wrote to it, so
# any per-scene count or metric scraped from it mixes the two.
#
# This waits for the eval to finish, splits the log at the boundary, drops the
# correct half into each run directory, and patches launch_train.sh so future
# runs log straight into their own run directory.
set -uo pipefail

OUT=/mnt/efs/users/rod/repos/alpasim
RUN="$OUT/alpamayo_eval_results/2026-10-20_train_mix200"
ABORTED="$OUT/alpamayo_eval_results/ABORTED_2026-10-20_train_sample400"
LOG="$OUT/fix_run_logs.log"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "=== log fixer started (pid $$) ==="

for i in $(seq 1 960); do   # up to 8h
  grep -q "wizard exited rc=" "$OUT/launch_train.log" 2>/dev/null && break
  sleep 30
done
if ! grep -q "wizard exited rc=" "$OUT/launch_train.log" 2>/dev/null; then
  say "ABORT: eval did not finish within 8h"; exit 1
fi
# launch_train.sh copies the log into the run dir right after the wizard exits.
sleep 180

# Boundary: the aborted run ends with its own hydra error line naming it.
BOUNDARY=$(grep -n "Error executing job with overrides" "$OUT/train_run.log" 2>/dev/null \
           | grep "train_sample400" | tail -1 | cut -d: -f1)
if [[ -z "$BOUNDARY" ]]; then
  say "ABORT: could not locate the run boundary in train_run.log"; exit 1
fi
total=$(wc -l < "$OUT/train_run.log")
say "boundary at line $BOUNDARY of $total"

sed -n "1,${BOUNDARY}p" "$OUT/train_run.log" > "$ABORTED/wizard.log"
sed -n "$((BOUNDARY+1)),\$p" "$OUT/train_run.log" > "$RUN/wizard.log"
say "aborted run log: $(wc -l < "$ABORTED/wizard.log") lines"
say "mix200 run log:  $(wc -l < "$RUN/wizard.log") lines"

# Sanity: the mix200 half must not mention the aborted run's directory.
if grep -q "2026-10-20_train_sample400" "$RUN/wizard.log"; then
  say "WARNING: mix200 log still references the aborted run -- check the split"
else
  say "split verified: no cross-contamination"
fi

# Future runs: log into the run directory, not a shared file.
python3 - "$OUT/launch_train.sh" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'wizard.log_dir="$RUN_DIR" >> "$OUT/train_run.log" 2>&1'
new = 'wizard.log_dir="$RUN_DIR" >> "$RUN_DIR/wizard.log" 2>&1'
if old in s:
    s = s.replace(old, new)
    # the end-of-run copy is now redundant
    s = s.replace('cp "$OUT/train_run.log" "$RUN_DIR/wizard.log" 2>/dev/null\n', '')
    open(p, 'w').write(s)
    print("patched: wizard now logs into its own run directory")
else:
    print("launch_train.sh already patched or changed; left alone")
PY
say "launcher patch: $(cd "$OUT" && grep -c 'RUN_DIR/wizard.log' launch_train.sh) reference(s) to per-run log"
say "done"
