#!/usr/bin/env bash
# Extract the BC dataset from the 1761-scene force-GT sweep.
#
# extract_bc_dataset.py handles one ASL or a whole run directory, but a run
# directory is processed sequentially: ~3 s per scene is ~90 min for 1761.
# Each scene writes its own independent output directory, so the work fans out
# cleanly; xargs runs 8 at a time against individual ASL paths instead.
set -uo pipefail

OUT=/mnt/efs/users/rod/repos/alpasim
REPO=/mnt/efs/users/rod/repos/alpasim-e2e
RUN=$OUT/alpamayo_eval_results/2026-10-23_bc_collect_train1761
DEST=/mnt/efs/users/rod/bc_nurec_render_curated_train1761
LOG=$OUT/run_extract_bc.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim-e2e
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e

mkdir -p "$DEST"
cd "$REPO" || { say "ABORT: no repo"; exit 1; }

mapfile -t asls < <(find "$RUN/rollouts" -name rollout.asl | sort)
say "=== extraction started: ${#asls[@]} ASLs -> $DEST ==="
start=$SECONDS

printf '%s\n' "${asls[@]}" \
  | xargs -P 8 -I{} uv run --no-sync python "$OUT/extract_bc_dataset.py" {} --out "$DEST" \
  >> "$OUT/run_extract_bc.stdout" 2>&1
rc=$?

elapsed=$((SECONDS - start))
n=$(find "$DEST" -name samples.npz | wc -l)
say "xargs rc=$rc after $((elapsed / 60))m$((elapsed % 60))s; scenes with samples.npz: $n/${#asls[@]}"
say "samples: $(python3 -c "
import json, pathlib
ms = [json.loads(p.read_text()) for p in pathlib.Path('$DEST').glob('*/manifest.json')]
print(f'{sum(m[\"samples\"] for m in ms)} total, '
      f'{sum(m[\"stationary_samples\"] for m in ms)} stationary, '
      f'across {len(ms)} scenes')" 2>&1)"
say "size: $(du -sh "$DEST" | cut -f1)"
say "dataset: $DEST"
