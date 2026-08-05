#!/usr/bin/env python
"""Compare nav-conditioning ablation runs scene by scene.

Usage:
    uv run python compare_nav_ablation.py nav_off_<stamp> nav_on_<stamp>

Reads each run's aggregate/results-summary.json. If a run has no aggregate/
(the runtime skips it when any rollout failed), aggregate it first with:

    python -m eval.aggregation.main --array_job_dir <dir> --config_path <dir>/eval-config.yaml
"""

import json
import pathlib
import statistics
import sys

# What each scene is meant to probe, keyed by clipgt id prefix.
SCENE_LABELS = {
    "clipgt-08990ec8": "LEFT      (all 197 updates)",
    "clipgt-0a18c5a4": "RIGHT     (model turns left)",
    "clipgt-054b5901": "STRAIGHT  (control)",
}


def load(run_dir: pathlib.Path) -> dict[str, list[dict]]:
    summary = run_dir / "aggregate" / "results-summary.json"
    if not summary.exists():
        sys.exit(
            f"No aggregate in {run_dir}.\n"
            f"Run: python -m eval.aggregation.main --array_job_dir {run_dir} "
            f"--config_path {run_dir}/eval-config.yaml"
        )
    by_scene: dict[str, list[dict]] = {}
    for rollout in json.loads(summary.read_text())["rollouts"]:
        by_scene.setdefault(rollout["clipgt_id"], []).append(rollout)
    return by_scene


def mean_score(rollouts: list[dict]) -> float | None:
    scores = [r["score"] for r in rollouts if r.get("score") is not None]
    return statistics.fmean(scores) if scores else None


def fmt(rollouts: list[dict] | None) -> str:
    if not rollouts:
        return "     --      "
    m = mean_score(rollouts)
    if m is None:
        return "     --      "
    passed = sum(1 for r in rollouts if r.get("passed"))
    spread = ""
    if len(rollouts) > 1:
        scores = [r["score"] for r in rollouts if r.get("score") is not None]
        spread = f" ±{statistics.pstdev(scores):.2f}"
    return f"{m:.3f}{spread} ({passed}/{len(rollouts)})"


def main() -> int:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    runs = [pathlib.Path(d) for d in sys.argv[1:]]
    data = {run.name: load(run) for run in runs}

    names = list(data)
    print()
    print(f"{'scene':30s} " + " ".join(f"{n[:22]:>22s}" for n in names))
    print("-" * (30 + 23 * len(names)))

    scenes = sorted({s for d in data.values() for s in d})
    for scene in scenes:
        label = next(
            (v for k, v in SCENE_LABELS.items() if scene.startswith(k)), scene[:28]
        )
        cells = " ".join(f"{fmt(data[n].get(scene)):>22s}" for n in names)
        print(f"{label:30s} {cells}")

    print("-" * (30 + 23 * len(names)))
    overall = []
    for n in names:
        all_rollouts = [r for rs in data[n].values() for r in rs]
        overall.append(f"{fmt(all_rollouts):>22s}")
    print(f"{'ALL':30s} " + " ".join(overall))
    print("\nmean score ±stdev (passed/total). Score is 0 on collision or offroad,")
    print("otherwise it is the progress term, so a passing scene can still score low.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
