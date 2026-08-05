#!/usr/bin/env python
"""Compare Chain-of-Cognition ablation runs: driving score vs inference latency.

For each run it reports, per scene, the mean driving score (does quality hold when
the reasoning rollout is skipped?) and, per run, the mean ``driver/drive`` latency
from the Prometheus TSDB (how much faster is it?). The whole point of skip_cot is a
latency win; this shows whether that win costs anything in score.

Usage:
    uv run python compare_cot_ablation.py cot_on_<stamp> cot_off_<stamp>

Reads each run's aggregate/results-summary.json. If a run has no aggregate/
(the runtime skips it when any rollout failed), aggregate it first with:

    python -m eval.aggregation.main --array_job_dir <dir> --config_path <dir>/eval-config.yaml
"""

import json
import pathlib
import statistics
import sys

from measure_drive_latency import measure_run

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

    # --- Scores -----------------------------------------------------------
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
    overall = {
        n: mean_score([r for rs in data[n].values() for r in rs]) for n in names
    }
    cells = " ".join(
        f"{fmt([r for rs in data[n].values() for r in rs]):>22s}" for n in names
    )
    print(f"{'ALL':30s} {cells}")
    print("\nmean score ±stdev (passed/total). Score is 0 on collision or offroad,")
    print("otherwise it is the progress term, so a passing scene can still score low.")

    # --- Latency ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("inference latency (from each run's Prometheus TSDB)")
    print("=" * 60)
    drive_means: dict[str, float] = {}
    for run in runs:
        stats = measure_run(run, ["drive", "batch_render_rgb"])
        drive = stats.get("drive", {})
        step = stats.get("step", {})
        if drive:
            drive_means[run.name] = drive["mean_s"]
        print(f"\n  {run.name}")
        if drive:
            print(f"    drive : {drive['mean_s']:.4f}s/call  ({int(drive['count']):,} calls)")
        if step:
            print(f"    step  : {step['mean_s']:.4f}s/step")

    # --- Verdict ----------------------------------------------------------
    if len(drive_means) > 1:
        base_name = names[0]
        base_lat = drive_means.get(base_name)
        print("\n" + "=" * 60)
        print("verdict  (arm 1 is the baseline)")
        print("=" * 60)
        for n in names:
            lat = drive_means.get(n)
            sc = overall.get(n)
            if lat is None:
                continue
            speed = f"{base_lat / lat:.2f}x" if base_lat else "  -- "
            sc_str = f"{sc:.3f}" if sc is not None else "  -- "
            print(f"  {n[:34]:34s}  drive {lat:7.3f}s  {speed:>6s}  score {sc_str}")
        if base_lat and overall.get(base_name) is not None:
            print(
                "\n  A large speedup with little score drop means skip_cot is worth "
                "running on the full eval."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
