#!/usr/bin/env python3
"""Report mean per-RPC latency from an alpasim run's Prometheus TSDB.

Each wizard run writes a Prometheus time-series DB under ``<run_dir>/prometheus``.
The RPC timers there (``alpasim_rpc_duration_seconds_{sum,count}``) are cumulative
counters, so the run-average latency of a method is simply ``sum / count`` at the
last sample. This spins up a throwaway Prometheus on a *copy* of the TSDB (the
original is never touched), queries those counters, and prints a per-method table.

Primary use: compare ``driver/drive`` latency between runs -- e.g. Chain-of-Cognition
on vs off -- alongside the end-to-end ``step`` time.

Usage:
    python measure_drive_latency.py RUN_DIR [RUN_DIR ...]
    python measure_drive_latency.py --methods drive,render_rgb RUN_DIR
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

PROM_IMAGE = "prom/prometheus:latest"

# Methods to report by default, in display order. drive = the VLA inference call
# (dominant); step = whole-step wall clock; the rest are context.
DEFAULT_METHODS = [
    "drive",
    "batch_render_rgb",
    "render_rgb",
    "run_controller_and_vehicle",
    "ground_intersection",
    "submit_image_observation",
    "submit_route",
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(base_url: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/-/ready", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, socket.timeout):
            time.sleep(0.5)
    raise TimeoutError(f"Prometheus at {base_url} not ready after {timeout_s}s")


def _query_scalar(base_url: str, expr: str) -> float | None:
    """Return the last value of ``expr`` over a wide range, or None if no data.

    The counters carry the run's original timestamps (possibly days old), so we
    sweep a long window and take the final non-empty point.
    """
    end = time.time()
    start = end - 60 * 24 * 3600  # 60 days back covers any recent run
    params = urllib.parse.urlencode(
        {"query": expr, "start": start, "end": end, "step": 900}
    )
    url = f"{base_url}/api/v1/query_range?{params}"
    with urllib.request.urlopen(url, timeout=30) as r:
        payload = json.load(r)
    result = payload["data"]["result"]
    if not result:
        return None
    values = result[0]["values"]
    if not values:
        return None
    return float(values[-1][1])


def measure_run(run_dir: Path, methods: list[str]) -> dict[str, dict[str, float]]:
    """Start Prometheus on a copy of ``run_dir``'s TSDB and return per-method stats."""
    data_dir = run_dir / "prometheus" / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No Prometheus TSDB at {data_dir}")

    tmp = Path(tempfile.mkdtemp(prefix="prom_measure_"))
    container = f"prom_measure_{uuid.uuid4().hex[:8]}"
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    try:
        # Copy so we never lock or mutate the real run dir (queries.active, WAL replay).
        shutil.copytree(data_dir, tmp / "data")
        subprocess.run(
            [
                "docker", "run", "-d", "--rm", "--name", container,
                "--user", "0",
                "-p", f"127.0.0.1:{port}:9090",
                "-v", f"{tmp / 'data'}:/prometheus:z",
                PROM_IMAGE,
                "--config.file=/etc/prometheus/prometheus.yml",
                "--storage.tsdb.path=/prometheus",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        _wait_ready(base_url)

        stats: dict[str, dict[str, float]] = {}
        for method in methods:
            sel = f'{{method="{method}"}}'
            total = _query_scalar(base_url, f"sum(alpasim_rpc_duration_seconds_sum{sel})")
            count = _query_scalar(base_url, f"sum(alpasim_rpc_duration_seconds_count{sel})")
            if not total or not count:
                continue
            stats[method] = {
                "mean_s": total / count,
                "count": count,
                "total_s": total,
            }

        # End-to-end per-step wall time (not an RPC method).
        step_total = _query_scalar(base_url, "sum(step_duration_seconds_sum)")
        step_count = _query_scalar(base_url, "sum(step_duration_seconds_count)")
        if step_total and step_count:
            stats["step"] = {
                "mean_s": step_total / step_count,
                "count": step_count,
                "total_s": step_total,
            }
        return stats
    finally:
        subprocess.run(["docker", "rm", "-f", container],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)


def _print_table(run_dir: Path, stats: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {run_dir.name} ===")
    if not stats:
        print("  (no timing data found)")
        return
    print(f"  {'method':<28} {'mean':>10} {'calls':>12} {'total':>12}")
    print(f"  {'-'*28} {'-'*10} {'-'*12} {'-'*12}")
    # step last, everything else in the order measured
    order = [m for m in stats if m != "step"] + (["step"] if "step" in stats else [])
    for method in order:
        s = stats[method]
        print(
            f"  {method:<28} {s['mean_s']:>9.4f}s {int(s['count']):>12,} "
            f"{s['total_s']:>11,.1f}s"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path, help="Run dir(s) with prometheus/")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                    help="Comma-separated RPC methods to report")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    all_stats: list[tuple[Path, dict]] = []
    for run_dir in args.run_dirs:
        stats = measure_run(run_dir, methods)
        _print_table(run_dir, stats)
        all_stats.append((run_dir, stats))

    # Cross-run drive summary when comparing more than one.
    drive = [(d, s["drive"]["mean_s"]) for d, s in all_stats if "drive" in s]
    if len(drive) > 1:
        print("\n=== driver/drive comparison ===")
        base = drive[0][1]
        for d, mean in drive:
            speedup = base / mean if mean else float("nan")
            print(f"  {mean:>9.4f}s   {speedup:>5.2f}x   {d.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
