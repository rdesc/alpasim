#!/usr/bin/env python
"""Render AutoVLA's per-step inference dumps into inspectable panels (and a video).

AutoVLA ships no inference-time visualisation: ``tools/visualization/
visualize_action_token.py`` is an empty file, and ``waymo_e2e_visualization.py``
draws the *ground-truth* trajectory and the *annotation* chain-of-thought from
preprocessed Waymo JSON, not what the model itself saw and said.

This renders, per step: the three camera views the policy actually consumed
(rectified, after resize), a bird's-eye view of its decoded 5 s plan against the
served route, its conditioning (command / speed / acceleration), and its
reasoning text — with the thinking and the action tokens split apart so the
adaptive-reasoning behaviour is visible at a glance.

Point it at a directory written by the driver with ``AUTOVLA_DEBUG_DIR`` set:

    python visualize_autovla_steps.py RUN_DIR/debug -o out/ --video out/trace.mp4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from PIL import Image  # noqa: E402

CAMERA_PANELS = [
    ("front_left_camera", "front-left (nuPlan L0)"),
    ("front_camera", "front (nuPlan F0)"),
    ("front_right_camera", "front-right (nuPlan R0)"),
]
ACTION_TOKEN_RE = re.compile(r"<action_\d+>")


def split_reasoning(text: str) -> tuple[str, str]:
    """Separate the chain-of-thought from the emitted action tokens.

    AutoVLA answers as ``<think>…</think><answer>… <action_N>… </answer>``, and
    switches between fast (no reasoning) and slow (reasoning) modes on its own,
    so it is worth seeing which mode produced a given step.
    """
    think = ""
    match = re.search(r"<think>(.*?)</think>", text, re.S)
    if match:
        think = match.group(1).strip()
    actions = ACTION_TOKEN_RE.findall(text)
    return think, " ".join(actions)


def draw_step(record: dict, step_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    poses = np.asarray(record["poses"], dtype=float)
    route = np.asarray(record.get("route_xy") or [], dtype=float).reshape(-1, 2)
    think, action_tokens = split_reasoning(record.get("reasoning_text", ""))

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    grid = gridspec.GridSpec(
        2, 3, height_ratios=[1.05, 1.0], hspace=0.18, wspace=0.12, figure=fig
    )

    # --- what the model saw -------------------------------------------------
    for col, (role, title) in enumerate(CAMERA_PANELS):
        ax = fig.add_subplot(grid[0, col])
        image_path = step_dir / f"{role}.jpg"
        if image_path.is_file():
            ax.imshow(Image.open(image_path))
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # --- where it decided to go --------------------------------------------
    ax_bev = fig.add_subplot(grid[1, 0])
    if len(route):
        # Rig frame is x forward / y left; plot y to the left of the page.
        ax_bev.plot(-route[:, 1], route[:, 0], color="0.6", lw=1.2, ls="--",
                    marker=".", ms=3, label="served route")
    ax_bev.plot(-poses[:, 1], poses[:, 0], color="tab:orange", lw=2.2,
                marker="o", ms=4, label="AutoVLA plan (5 s)")
    ax_bev.scatter([0], [0], marker="s", s=70, color="tab:green", zorder=5, label="ego")
    ax_bev.set_xlabel("left  (m)")
    ax_bev.set_ylabel("forward  (m)")
    ax_bev.set_title("bird's-eye view, rig frame", fontsize=10)
    ax_bev.grid(alpha=0.3)
    # Fixed extents so the plan's curvature is comparable across steps.
    ax_bev.set_xlim(-32, 32)
    ax_bev.set_ylim(-5, 62)
    ax_bev.set_aspect("equal", adjustable="box")
    ax_bev.legend(fontsize=8, loc="upper left")

    # --- what it was told, and what it said ---------------------------------
    ax_text = fig.add_subplot(grid[1, 1:])
    ax_text.axis("off")

    mode = "SLOW (chain-of-thought)" if len(think) > 120 else "FAST (direct action)"
    header = (
        f"step {record['step']:>4d}    "
        f"command: {record['instruction']!r}    "
        f"speed: {record['speed_mps']:.2f} m/s    "
        f"accel: {record['acceleration_mps2']:+.2f} m/s²\n"
        f"reasoning mode: {mode}"
    )
    ax_text.text(0, 1.0, header, fontsize=11, family="monospace",
                 va="top", transform=ax_text.transAxes, weight="bold")

    body = textwrap.fill(think, width=104) if think else "(no reasoning emitted)"
    ax_text.text(0, 0.84, body, fontsize=9.5, family="monospace", va="top",
                 transform=ax_text.transAxes, wrap=True)

    ax_text.text(0, 0.16, "action tokens:", fontsize=10, family="monospace",
                 va="top", transform=ax_text.transAxes, weight="bold")
    ax_text.text(0, 0.10, textwrap.fill(action_tokens, width=104), fontsize=8.5,
                 family="monospace", va="top", transform=ax_text.transAxes,
                 color="tab:orange")

    fig.savefig(out_path, dpi=96, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("debug_dir", type=pathlib.Path,
                        help="directory written via AUTOVLA_DEBUG_DIR")
    parser.add_argument("-o", "--out", type=pathlib.Path, required=True,
                        help="where to write the per-step panels")
    parser.add_argument("--video", type=pathlib.Path,
                        help="also stitch the panels into this mp4 (needs ffmpeg)")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="render only the first N steps (0 = all)")
    args = parser.parse_args()

    step_dirs = sorted(args.debug_dir.glob("step_*"))
    if args.limit:
        step_dirs = step_dirs[: args.limit]
    if not step_dirs:
        print(f"no step_* directories under {args.debug_dir}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for step_dir in step_dirs:
        record_path = step_dir / "step.json"
        if not record_path.is_file():
            continue
        record = json.loads(record_path.read_text())
        draw_step(record, step_dir, args.out / f"{step_dir.name}.png")
        rendered += 1
    print(f"rendered {rendered} panels -> {args.out}")

    if args.video and rendered:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(args.fps),
             "-pattern_type", "glob", "-i", str(args.out / "step_*.png"),
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(args.video)],
            check=True,
        )
        print(f"video -> {args.video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
