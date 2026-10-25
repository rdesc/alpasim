#!/usr/bin/env python3
"""Render a BC sample extracted by extract_bc_dataset.py as a single figure.

Each figure shows what the student sees and what it is asked to predict:
the four camera frames for one timestep, with the target trajectory projected
into them, plus a bird's-eye panel of the same target and the pose history.

Projecting the target into the images is the check that matters: the waypoints
are stored in the ego rig frame at that timestep, so if the extraction or the
frame convention were wrong, the overlay would not sit on the road.
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from alpasim_grpc.v0.sensorsim_pb2 import AvailableCamerasReturn
from eval.data import CameraCalibration, CameraProjector

# Left-to-right as they sit on the car, with the tele last.
LAYOUT = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]


def load_projectors(scene_dir: pathlib.Path) -> dict[str, CameraCalibration]:
    cams = AvailableCamerasReturn()
    cams.ParseFromString((scene_dir / "calibration.pb").read_bytes())
    return {
        c.logical_id: CameraCalibration.from_available_camera(c)
        for c in cams.available_cameras
    }


def draw(scene_dir: pathlib.Path, index: int, out_png: pathlib.Path) -> None:
    man = json.loads((scene_dir / "manifest.json").read_text())
    d = np.load(scene_dir / "samples.npz")
    step = int(d["step"][index])
    fut = d["future_xyz"][index]
    hist = d["history_xyz"][index]
    speed = float(d["speed_mps"][index])
    t_rel = (int(d["timestamp_us"][index]) - int(d["timestamp_us"][0])) / 1e6

    calib = load_projectors(scene_dir)
    cams = [c for c in LAYOUT if c in man["cameras"]]

    fig = plt.figure(figsize=(16, 7.4), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.08,
                      wspace=0.22, width_ratios=[1.0, 1.0, 1.0])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0])]
    ax_bev = fig.add_subplot(gs[1, 2])
    ax_txt = fig.add_subplot(gs[1, 1])

    # Colour waypoints by time so the overlay and the BEV panel can be matched.
    colors = plt.cm.viridis(np.linspace(0, 1, len(fut)))

    for ax, cam in zip(axes, cams):
        img_path = scene_dir / "images" / f"{step:04d}_{cam}.jpg"
        img = Image.open(io.BytesIO(img_path.read_bytes()))
        ax.imshow(img)
        proj = CameraProjector(calib[cam], actual_resolution=img.size)
        # Put the waypoints at the height of the rig origin's road contact.
        pts = fut.copy()
        px, mask = proj.project_points(pts)
        if len(px):
            ax.scatter(px[:, 0], px[:, 1], s=26, c=colors[mask],
                       edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_title(cam.replace("camera_", "").replace("_", " "), fontsize=9)
        ax.set_xlim(0, img.size[0]); ax.set_ylim(img.size[1], 0)
        ax.set_xticks([]); ax.set_yticks([])

    ax_bev.plot(hist[:, 1], hist[:, 0], "-", color="0.6", lw=2, label="history 1.5s")
    ax_bev.scatter(fut[:, 1], fut[:, 0], s=22, c=colors, zorder=3,
                   label=f"target {man['horizon_s']}s")
    ax_bev.plot(0, 0, "r*", ms=15, zorder=4, label="ego now")
    ax_bev.set_xlabel("y, left (m)"); ax_bev.set_ylabel("x, forward (m)")
    ax_bev.invert_xaxis()
    ax_bev.set_aspect("equal", adjustable="datalim")
    ax_bev.grid(alpha=0.3); ax_bev.legend(fontsize=8, loc="upper left")
    ax_bev.set_title("target in ego rig frame", fontsize=9)

    reach = float(np.linalg.norm(fut[-1, :2]))
    lateral = float(fut[-1, 1])
    ax_txt.axis("off")
    ax_txt.text(
        0.0, 0.98,
        f"scene      {man['scene_id'][:26]}\n"
        f"sample     {index + 1}/{len(d['step'])}   (frame {step}, t={t_rel:.1f}s)\n"
        f"speed      {speed:.2f} m/s  ({speed * 3.6:.1f} km/h)\n"
        f"target     {len(fut)} waypoints @ {man['waypoint_hz']:.0f} Hz "
        f"over {man['horizon_s']}s\n"
        f"reaches    {reach:.1f} m ahead, {lateral:+.2f} m lateral\n"
        f"history    {len(hist)} poses over {man['history_s']}s\n\n"
        f"per scene  {man['samples']} samples kept\n"
        f"           {man['dropped_no_history']} dropped (no pose history)\n"
        f"           {man['dropped_truncated_future']} dropped (future past clip end)",
        va="top", ha="left", family="monospace", fontsize=9,
    )

    fig.suptitle(
        "BC sample: four camera inputs with the logged-trajectory target projected in",
        fontsize=12, y=0.985,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=96, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene_dir", type=pathlib.Path)
    ap.add_argument("--index", type=int, nargs="+", default=[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.scene_dir / "figures"
    for i in args.index:
        p = out_dir / f"sample_{i:03d}.png"
        draw(args.scene_dir, i, p)
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
