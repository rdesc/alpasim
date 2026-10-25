#!/usr/bin/env python3
"""Extract a behaviour-cloning dataset from force-GT AlpaSim rollout logs.

Each ASL from a ground-truth replay holds the rendered camera frames along the
logged human trajectory (as ``driver_camera_image``) and that trajectory itself
(as ``rollout_metadata.ego_rig_recorded_ground_truth_trajectory``). This turns
the pair into per-timestep training samples:

    inputs  : the JPEG frame from each camera at time t
    target  : the next ``--horizon`` seconds of the logged trajectory, expressed
              in the ego rig frame at time t
    context : the preceding ``--history`` seconds of the same trajectory

The stored trajectory is a single polyline in one fixed frame, not a future per
timestep, so each target is produced by re-expressing it relative to the ego
pose at t. That pose is exact: force-GT replay puts the ego on the logged path
to within 0.00 m, so the ego pose at t *is* the logged pose at t.

Two groups of timesteps cannot yield a sample and are dropped:
  * the first ``--history`` seconds, where no pose history exists yet;
  * the last ``--horizon`` seconds, where the logged clip ends before the
    target does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

import numpy as np

from alpasim_utils.logs import async_read_pb_log


def quat_to_R(q) -> np.ndarray:
    """Rotation matrix from a gRPC quaternion (w, x, y, z)."""
    w, x, y, z = q.w, q.x, q.y, q.z
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) from a rotation matrix."""
    t = R.trace()
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ]
        )
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


class GtPath:
    """The logged trajectory, sampled at arbitrary times by interpolation."""

    def __init__(self, trajectory) -> None:
        poses = trajectory.poses
        if len(poses) < 2:
            raise ValueError("recorded ground-truth trajectory has <2 poses")
        self.t = np.array([p.timestamp_us for p in poses], dtype=np.int64)
        self.xyz = np.array(
            [[p.pose.vec.x, p.pose.vec.y, p.pose.vec.z] for p in poses]
        )
        self.quat = np.array(
            [[p.pose.quat.w, p.pose.quat.x, p.pose.quat.y, p.pose.quat.z]
             for p in poses]
        )
        # Keep quaternion signs consistent so slerp never takes the long way.
        for i in range(1, len(self.quat)):
            if self.quat[i] @ self.quat[i - 1] < 0:
                self.quat[i] = -self.quat[i]

    @property
    def t_start(self) -> int:
        return int(self.t[0])

    @property
    def t_end(self) -> int:
        return int(self.t[-1])

    def at(self, t_us: float) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated (position, quaternion) at ``t_us``."""
        i = int(np.clip(np.searchsorted(self.t, t_us) - 1, 0, len(self.t) - 2))
        span = self.t[i + 1] - self.t[i]
        a = 0.0 if span == 0 else float((t_us - self.t[i]) / span)
        a = min(max(a, 0.0), 1.0)
        pos = self.xyz[i] * (1 - a) + self.xyz[i + 1] * a

        q0, q1 = self.quat[i], self.quat[i + 1]
        dot = float(np.clip(q0 @ q1, -1.0, 1.0))
        if dot > 0.9995:
            q = q0 * (1 - a) + q1 * a
        else:
            th = np.arccos(dot)
            s = np.sin(th)
            q = (np.sin((1 - a) * th) / s) * q0 + (np.sin(a * th) / s) * q1
        return pos, q / np.linalg.norm(q)

    def se3_at(self, t_us: float) -> tuple[np.ndarray, np.ndarray]:
        pos, q = self.at(t_us)

        class _Q:
            pass

        qq = _Q()
        qq.w, qq.x, qq.y, qq.z = q
        return pos, quat_to_R(qq)


async def read_rollout(asl: pathlib.Path):
    """Pull the camera frames, the logged trajectory and the calibration."""
    frames: dict[str, list[tuple[int, bytes]]] = {}
    gt = None
    cameras = None
    scene_id = None
    async for entry in async_read_pb_log(str(asl)):
        kind = entry.WhichOneof("log_entry")
        if kind == "driver_camera_image":
            img = entry.driver_camera_image.camera_image
            frames.setdefault(img.logical_id, []).append(
                (img.frame_start_us, img.image_bytes)
            )
        elif kind == "rollout_metadata":
            meta = entry.rollout_metadata
            gt = meta.ego_rig_recorded_ground_truth_trajectory
            scene_id = meta.session_metadata.scene_id
        elif kind == "available_cameras_return":
            cameras = entry.available_cameras_return
    if gt is None:
        raise ValueError(f"{asl}: no rollout_metadata")
    if not frames:
        raise ValueError(f"{asl}: no driver_camera_image entries")
    for v in frames.values():
        v.sort(key=lambda p: p[0])
    return frames, gt, cameras, scene_id


def extract(
    asl: pathlib.Path,
    out_dir: pathlib.Path,
    horizon_s: float,
    history_s: float,
    waypoint_hz: float,
    stride: int,
    write_images: bool,
) -> dict:
    frames, gt_proto, cameras, scene_id = asyncio.run(read_rollout(asl))
    gt = GtPath(gt_proto)

    # The step timebase is the camera with the most frames; every other camera
    # contributes the frame nearest that step's timestamp.
    base_cam = max(frames, key=lambda c: len(frames[c]))
    base_ts = [t for t, _ in frames[base_cam]]
    cam_ids = sorted(frames)
    cam_ts = {c: np.array([t for t, _ in frames[c]]) for c in cam_ids}

    n_fut = int(round(horizon_s * waypoint_hz))
    n_hist = int(round(history_s * waypoint_hz))
    dt_us = 1e6 / waypoint_hz

    kept, dropped_start, dropped_tail = [], 0, 0
    for step, t in enumerate(base_ts):
        if t - history_s * 1e6 < gt.t_start:
            dropped_start += 1
            continue
        if t + horizon_s * 1e6 > gt.t_end:
            dropped_tail += 1
            continue
        if step % stride:
            continue
        kept.append((step, t))

    records = {
        "step": [], "timestamp_us": [], "future_xyz": [], "future_heading": [],
        "history_xyz": [], "speed_mps": [], "frame_index": [],
    }
    for step, t in kept:
        pos0, R0 = gt.se3_at(t)
        R0T = R0.T

        fut = np.array([gt.at(t + (k + 1) * dt_us)[0] for k in range(n_fut)])
        fut_rel = (fut - pos0) @ R0
        # Heading of travel between consecutive waypoints, in the rig frame.
        prev = np.vstack([np.zeros((1, 3)), fut_rel[:-1]])
        d = fut_rel - prev
        head = np.arctan2(d[:, 1], d[:, 0])

        hist = np.array([gt.at(t - (k + 1) * dt_us)[0] for k in range(n_hist)])
        hist_rel = (hist - pos0) @ R0

        nxt, _ = gt.at(t + dt_us)
        speed = float(np.linalg.norm(nxt - pos0) / (dt_us / 1e6))

        records["step"].append(step)
        records["timestamp_us"].append(t)
        records["future_xyz"].append(fut_rel)
        records["future_heading"].append(head)
        records["history_xyz"].append(hist_rel)
        records["speed_mps"].append(speed)
        records["frame_index"].append(
            [int(np.argmin(np.abs(cam_ts[c] - t))) for c in cam_ids]
        )
        del R0T

    scene_dir = out_dir / (scene_id or asl.parent.parent.name)
    scene_dir.mkdir(parents=True, exist_ok=True)

    if write_images:
        img_dir = scene_dir / "images"
        img_dir.mkdir(exist_ok=True)
        for (step, _), idxs in zip(kept, records["frame_index"]):
            for c, fi in zip(cam_ids, idxs):
                (img_dir / f"{step:04d}_{c}.jpg").write_bytes(frames[c][fi][1])

    np.savez_compressed(
        scene_dir / "samples.npz",
        step=np.array(records["step"], dtype=np.int32),
        timestamp_us=np.array(records["timestamp_us"], dtype=np.int64),
        future_xyz=np.array(records["future_xyz"], dtype=np.float32),
        future_heading=np.array(records["future_heading"], dtype=np.float32),
        history_xyz=np.array(records["history_xyz"], dtype=np.float32),
        speed_mps=np.array(records["speed_mps"], dtype=np.float32),
    )
    if cameras is not None:
        (scene_dir / "calibration.pb").write_bytes(cameras.SerializeToString())

    manifest = {
        "scene_id": scene_id,
        "asl": str(asl),
        "cameras": cam_ids,
        "base_camera": base_cam,
        "horizon_s": horizon_s,
        "history_s": history_s,
        "waypoint_hz": waypoint_hz,
        "stride": stride,
        "n_waypoints": n_fut,
        "n_history": n_hist,
        "frames_per_camera": {c: len(frames[c]) for c in cam_ids},
        "samples": len(kept),
        "dropped_no_history": dropped_start,
        "dropped_truncated_future": dropped_tail,
        "gt_span_s": (gt.t_end - gt.t_start) / 1e6,
        "stationary_samples": int(
            sum(1 for s in records["speed_mps"] if s < 0.1)
        ),
    }
    (scene_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=pathlib.Path,
                    help="a wizard log_dir, or a single rollout.asl")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("/mnt/efs/users/rod/bc_dataset"))
    ap.add_argument("--horizon", type=float, default=4.0,
                    help="target trajectory length in seconds")
    ap.add_argument("--history", type=float, default=1.5,
                    help="pose history required before a step is usable")
    ap.add_argument("--waypoint-hz", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=5,
                    help="keep every Nth camera frame (5 = one sample per 0.5s)")
    ap.add_argument("--no-images", action="store_true",
                    help="write only trajectories, no JPEGs")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.run_dir.suffix == ".asl":
        asls = [args.run_dir]
    else:
        asls = sorted(args.run_dir.rglob("rollout.asl"))
    if args.limit:
        asls = asls[: args.limit]
    if not asls:
        print(f"no rollout.asl under {args.run_dir}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, asl in enumerate(asls, 1):
        m = extract(asl, args.out, args.horizon, args.history,
                    args.waypoint_hz, args.stride, not args.no_images)
        total += m["samples"]
        print(f"[{i}/{len(asls)}] {m['scene_id']}: {m['samples']} samples "
              f"(dropped {m['dropped_no_history']} startup, "
              f"{m['dropped_truncated_future']} tail; "
              f"{m['stationary_samples']} stationary)")
    print(f"\n{total} samples from {len(asls)} scenes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
