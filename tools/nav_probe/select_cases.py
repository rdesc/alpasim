"""Stage 1 of the Alpamayo 2 navigation-response probe: pick driver steps where the car is still on the human
path and the human's satnav (nav_satnav.instruction on the recorded drive ahead) says turn/bear 10-30 m ahead,
and save that step's exact driver inputs (newest 4 frames per camera, ego poses, route) from rollout.asl.

usage: python select_cases.py <out_dir> <run_dir> [<run_dir> ...]
"""
import asyncio
import collections
import glob
import os
import pickle
import sys

import numpy as np
from alpasim_driver.models import nav_satnav
from alpasim_utils.logs import async_read_pb_log

CAMS = ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov", "camera_front_tele_30fov"]
MAX_OFF_GT_M = 1.0       # car still on the human path
DIST_RANGE_M = (10, 30)  # manoeuvre this far ahead
PER_KIND = 3             # cases per satnav manoeuvre kind


def yaw_of(q):
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def path_ahead(gt, i):
    x, y, yaw = gt[i]
    c, s = np.cos(-yaw), np.sin(-yaw)
    rel = gt[i:, :2] - [x, y]
    p = np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], 1)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return p[arc <= 80.0]


async def scan(asl):
    """Yield (step index, satnav instr, state) at each driver_request where the car is on the human path."""
    gt = None
    frames = collections.defaultdict(list)
    poses = {}
    route = None
    step = 0
    async for e in async_read_pb_log(asl):
        k = e.WhichOneof("log_entry")
        if k == "rollout_metadata":
            gt = np.array([(p.pose.vec.x, p.pose.vec.y, yaw_of(p.pose.quat))
                           for p in e.rollout_metadata.ego_rig_recorded_ground_truth_trajectory.poses])
        elif k == "driver_camera_image":
            ci = e.driver_camera_image.camera_image
            frames[ci.logical_id].append((ci.frame_end_us, bytes(ci.image_bytes)))
            frames[ci.logical_id] = frames[ci.logical_id][-4:]
        elif k == "driver_ego_trajectory":
            for p in e.driver_ego_trajectory.trajectory.poses:
                poses[p.timestamp_us] = p.SerializeToString()
        elif k == "route_request":
            route = e.route_request.route.SerializeToString()
        elif k == "driver_request":
            step += 1
            if gt is None or route is None or not poses or any(len(frames[c]) < 4 for c in CAMS):
                continue
            t_last = max(poses)
            from alpasim_grpc.v0.common_pb2 import PoseAtTime
            last = PoseAtTime.FromString(poses[t_last]).pose.vec
            d = np.linalg.norm(gt[:, :2] - [last.x, last.y], axis=1)
            i = int(np.argmin(d))
            if d[i] > MAX_OFF_GT_M:
                continue
            instr = nav_satnav.instruction(path_ahead(gt, i))
            if instr is None:
                continue
            yield step, instr, {
                "frames": {c: list(frames[c]) for c in CAMS},
                "poses": [poses[t] for t in sorted(poses)],
                "route": route,
                "time_now_us": e.driver_request.time_now_us,
            }


async def main(out_dir, run_dirs):
    os.makedirs(out_dir, exist_ok=True)
    picked = collections.defaultdict(list)
    used_scenes = set()
    asls = sorted({a for r in run_dirs for a in glob.glob(os.path.join(r, "rollouts", "*", "*", "rollout.asl"))})
    for asl in asls:
        scene = asl.split("/rollouts/")[1].split("/")[0]
        if scene in used_scenes:
            continue
        async for step, instr, state in scan(asl):
            kind = instr["maneuver"].split()[0]  # turn / bear / straight / roundabout
            if kind not in ("turn", "bear") or not DIST_RANGE_M[0] <= instr["distance_m"] <= DIST_RANGE_M[1]:
                continue
            if len(picked[kind]) >= PER_KIND:
                break
            case = {"scene": scene, "asl": asl, "step": step, "satnav": instr, **state}
            name = f"{kind}_{len(picked[kind])}_{scene[7:15]}_s{step}"
            with open(os.path.join(out_dir, name + ".pkl"), "wb") as f:
                pickle.dump(case, f)
            picked[kind].append(name)
            used_scenes.add(scene)
            print(name, instr["text"])
            break
        if all(len(picked[k]) >= PER_KIND for k in ("turn", "bear")):
            break
    print({k: len(v) for k, v in picked.items()})


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2:]))
