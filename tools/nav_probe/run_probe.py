"""Stage 2 of the Alpamayo 2 navigation-response probe: replay each saved driver step through Alpamayo2Model.predict
(the closed-loop driver's own code path: newest 4 frames per camera decoded with nvJPEG, ego poses, route) with the
navigation instruction forced to each condition, several seeds each, and record the planned path and reasoning.

Run inside alpasim-base:0.89.0-a2 with the branch's src mounted at /repo/src, on one free GPU.
usage: python run_probe.py <cases_dir> <out.json> [seeds]
"""
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from torchvision.io import decode_jpeg

from alpasim_driver.models import alpamayo2_model as A2
from alpasim_driver.models.base import DriveCommand, PredictionInput
from alpasim_grpc.v0.common_pb2 import PoseAtTime
from alpasim_grpc.v0.egodriver_pb2 import Route

CAMS = ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov", "camera_front_tele_30fov"]


def conditions(case):
    d = max(5, int(case["satnav"]["distance_m"]))
    return {"none": None, "straight": "Continue straight", "left": f"Turn left in {d}m", "right": f"Turn right in {d}m"}


def main(cases_dir, out_path, seeds=3):
    device = torch.device("cuda")
    model = A2.Alpamayo2Model(checkpoint_path="nvidia/Alpamayo2-Super", device=device, camera_ids=CAMS,
                              context_length=4, force_determinism=True, use_nav_text=True)
    forced = {"text": None}
    A2.route_to_nav_text = lambda _route: forced["text"]  # the model's nav source, forced per condition
    results = []
    for path in sorted(glob.glob(os.path.join(cases_dir, "*.pkl"))):
        case = pickle.load(open(path, "rb"))
        images = {c: [(ts, decode_jpeg(torch.frombuffer(bytearray(b), dtype=torch.uint8), device=device).permute(1, 2, 0))
                      for ts, b in case["frames"][c]] for c in CAMS}
        poses = [PoseAtTime.FromString(p) for p in case["poses"]]
        route = Route.FromString(case["route"])
        for cond, text in conditions(case).items():
            for seed in range(seeds):
                forced["text"] = text
                t = time.time()
                pred = model.predict(PredictionInput(camera_images=images, command=DriveCommand.STRAIGHT, speed=0.0,
                                                     acceleration=0.0, ego_pose_history=poses, inference_seed=seed,
                                                     previous_plan=None, route=route))
                xy = pred.selected_positions[:, :2]
                head = np.degrees(np.arctan2(*(xy[-1] - xy[-6])[::-1]))
                results.append({"case": os.path.basename(path)[:-4], "satnav": case["satnav"]["text"], "cond": cond,
                                "nav_text": text, "seed": seed, "xy_3s": xy[29].tolist(), "xy_end": xy[-1].tolist(),
                                "heading_end_deg": float(head), "reasoning": pred.reasoning_text,
                                "secs": round(time.time() - t, 2)})
                r = results[-1]
                print(f"{r['case']:28s} {cond:8s} s{seed} end=({r['xy_end'][0]:6.1f},{r['xy_end'][1]:6.1f}) "
                      f"head {r['heading_end_deg']:6.1f}  {str(r['reasoning'])[:70]}", flush=True)
        json.dump(results, open(out_path, "w"), indent=1)
    print("done", len(results))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
