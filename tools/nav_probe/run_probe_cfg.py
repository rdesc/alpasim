"""Can Alpamayo 2 Super run navigation classifier-free guidance on ONE A100, and does it then follow the instruction?

Same saved driver steps as run_probe.py; inputs built exactly as Alpamayo2Model.predict builds them (frames, camera
indices, ego history), then NVIDIA's own guided sampler from alpamayo2's examples/two_gpu_nav_cfg_demo.py
(prepare_nav_model_inputs + sample_with_nav_cfg) with the VLM and the expert on the same GPU (the demo's own
argument check refuses that; the sampler itself only moves caches to the expert's device). Guidance weight: the
checkpoint's (inference_guidance_weight). Records peak GPU memory and time per prediction.

usage: python run_probe_cfg.py <cases_dir> <out.json> <alpamayo2_examples_dir> [seeds]
"""
import glob
import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from torchvision.io import decode_jpeg

from alpamayo2_super import helper
from alpasim_driver.models import alpamayo2_model as A2
from alpasim_driver.models.alpamayo_base import build_ego_history
from alpasim_grpc.v0.common_pb2 import PoseAtTime

CAMS = ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov", "camera_front_tele_30fov"]


def main(cases_dir, out_path, examples_dir, seeds=3):
    spec = importlib.util.spec_from_file_location("nav_cfg_demo", os.path.join(examples_dir, "two_gpu_nav_cfg_demo.py"))
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    device = torch.device("cuda")
    wrapper = A2.Alpamayo2Model(checkpoint_path="nvidia/Alpamayo2-Super", device=device, camera_ids=CAMS,
                                context_length=4)
    model = wrapper._model
    print("guidance weight", model.expert.diffusion.inference_guidance_weight, "| memory after load",
          round(torch.cuda.memory_allocated() / 2**30, 1), "GiB", flush=True)
    results = []
    for path in sorted(glob.glob(os.path.join(cases_dir, "*.pkl"))):
        case = pickle.load(open(path, "rb"))
        images = {c: [(ts, decode_jpeg(torch.frombuffer(bytearray(b), dtype=torch.uint8), device=device).permute(1, 2, 0))
                      for ts, b in case["frames"][c]] for c in CAMS}
        poses = [PoseAtTime.FromString(p) for p in case["poses"]]
        t0 = int(poses[-1].timestamp_us)
        xyz, rot, _ = build_ego_history(poses, t0, wrapper.NUM_HISTORY_STEPS, wrapper.HISTORY_TIME_STEP)
        data = {"image_frames": wrapper._preprocess_images(images), "camera_indices": wrapper._camera_indices(),
                "ego_history_xyz": xyz, "ego_history_rot": rot}
        d = max(5, int(case["satnav"]["distance_m"]))
        for cond, text in {"straight": "Continue straight", "left": f"Turn left in {d}m",
                           "right": f"Turn right in {d}m"}.items():
            inputs = helper.to_device(demo.prepare_nav_model_inputs(data, model, text), device)
            n_prompt = int(inputs["tokenized_data"]["input_ids"].shape[1])
            for seed in range(seeds):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
                t = time.time()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, _rot, _lp, extra, w = demo.sample_with_nav_cfg(model, inputs)
                torch.cuda.synchronize()
                xy = pred_xyz[0, 0, 0, :, :2].float().cpu().numpy()
                head = float(np.degrees(np.arctan2(*(xy[-1] - xy[-6])[::-1])))
                r = {"case": os.path.basename(path)[:-4], "satnav": case["satnav"]["text"], "cond": cond,
                     "nav_text": text, "seed": seed, "xy_3s": xy[29].tolist(), "xy_end": xy[-1].tolist(),
                     "heading_end_deg": head, "reasoning": str(demo._first_cot(extra)), "guidance_weight": w,
                     "prompt_tokens": n_prompt, "secs": round(time.time() - t, 2),
                     "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                     "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
                results.append(r)
                print(f"{r['case']:24s} {cond:8s} s{seed} end=({r['xy_end'][0]:6.1f},{r['xy_end'][1]:6.1f}) "
                      f"head {head:6.1f} {r['secs']:5.1f}s peak {r['peak_reserved_gib']:5.1f}GiB  {r['reasoning'][:60]}",
                      flush=True)
            json.dump(results, open(out_path, "w"), indent=1)
    print("done", len(results))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 3)
