"""Fetch per-scene labels.json from the NuRec HuggingFace dataset.

AlpaSim does not read these, but each released scene ships an eight-axis
taxonomy (behavior, layout, lighting, road_types, surface_conditions,
traffic_density, vrus, weather) next to its USDZ. Joining it to eval results
is the only way to break failures down by scenario type.
"""

import concurrent.futures as cf
import csv
import json
import os
import sys
import urllib.request

REPO = "/mnt/efs/users/rod/repos/alpasim-e2e"
BASE = "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/resolve"
TOKEN = open("/opt/dlami/nvme/rod/cache/hf/token").read().strip()
OUT = sys.argv[1] if len(sys.argv) > 1 else "scene_labels.json"

# scene_id -> (hf_revision, artifact directory) from the catalogs
meta: dict[str, tuple[str, str]] = {}
for fn in ("sim_scenes.csv", "sim_scenes_2604.csv"):
    for r in csv.DictReader(open(f"{REPO}/data/scenes/{fn}")):
        meta[r["scene_id"]] = (r["hf_revision"], os.path.dirname(r["path"]))

scenes = sorted(meta)
print(f"catalog: {len(scenes)} scenes", flush=True)


def fetch(scene_id: str):
    rev, folder = meta[scene_id]
    req = urllib.request.Request(
        f"{BASE}/{rev}/{folder}/labels.json",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as fh:
            return scene_id, json.load(fh)
    except Exception as exc:
        return scene_id, {"_error": str(exc)[:80]}


labels = {}
with cf.ThreadPoolExecutor(max_workers=24) as pool:
    for i, (sid, lab) in enumerate(pool.map(fetch, scenes), 1):
        labels[sid] = lab
        if i % 250 == 0:
            print(f"  {i}/{len(scenes)}", flush=True)

ok = sum(1 for v in labels.values() if "_error" not in v)
json.dump(labels, open(OUT, "w"), indent=1, sort_keys=True)
print(f"fetched {ok}/{len(scenes)} -> {OUT}")
