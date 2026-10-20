"""Pre-download the nurec_train_mix200 USDZ artifacts into the NVMe scene cache.

Uses the wizard's own USDZManager so the files land exactly where a later
`alpasim_wizard` run expects them, and that run finds the cache warm.
"""

import logging
import sys

from alpasim_wizard.schema import ScenesConfig
from alpasim_wizard.scenes.sceneset import USDZManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO = "/mnt/efs/users/rod/repos/alpasim-e2e"
CACHE = "/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts"
SUITE = "nurec_train_mix200"

cfg = ScenesConfig(
    test_suite_id=SUITE,
    scene_cache=CACHE,
    scenes_csv=[f"{REPO}/data/scenes/sim_scenes.csv", f"{REPO}/data/scenes/sim_scenes_2604.csv"],
    suites_csv=[f"{REPO}/data/scenes/sim_suites.csv", f"{REPO}/data/scenes/sim_suites_curated.csv",
                f"{REPO}/data/scenes/sim_suites_train_mix200.csv"],
)

manager = USDZManager.from_cfg(cfg)
scenes = manager.query_by_suite_id(SUITE)
print(f"{SUITE}: {len(scenes)} scenes", flush=True)
if len(scenes) != 200:
    sys.exit(f"expected 200 scenes, got {len(scenes)}")

sceneset_dir = manager.create_sceneset_directory([s.uuid for s in scenes])
print(f"sceneset ready: {sceneset_dir}", flush=True)
