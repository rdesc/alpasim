# AutoVLA driver plugin for AlpaSim

Runs [AutoVLA](https://github.com/ucla-mobility/AutoVLA) (NeurIPS 2025) — Qwen2.5-VL-3B
plus discrete action tokens — as an AlpaSim ego driver.

## Layout

| file | what |
|---|---|
| `alpasim_autovla/autovla_model.py` | `BaseTrajectoryModel` adapter, registered as the `autovla` entry point |
| `alpasim_autovla/inference.py` | AutoVLA inference without pytorch-lightning/nuplan/navsim |
| `alpasim_autovla/action_tokenizer.py` | vendored from AutoVLA `models/action_tokenizer.py` |
| `alpasim_autovla/data/agent_vocab.pkl` | vendored action codebook |
| `alpasim_autovla/configs/driver/autovla{,_10hz}.yaml` | wizard driver configs (2 Hz / 10 Hz control) |
| `configs/autovla_standalone{,_10hz}.yaml` | standalone configs for running the driver outside Docker |

Launchers live in the repo root: `launch_autovla_driver.sh`, `launch_autovla_sim.sh`
(one scene), `launch_autovla_sweep.sh` (a test suite, sharded across GPUs).

## Running it

The driver runs as an **external host process** rather than in the `alpasim-base`
image, so the image needs no rebuild to carry this plugin:

```bash
# terminal 1 — the policy
RUN_DIR=/path/out GPU=7 ./launch_autovla_driver.sh

# terminal 2 — the simulator
RUN_DIR=/path/out RENDER_GPU=6 ./launch_autovla_sim.sh
```

`driver_source=external_static` plus `wizard.debug_flags.use_localhost=true`
(which adds `network_mode: host`) is what lets the containerised runtime reach it.

For a whole suite:

```bash
SWEEP=/path/out LANE_GPUS=0,1,2,3 ./launch_autovla_sweep.sh
```

Each lane is one GPU with its own driver process and its own wizard over a disjoint
slice of the scenes — AutoVLA has no batched inference path, so one driver per GPU
beats one driver serving many concurrent rollouts.

## Seeing what the model did

AutoVLA ships no inference-time visualisation — `tools/visualization/
visualize_action_token.py` is an empty file, and `waymo_e2e_visualization.py` draws
the *ground-truth* trajectory and the *annotation* CoT from preprocessed Waymo JSON,
not the model's own output. So:

```bash
# dump every inference (3 camera views + step.json) while driving
RUN_DIR=/path/out AUTOVLA_DEBUG_DIR=/path/out/debug ./launch_autovla_driver.sh

# render panels + an mp4 of the trace
python plugins/autovla_driver/tools/visualize_autovla_steps.py \
    /path/out/debug -o /path/out/panels --video /path/out/trace.mp4
```

Each panel shows the three rectified camera views the policy consumed, a BEV of its
decoded 5 s plan against the served route, the conditioning (command / speed /
acceleration), and its reasoning text with the chain-of-thought split from the action
tokens. Leave `AUTOVLA_DEBUG_DIR` unset for sweeps — it writes 3 JPEGs per step.

Note: on every scene tried so far the released RFT checkpoint stays in **fast mode**,
emitting the same "This is a straightforward scenario..." string each step, so the
adaptive-reasoning behaviour is not observable from this checkpoint alone.

## Things that are easy to get wrong

**Checkpoint key layout.** The released `AutoVLA_PDMS_89.ckpt` was written under
transformers 4.49 (`visual.*`, `model.*`); 4.52+ nests these under `model.visual.*`
and `model.language_model.*`. `from_pretrained` applies
`_checkpoint_conversion_mapping` for this, `load_state_dict` does not. `inference.py`
applies it by hand and **raises** on any missing/unexpected key — without that the
policy silently drives on base Qwen weights and looks merely "bad" rather than broken.

**Frame rate vs control rate.** AutoVLA's prompt states its four frames per camera are
"sampled at 2 Hz", so the context spacing must stay 0.5 s. `autovla_10hz` therefore
runs the cameras at 10 Hz and sets `subsample_factor: 5`: the policy replans every
0.1 s but still sees frames at t, t−0.5, t−1.0, t−1.5.

**Camera roles.** AutoVLA was trained on nuPlan CAM_F0/L0/R0. The plugin maps
`camera_front_wide_120fov`/`camera_cross_left_120fov`/`camera_cross_right_120fov` onto
those and reuses the Transfuser plugin's nuPlan pinhole intrinsics and extrinsics,
since LTFv6 targets the same rig. The rendered f-theta frames are rectified to that
pinhole model before the model sees them.

**Frames are plain tuples.** `alpasim_driver.main` builds `camera_images` as
`(timestamp_us, image)` tuples even though the type is annotated `CameraFrame`;
unpack positionally, do not use `.image`.

**`wizard.external_services` defaults to `None`**, and OmegaConf cannot merge a CLI
override into `None`. The `*_configs.yaml` files materialise it as `{}` so
`+wizard.external_services.driver=[localhost:6900]` works.

**Suite CSVs are CRLF.** A trailing `\r` inside a scene id makes Hydra's override
grammar reject `scenes.scene_ids=[...]`; the sweep script strips it.

## Setup

```bash
export VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e
uv pip install --no-deps -e plugins/autovla_driver
uv pip install --no-deps qwen-vl-utils==0.0.10
```

`--no-deps` keeps it from perturbing the solved workspace environment; `qwen-vl-utils`
is pure Python and has no top-level `av` import, so the list-of-PIL-images path it is
used for pulls nothing else in. Verify with `alpasim-info` — `autovla` should appear
under both `alpasim.models` and `alpasim.configs`.

Weights (not in this repo):

- `AutoVLA_PDMS_89.ckpt` from <https://huggingface.co/Zewei-Zhou/AutoVLA> (LoRA already
  merged, so no peft needed) → `/mnt/efs/users/rod/ckpts/AutoVLA/`
- `Qwen/Qwen2.5-VL-3B-Instruct` → `/opt/dlami/nvme/rod/cache/hf-models/`

Both paths are set in `configs/autovla_standalone*.yaml`.

## Cost

~2.2 s per inference on an A100 (3B model, autoregressive, ~10 GiB). At 10 Hz control
that is roughly 0.03x real time, about 10 min per 20 s scene including rendering.
