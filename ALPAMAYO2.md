# Alpamayo 2 Super in Alpasim (branch `rod/alpamayo2`)

Closed-loop evaluation of NVIDIA's Alpamayo 2 Super (34B: 32B VLM + 2.3B diffusion action expert) on Alpasim's
Physical AI AV (NuRec) scenes, next to Alpamayo 1.5 and AgenticDriver's Claude teacher, on the same simulator and
the same scorer (Alpasim's scene score). Written 2026-11-13.

## What this branch is

`rod/e2e-main` (the simulator, controller, renderer, traffic, eval and wizard the Alpamayo 1.5 runs used) plus
**upstream NVlabs/alpasim main's driver package only** (`affc2ea`). A full merge of upstream (~37k lines, incl. a
new controller MPC speed bound) would have put A2 on a different simulator from the A1.5 and teacher runs. The
upstream driver talks to e2e-main's runtime unchanged (the only driver-side proto change is two fields it ignores).

| commit | what |
|---|---|
| `713bb7e` | upstream driver package + `driver/alpamayo2.yaml` |
| `16fe722` | navigation-text option (`nav_text`), val100 suite, `8gpu_a2super` topology, launcher |
| `3c4df01` | launcher runs on its own image `alpasim-base:0.89.0-a2` |
| `9260d47` | `4gpu_a2super` topology (GPUs 4-7), `DRIVER_CONC` |
| `b86d254` | launcher GPU check: driver GPUs empty, renderer GPUs only need room |
| `c6358a9` | teacher satnav as navigation text (`driver=alpamayo2_satnav`, `models/nav_satnav.py`) |
| `eae1a88` | upstream route by default (offset 0); `tools/nav_probe/` |
| `8694795` | upstream `driver/alpamayo1_5.yaml` (e2e-main's used a key the upstream driver renamed) |
| `e7cc087` | `IMAGE_DECODE=cpu\|cuda` |
| `6e756fd` | this file |
| (next) | `IMAGE_DECODE` defaults to `cpu`; A1.5 val100 / dev100 results on this stack |

## Where everything lives

| what | where |
|---|---|
| weights (67 GB, snapshot `00554695`) | run copy `/opt/dlami/nvme/rod/cache/hf/hub/models--nvidia--Alpamayo2-Super/` (NVMe, ephemeral); original `/mnt/efs/users/rod/hf_cache/hub/models--nvidia--Alpamayo2-Super/` |
| model code `alpamayo2_super` | in the image from `NVlabs/alpamayo2` `5977692`; readable copy `/opt/dlami/nvme/rod/venvs/alpasim-a2/lib/python3.12/site-packages/alpamayo2_super/`; rod's fork `/mnt/efs/users/rod/repos/alpamayo2` |
| this worktree | `/mnt/efs/users/rod/repos/alpasim-a2` (branch `rod/alpamayo2`, pushed to `github.com/rdesc/alpasim`) |
| Docker image | `alpasim-base:0.89.0-a2` (local only; rebuild: `docker build -t alpasim-base:0.89.0-a2 --secret id=netrc,src=$HOME/.netrc .` here). `alpasim-base:0.89.0` stays e2e-main's |
| host venv (tests, probes' stage 1) | `/opt/dlami/nvme/rod/venvs/alpasim-a2` (rebuild: `source setup_local_env.sh` with `UV_PROJECT_ENVIRONMENT` set to it) |
| results | `/mnt/efs/users/rod/repos/alpasim/alpamayo_eval_results/` |
| saved-results links | `/mnt/efs/users/rod/agentic_driver_save_good_results/` |

### Inference code path

1. `src/driver/src/alpasim_driver/main.py`: gRPC server. `submit_image_observation` decodes each camera JPEG
   (`image_decode_device`), `drive` takes a plan request, `_run_batch` builds `PredictionInput` and calls
   `predict_batch`.
2. `src/driver/src/alpasim_driver/models/alpamayo2_model.py`: `predict` (newest 4 frames per camera, ego history,
   navigation text, tokenise, sample, select, reasoning), `_prepare_model_inputs_with_nav` (prompt with the
   `nav_instruction` slot), `_sample_trajectory_candidates` -> `Alpamayo2Super.sample_trajectories_from_data`.
   Navigation text: `models/nav_text.py` (upstream's `route_to_nav_text`) or `models/nav_satnav.py`.
3. `alpamayo2_super`: `helper.prepare_model_inputs`, `models/alpamayo2_super.py` `sample_trajectories_from_data`
   / `_generate_with_shared_prefill` (CoC generation, then the diffusion expert).

## Running

```bash
cd /mnt/efs/users/rod/repos/alpasim-a2
DRIVER=alpamayo2_nav SUITE=val100 DRIVER_CONC=1 IMAGE_DECODE=cuda ./launch_alpamayo2.sh  # what produced the A2 val100 result
DRIVER=alpamayo1_5  SUITE=val100 ./launch_alpamayo2.sh                                    # A1.5 on the same stack (3 per driver, CPU decode)
RUN_DIR=<existing run dir> ... ./launch_alpamayo2.sh                                   # resume: skips scenes with _complete
```

| setting | values (default first) | notes |
|---|---|---|
| `SUITE` | `dev100`, `val100` | AgenticDriver's split: dev100 = 100 curated_train scenes, val100 = 100 of the 441 curated_val |
| `DRIVER` | `alpamayo2` (no navigation, as upstream ships it), `alpamayo2_nav` (upstream `route_to_nav_text`), `alpamayo2_satnav` (teacher satnav), `alpamayo1_5` | |
| `DRIVER_CONC` | `3` | scenes each driver serves at once. **Use 1 with `IMAGE_DECODE=cuda`** (see the crash below) |
| `IMAGE_DECODE` | `cpu`, `cuda` | `cpu` since 2026-11-14 (A1.5 val100 + dev100, 3 per driver: 200 scenes, ~36k steps, 0 crashes); the A2 val100 run used `cuda` at 1 per driver |
| `TOPOLOGY` | `4gpu_a2super` (drivers GPUs 6-7, renderer + physics 4-5), `8gpu_a2super` (drivers 4-7, renderer 0-3) | a 34B driver needs a whole A100 (~70-76 GB) |
| route | `MAP`, offset 0 (upstream's default); `RECORDED`, offset 0 for `alpamayo2_satnav` | `ROUTE_OFFSET_OVERRIDE=40.0` reproduces the pre-2026-11-13 runs |
| `SMOKE=<n>` | | only the first n scenes of the suite |

Never edit `launch_alpamayo2.sh` while a launch is running: bash reads the script as it goes.

Settings in every run: 4 cameras (cross left, front wide, cross right, front tele) at 10 Hz, 4 context frames,
reasoning (CoC) on, 1 trajectory sample, no guidance, no trajectory optimiser, 1 rollout per scene, autoresume.
Start-up often loses 1-10 rollouts to the renderer's warm-up (`connection attempt timed out before receiving SETTINGS
frame`): re-run the same `RUN_DIR`.

## Findings (2026-11-12/13)

**1. The navigation text was almost always "Continue straight" (40 m route offset).** Upstream's
`route_to_nav_text` looks for turns 5-40 m ahead on the served route. Our A1.5 launchers (from `5c62d06`,
2026-10-20, copying the E2E-challenge configs) served the route from 40 m ahead (`route_start_offset_m 40`), so no
route point fell in the window: A1.5 dev100 got a turn instruction on 104 of 18,072 steps. Upstream's default is 0
(the route from the car); now the launcher default. With it, the A2 val100 run got 1,135 "Turn left", 1,580
"Turn right" and 15,378 "Continue straight" steps. Fixing it moved A1.5 curated_val only 0.561 -> 0.572.

**2. Alpamayo 2 barely follows the navigation instruction, with or without guidance.**
`tools/nav_probe/` replays recorded driver steps through `predict` with the instruction forced (5 turn/bear
steps x "straight"/"left"/"right"/none x 3 seeds; results in
`alpamayo_eval_results/2026-11-13_a2super_nav_probe/`): the instruction moves the planned end point less than the
random seed does; "Turn left" vs "Turn right" ends 0.1-0.3 m apart sideways on 4 of 5 steps; the reasoning text is
unchanged in 34 of 45. NVIDIA's guided sampler (`two_gpu_nav_cfg_demo.py`, weight 3.0) runs on ONE A100 (69.5 GiB
peak, 5.1 s/step vs 4.8 s; 4-camera prompt ~3,085 tokens) but does not change that (left-right 0.3-6 m, one case the
wrong way). So the navigation source matters little for A2.

**3. NaN crash: GPU JPEG decoding with several scenes per driver.** `Assertion 'probability tensor contains either
inf, nan or element < 0' failed` in CoC sampling kills the driver's CUDA context (every later call fails with CUDA
error 710). Seen with `image_decode_device: cuda` (upstream's default) and 3 scenes per driver: A2 lost 3 of 4 drivers
within ~1 h; A1.5 asserted after 7 steps. Not seen with 1 scene per driver (A2 val100: 100 scenes, 0 crashes) or
with CPU decoding (A1.5, 3 per driver: val100 + dev100, 200 scenes, ~36k steps, 0 crashes; the old e2e-main A1.5 runs
decoded on the CPU, ~540 scenes, 0 crashes). Likely mechanism (not traced): nvJPEG decodes one scene's frames in a gRPC thread while the model
runs another scene's inference on the same GPU, and the caching allocator reuses memory across streams. Possibly an
upstream bug worth reporting.

**4. Environment checks.** Attention: A2 picks SDPA everywhere even with flash-attn installed (NVIDIA's A2 eval
ignores `--attn_implementation`), so the Alpasim image (no flash-attn) matches the reference setup. A flash-attn image
`alpasim-base:0.89.0-a2fa` exists but is unused. Inputs: no NaN poses, empty routes or bad images in 100 val100
recordings. Memory: 72.9-76 GB of 85.9 GB per driver GPU, flat over the run. Speed: ~4.4 s/step (3 per driver),
~5.7 s/step (1 per driver).

## Results (val100, Alpasim scene score, 1 rollout per scene, same 100 scenes)

Run folders under `alpamayo_eval_results/` are in `/mnt/efs/users/rod/repos/alpasim/alpamayo_eval_results/`.

| client | run | mean | failed / partial / full | at-fault coll. | off-road | corridor |
|---|---|---|---|---|---|---|
| Claude teacher `gt_rq25_3db3` (Opus) | `/mnt/efs/data/AgenticDriver/agentic_driver_data/trials/rod-frames/eval/val100/claude_gt_rq25_3db3_us.anthropic.claude-opus-5-5/` | **0.671** +- 0.043 | 25 / 25 / 50 | 10 | 4 | 11 |
| **Alpamayo 2 Super**, `alpamayo2_nav`, route from the car | `alpamayo_eval_results/2026-11-13_a2super_navupstream_4gpu_conc1_val100/` | **0.590** +- 0.048 | 38 / 13 / 49 | 7 | 13 | 18 |
| Alpamayo 1.5, route from the car (this branch, CPU decode, 3 per driver) | `alpamayo_eval_results/2026-11-13_a15_navupstream_cpudecode_4gpu_val100/` | **0.565** +- 0.049 | 42 / 7 / 51 | 8 | 10 | 24 |
| Alpamayo 1.5, route from the car (e2e-main) | `/mnt/efs/users/rod/repos/alpasim/eval_alpamayo1_5_curated_val_nav_off0_ip-10-225-141-207_1023_2117/` (441 scenes; 0.572 on all) | **0.567** +- 0.048 | 40 / 11 / 49 | 9 | 9 | 22 |
| Alpamayo 1.5, route from 40 m (e2e-main) | `alpamayo_eval_results/2026-10-20_curated_val/` (441; 0.561 on all) | 0.552 +- 0.049 | 43 / 4 / 53 | 8 | 10 | 25 |

Paired on the same scenes: teacher - A2 = +0.082 +- 0.057 (teacher better 31, A2 better 26, tie 43);
A2 - A1.5 (route from the car, e2e-main) = +0.022 +- 0.044 (14 / 18 / 68); A2 - A1.5 (this branch) = +0.025 +- 0.047
(16 / 17 / 67). A1.5 on this branch vs on e2e-main, same settings: -0.003 +- 0.041, so the branch's simulator is
e2e-main's. Run-to-run noise of one client is ~0.07, so neither gap is settled by one rollout. Per-scene scores of all three: `2026-11-13_a2super_navupstream_4gpu_conc1_val100/
val100_three_way.json`. A2 drives further (164.5 m vs the teacher's 140.6 m) and rarely collides; it fails by leaving
the road or the corridor.

NVIDIA's model-card "AlpaSim Score" (A2 Super 1.50 +- 0.13 on 913 scenes, A1.5 1.37) is **km between incidents**
(Alpamayo-R1 paper, Sec. 6.1), not the scene score. The card does not say whether it counts all incidents or only
at-fault ones; Alpasim reports both (`avg_dist_between_incidents[_at_fault]` in `aggregate/metrics_results.txt`).
Ours on val100: A2 0.57 / 0.82 km, A1.5 0.63 / 0.87 km (all / at fault). Different scenes and settings, so not
directly comparable to 1.50.

Paper-table rows (val100; score, at-fault km per incident, then % of scenes):

| client | score | km/inc. (at fault) | at-fault coll. | off-road | corridor | progress |
|---|---|---|---|---|---|---|
| Alpamayo 1.5 (this branch) | 56.5 | 0.870 | 8.0 | 10.0 | 24.0 | 77.9 |
| Alpamayo 2 Super | 59.0 | 0.823 | 7.0 | 13.0 | 19.0 | 77.9 |
| Claude teacher `gt_rq25_3db3` (Opus) | 67.1 | 1.00 | 10.0 | 4.0 | 12.0 | 73.1 |

### dev100 (100 curated_train scenes)

| client | run | mean | failed / partial / full | km/inc. all / at fault | at-fault coll. | off-road | corridor | progress |
|---|---|---|---|---|---|---|---|---|
| Alpamayo 1.5, route from the car (this branch) | `alpamayo_eval_results/2026-11-14_a15_navupstream_cpudecode_4gpu_dev100/` | **0.625** +- 0.048 | 36 / 5 / 59 | 0.64 / 0.840 | 2 | 14 | 20 | 77.5 |
| Alpamayo 1.5, route from 40 m (e2e-main) | `alpamayo_eval_results/2026-11-06_dev100/` | 0.586 +- 0.049 | 40 / 6 / 54 | 0.73 / 0.871 | 5 | 11 | 24 | 76.0 |
| Claude teacher `gt_rq25_3db3` (Opus, 2 rollouts) | rod-frames trial | 0.715 | | | | | | |

Route from the car vs from 40 m, paired: +0.039 +- 0.037 (better 12, worse 9, tie 79): the same direction as on
curated_val (+0.011), within noise. dev100 took 8.6 h on GPUs 4-7 (7.7 s per driver call at 3 per driver with CPU
decode; the e2e-main run: 4.7 s).
