# Alpamayo 1.5 closed-loop eval on `nurec_curated_val` — run notes

Started 2026-10-20, host `ip-10-225-141-207` (8x A100-80GB).

## Goal

First Alpamayo 1.5 numbers on the 441-scene curated validation split. No such
numbers exist publicly: the model card's 1.37 +/- 0.10 is on `public_2601`
(913 scenes, 3 rollouts), and the only Alpamayo subject in the challenge
reference bundle is **Alpamayo 1**, not 1.5.

This run is **1 rollout per scene**, so it is not directly comparable to the
reference bundle (all 8 reference subjects are 441 x 3). It is a first look.

## Where things are

| what | path |
|---|---|
| checkout (detached at `2023534`, `origin/e2e_challenge`) | `/mnt/efs/users/rod/repos/alpasim-e2e` |
| venv | `/opt/dlami/nvme/rod/venvs/alpasim-e2e` |
| scene cache (441 USDZ, 753 GB) | `/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts` |
| launcher | `/mnt/efs/users/rod/repos/alpasim/launch_val.sh` |
| scene pre-download helper | `/mnt/efs/users/rod/repos/alpasim/warm_cache.py` |
| launcher log | `/mnt/efs/users/rod/repos/alpasim/launch_val.log` |
| eval log | `/mnt/efs/users/rod/repos/alpasim/val_run.log` |
| results | `/mnt/efs/users/rod/repos/alpasim/alpamayo_eval_results/2026-10-20_curated_val/aggregate/` |

Scene cache and venv are on **ephemeral NVMe** — intentional, both are caches.
All results and logs go to EFS.

## The launcher

`launch_val.sh` runs detached (`setsid`, reparented to init) so it survives
chat/terminal restarts. Pipeline:

1. wait for any running `warm_cache.py`, then ensure 441 artifacts are cached
2. wait for any running `alpasim_wizard`, then require a **passing smoke test**
   (3 scenes, 1 GPU) — aborts if the smoke produces no `metrics_results.txt`
3. wait for all 8 GPUs below 3 GiB used (up to 6 h)
4. launch the full run on `topology=8gpu_12rollouts`

## Config

```
deploy=local topology=8gpu_12rollouts driver=alpamayo1_5
+nurec_scenes=curated_val
runtime.simulation_config.n_rollouts=1
runtime.simulation_config.route_start_offset_m=40.0
eval.allow_aggregation_with_failed_rollouts=true
eval.parse_unstructured_debug_info=false
+runtime.endpoints.startup_timeout_s=900
defines.filesystem=/mnt/efs/users/rod/repos/alpasim/data
defines.hf_cache=/opt/dlami/nvme/rod/cache/hf
defines.sensordata=<scene cache>
scenes.scene_cache=<scene cache>
```

Inherited from repo defaults (not overridden): `n_sim_steps=200`,
`force_gt_duration_us=1_700_000`, `controller=nonlinear`, 4-camera set,
`max_dist_to_gt_trajectory=4.0`, `render_video=true`.

## Bugs hit and fixed

**1. Missing mount point.** `FileNotFoundError: Mount point does not exist:
.../alpasim-e2e/data/drivers`. The worktree lacks the untracked `data/drivers`
(2.9 GB VAVAM assets), `data/trafficsim-models` (68 MB CATK checkpoints) and
`data/prometheus` that the main checkout has. Fixed with
`defines.filesystem=/mnt/efs/users/rod/repos/alpasim/data`, which points
drivers/trafficsim/prometheus at the populated copy. Scenes stay on NVMe
because `sensordata`/`scene_cache` are overridden separately.

**2. Six-camera override broke the driver.** I had added
`+cameras=6cam_1080`, copied from the `e2e_challenge=dev` preset because the
PAI sensor contract delivers six cameras. The in-repo A1.5 driver rejects any
delivered camera outside `inference.use_cameras`
(`src/driver/.../main.py:787`), so all 3 smoke rollouts failed with
`Camera camera_rear_right_70fov not in desired cameras`.

Removed. `driver=alpamayo1_5` already pulls `alpamayo_configs.yaml` with
exactly the 4 cameras the model consumes — the repo was configured correctly
and the override was the error. Note `driver/alpamayo1.yaml` (the reference
subject) uses the **same** 4-camera config, so the published reference run
cannot have used 6 cameras either; the challenge README's "reproducible with
the command above" is loose on this point.

## Open question — `route_start_offset_m`

Repo default is `0.0`; `e2e_challenge/dev.yaml` sets `40.0`. Currently set to
**40.0**, per the decision to match the challenge contract for comparability.

Not verified: the reference bundle's `results-summary.json` records
`n_rollouts` but **not** `route_start_offset_m`, `n_sim_steps` or
`force_gt_duration_us`, so there is no way to confirm from the bundle what the
reference runs used.

This is worth revisiting. The camera bug showed that `e2e_challenge=dev`
settings do not all transfer to an in-repo driver — that preset is designed for
external contestant containers. `route_start_offset_m` changes where the ego
starts relative to the route, so 0.0 vs 40.0 produces genuinely different
driving and different scores. If a later run is meant to sit alongside repo
defaults rather than the challenge contract, drop this override.

`allow_aggregation_with_failed_rollouts=true` is kept deliberately but is
purely operational — it does not affect driving. It exists so a single bad
rollout cannot discard a 12-hour run, as happened to all three July
`public_2601` runs. **Check `n_clips` in `results-summary.json`** to confirm
the aggregate covers all 441 scenes.

## Expected timing

From the July `public_2601` run on this host: 36.3 scene-rollouts/hour at
`8gpu_12rollouts`. 441 rollouts therefore ~12 h once GPUs are free.

## Checking on it

```bash
tail -40 /mnt/efs/users/rod/repos/alpasim/launch_val.log   # pipeline state
tail -40 /mnt/efs/users/rod/repos/alpasim/val_run.log      # the eval itself
pgrep -af launch_val.sh                                    # still alive?
```

If the launcher aborted, the reason is the last line of `launch_val.log`.
Re-running `./launch_val.sh` is safe: the scene cache and a passing smoke are
both detected and skipped.

---

# RESULTS — val run complete (2026-10-20)

Run: `alpamayo_eval_results/2026-10-20_curated_val/`
(originally `eval_alpamayo1_5_curated_val_ip-10-225-141-207_1020_0346`)
Started 03:46, finished ~16:0x. **441/441 scenes, 0 failed rollouts**,
`n_clips: 441, n_rollouts/clip: 1`. Throughput 35.6 scenes/h (July projection
was 36.3). Output 109 GB.

Zero failures is the curated split working as intended: the 11 route-sanity
scenes that killed the July `public_2601` runs are excluded by construction, so
`allow_aggregation_with_failed_rollouts=true` never had to fire.

## Headline metrics

Values below are **after** the lateral-corridor-exit fix (#215). Pre-fix values
are in brackets where they differ; see "Scorer version" at the end.

| metric | value | (pre-fix) |
|---|---|---|
| avg_dist_between_incidents_at_fault | 0.99 | (1.03) |
| avg_dist_between_incidents | 0.71 | (0.73) |
| offroad_or_collision | 0.22 | |
| offroad_or_collision_at_fault | 0.16 | (0.15) |
| collision_any | 0.11 | |
| collision_at_fault | 0.05 | |
| offroad | 0.11 | (0.10) |
| wrong_lane | 0.28 | |
| left_corridor_laterally | **0.26** | (0.28) |
| progress | 0.77 | |
| progress_clipped_rel | 0.77 | |
| scene score (mean) | **0.5614** | (0.5455) |
| scenes passed | **257 / 441 (58.3%)** | (250 / 441) |
| driver drive RPC mean | 4.62 s (under 12-way concurrency) | |

## Drive-IRT vs the reference bundle

`val_local_evaluation/capability_ranking.csv`, algorithm `zoib`, no warnings.

| rank | subject | PCS | ±std | scene score | adbi_at_fault |
|---|---|---:|---:|---:|---:|
| 1 | **alpamayo1_5 (ours, 1 rollout)** | **2144** | 33.3 | 0.5478 | 1.031 |
| 2 | alpamayo1 (reference anchor) | 2000 | 19.8 | 0.5498 | 0.916 |
| 3 | vavam-linear | 1814 | 22.4 | 0.3144 | 0.347 |
| 4-9 | alternative_* / vavam-nonlinear | 480-1455 | | | |

**This fit uses pre-fix scores on purpose.** The bundled reference summaries
were produced before #215 and we only have their `results-summary.json`, not
their ASLs, so they cannot be rescored. Comparing our post-fix scores against
their pre-fix scores would credit us with corridor false positives removed from
our side only. Keep the comparison pre-fix, or rescore everything once upstream
publishes updated references.

A1.5 ranks 1st with interval 1-1. Note its *average scene score* is marginally
**lower** than Alpamayo 1 (0.5478 vs 0.5498) — it ranks higher because IRT
weights scenes by difficulty, i.e. A1.5 passes harder scenes. Its std is higher
(33.3 vs 19.8), consistent with 1 rollout vs the references' 3.

## Versus the published 1.37 +/- 0.10

| run | suite | scenes | rollouts | adbi_at_fault |
|---|---|---:|---:|---:|
| NVIDIA model card | public_2601 | 913 | 3 | 1.37 +/- 0.10 |
| ours (this run, post-fix) | curated_val | 441 | 1 | 0.99 |
| ours (this run, pre-fix) | curated_val | 441 | 1 | 1.03 |
| ours (July) | public_2601 | 905 | 1 | 0.93 |

Different suite and rollout count, so 1.37 is not a target this run should be
expected to hit. But both our runs sit below it.

Candidate explanation worth testing: **`route_start_offset_m=40.0`**. NVIDIA's
public_2601 number predates the challenge preset and was very likely produced
at the repo default `0.0`. A ~40-scene run at `0.0` would show whether the
offset systematically depresses the score.

The solid claim from this run is the head-to-head: same suite, same scenes,
same contract, A1.5 > Alpamayo 1.

## Train pre-screen (no new compute)

`train_hard_candidates.csv` — derived from `fulleval.log` (July A1.5 baseline
on public_2601), not from new measurement. Per-clip metrics joined to scene ids
via `Session COMPLETED` lines, intersected with `nurec_curated_train`.

- 702 of 1761 train scenes have July A1.5 results; **1059 are 26.04-only and unmeasured**
- 256 incident scenes (36.5%), 132 clean-but-wrong_lane, 314 fully clean

Caveats: raw pre-modifier metrics, 1 rollout, July config (offset 0.0, older
`alpamayo1_5.yaml`). A prior for where A1.5 struggled, not a current
measurement. Selecting train scenes because A1.5 failed them is biased by
construction — fine for debugging or a hard-case curriculum, misleading as a
train-split estimate.

Generating scripts kept alongside: `perscene.py`, `train_screen.py`,
`per_scene_metrics.csv`.

---

# RESOLVED: `route_start_offset_m` is inert for Alpamayo on `e2e_challenge`

It does **not** move where the ego starts. The ego starts at the scene start
pose and runs the full 200 steps. What it does is trim the **route guidance
sent to the driver**: `RouteGenerator.generate_route()` runs every policy step,
resamples the route from the current pose, then `_trim_route_start()` drops all
waypoints within the first `route_start_offset_m`. The driver is shown a route
beginning 40 m ahead of it, continuously.

**For Alpamayo it changes nothing**, because the route never reaches inference:

1. runtime trims + submits each step — `events/policy.py:109` `submit_route(...)`
2. driver converts it to a command — `main.py:390` `determine_command_from_route`
3. `AlpamayoBase._encode_command()` **returns `None`**
   ("Alpamayo models reason about navigation from context")

So the val numbers are **not** affected by the 40.0 override, and the offset
does not explain the gap to the reported 1.37. Keeping 40.0 for the train run
for exact parity with val.

**Caveat that matters for our own work:** this holds only because
`e2e_challenge` has no `nav_text.py` and its `navigation.py` carries only
`determine_command_from_route`. On `rod-dev` and upstream `main`, the nav-text
path turns the route into "Turn left in 30m" text that A1.5 conditions on.
There a 40 m trim would hide near-field turns and change behaviour materially.
Do not treat this parameter as free when evaluating the nav-text variant.

# `parse_unstructured_debug_info` — was off, now being recovered

Repo default is `true`; `e2e_challenge/dev.yaml` sets it `false` because the
field is pickle-encoded and unsafe for **untrusted contestant images**. Our
in-repo driver is trusted, so it should have been `true`. With it off,
`_extract_debug_extra()` returns `None` and `reasoning_text` is never
populated — no Chain-of-Cognition text in metrics or video overlays.

It is a **read-time** flag, not a storage flag: the raw bytes are in each
`rollout.asl` (223 MB/rollout, ~97% of the 109 GB). Nothing was lost.

Recovery for the val run: `aggregate/` backed up to
`aggregate_backup_norreasoning/`, `eval-config.yaml` flipped to `true` (original
at `eval-config.yaml.bak`), then `alpasim-reeval <run_dir>` re-derives metrics
and videos from the existing ASLs. CPU-only, no re-simulation.

Train run uses `parse_unstructured_debug_info=true` from the start.

## Pattern to watch

Three issues now all from the same root cause: `+e2e_challenge=dev` is built
for **external contestant containers**, and its settings do not transfer to an
in-repo driver.

| setting | dev preset | correct for in-repo A1.5 | impact |
|---|---|---|---|
| `cameras` | 6cam_1080 | driver's native 4-cam | **broke the run** (driver rejects extra cameras) |
| `parse_unstructured_debug_info` | false | true | dropped reasoning traces (recoverable) |
| `route_start_offset_m` | 40.0 | either | **none** for Alpamayo (route discarded) |

Do not copy dev-preset settings wholesale again. Check each against the
in-repo driver's actual code path.

# Train sample run

`nurec_train_sample400` — 400 of 1761 train scenes (22.7%), stratified by
release (159 from 26.01, 241 from 26.04, matching the 702/1059 split),
`random.seed(20261020)`. Unbiased estimate, comparable to the val run.

- suite CSV: `alpasim-e2e/data/scenes/sim_suites_train_sample400.csv`
- config: `alpasim-e2e/src/wizard/configs/nurec_scenes/train_sample400.yaml`
- download helper: `warm_cache_train.py`

Config identical to the val run except `+nurec_scenes=train_sample400` and
`parse_unstructured_debug_info=true`.

---

# Scorer version — lateral corridor exit fix (#215)

Upstream `e2e_challenge` landed `1b024f0` ("Bugfix: Lateral Corridor Exit",
[#215](https://github.com/NVlabs/alpasim/issues/198)) on 2026-10-20, after this
run was produced. `left_corridor_laterally` is a binary pass/fail criterion, so
the fix moves the scene score directly.

The old `_end_direction()` took the last two ground-truth coordinates. When the
recording ends with the ego stopped — a red light, the end of a clip — those are
duplicates or millimetre jitter, so the direction vector is noise. That vector
is what separates longitudinal overshoot from lateral departure, so overshoot
was being scored as a corridor exit. The fix walks backwards until it finds at
least 1 m of displacement.

Rescored from the existing ASLs with `alpasim-reeval` (scorer-only change, no
re-simulation):

| | before | after | delta |
|---|---:|---:|---:|
| scenes passed | 250 | **257** | +7 |
| corridor tripped | 122 | 115 | -7 |
| corridor-only failures | 121 | 114 | -7 |
| offroad failures | 48 | 48 | 0 |
| at-fault collision failures | 22 | 22 | 0 |
| mean scene score | 0.5455 | **0.5614** | +0.0159 |

So 7 scenes (1.6% of the suite) were false-positive corridor exits. The earlier
finding survives: corridor exit alone still accounts for 114/441 = **26%** of
scenes, and 60% of all failures.

Aggregates preserved in the run directory:
`aggregate_backup_pre_corridorfix/` (buggy scorer) and
`aggregate_backup_norreasoning/` (before reasoning traces were parsed).

**The fix's own tests cannot run on this branch.** 6 of 21 in
`test_ground_truth_corridor.py` fail at setup with `Protocol message
SessionMetadata has no "scene" field` — the new tests build a `scene` submessage
the `e2e_challenge` proto does not have. The failures are identical on a clean
upstream checkout, so this is pre-existing upstream breakage, not ours. They are
exactly the `test_stopped_endpoint_scoring_*` cases covering what the fix
addresses, so the fix is applied without executable regression coverage.

# External agreement — OPTED (arXiv:2609.20756)

That paper evaluates Alpamayo 1.5 on the same 441 `curated_val` scenes, giving
an independent check on our harness.

| metric | OPTED reported | ours (post-fix) |
|---|---:|---:|
| scene score (%) | 55.2 +/- 1.0 | 56.1 |
| km / infraction | 0.92 +/- 0.05 | 0.99 |
| collision at-fault (%) | 4.8 +/- 0.6 | 5.0 |
| off-road (%) | 12.0 +/- 1.4 | 10.9 |
| corridor (%) | 25.5 +/- 0.9 | 26.1 |
| progress (%) | 76.8 +/- 0.8 | 77.5 |

Every closed-loop metric lands within or essentially at their error bars;
`km/infraction` is the loosest at roughly 1.4 sigma. Note their numbers predate
#215, so the strictly matched comparison is against our pre-fix scene score of
54.6 rather than 56.1.

A LaTeX version of this table, structured after their Table I, is at
`alpamayo15_curated_val_table.tex`. Its ADE and latency columns are left blank
deliberately: we computed no open-loop ADE, and our 4.62 s driver RPC is
wall-clock under 12-way concurrency, not per-step single-stream latency.
