# Alpasim closed-loop eval results — `public_2601`

Closed-loop evaluation of Alpamayo driver policies in Alpasim on the public
`public_2601` test suite. Three full runs, one rollout per clip, all on the same
host (`ip-10-225-141-207`, A100).

## Runs

| Label | Checkpoint | Config | Started | Clips | Run UUID |
| --- | --- | --- | --- | --- | --- |
| A1.5 baseline | `nvidia/Alpamayo-1.5-10B` | full Chain-of-Cognition | 2026-07-13 01:55 | 905 | `b02f97e4` |
| A1.5 no-CoT | `nvidia/Alpamayo-1.5-10B` | `skip_cot: true` | 2026-07-17 06:13 | 905 | `fc39bf03` |
| R1 | `nvidia/Alpamayo-R1-10B` | default | 2026-07-14 15:35 | 904 | `c56ea9a7` |

Run directories on EFS (`/mnt/efs/users/rod/repos/alpasim/`):

- `eval_alpamayo1_5_public_2601_ip-10-225-141-207_0713_0130/`
- `eval_alpamayo1_5_nocot_public_2601_ip-10-225-141-207_0717_0613/`
- `eval_alpamayo1_public_2601_ip-10-225-141-207_0714_1535/`

Each contains `aggregate/metrics_results.txt`, `metrics_results.png`,
`metrics_results.parquet`, `metrics_unprocessed.parquet`, and
`results-summary.json`. The two A1.5 runs also have `aggregate/videos/`.

## Results

Lower is better for incident metrics; higher is better for progress metrics.

| Metric | Aggregation | A1.5 baseline | A1.5 no-CoT | R1 |
| --- | --- | --- | --- | --- |
| offroad_or_collision | max | 0.23 | 0.22 | 0.26 |
| offroad_or_collision_at_fault | max | 0.15 | 0.16 | 0.16 |
| collision_any | max | 0.13 | 0.11 | 0.16 |
| collision_at_fault | max | 0.05 | 0.05 | 0.06 |
| collision_front | max | 0.03 | 0.03 | 0.04 |
| collision_lateral | max | 0.02 | 0.02 | 0.02 |
| collision_rear | max | 0.08 | 0.07 | 0.11 |
| offroad | max | 0.10 | 0.11 | 0.10 |
| wrong_lane | max | 0.25 | 0.26 | 0.24 |
| progress | last | 0.72 | 0.73 | 0.73 |
| progress_rel | min | 0.81 | 0.82 | 0.81 |
| dist_traveled_m | last | 143.05 | 144.41 | 144.52 |
| gt_dist_traveled_m | last | 200.34 | 200.34 | 200.23 |
| dist_to_gt_location | max | 18.25 | 16.83 | 19.22 |
| dist_to_gt_trajectory | max | 2.81 | 2.95 | 2.65 |
| plan_deviation | mean | 0.32 | 0.31 | 0.33 |
| duration_frac_20s | last | 0.67 | 0.66 | 0.68 |
| avg_dist_between_incidents | — | 0.61 | 0.65 | 0.55 |
| avg_dist_between_incidents_at_fault | — | 0.93 | 0.91 | 0.93 |
| min_distance_to_lane_boundary_m | min | 0.05 | 0.04 | 0.05 |
| safety_monitor_triggered | max | 0.00 | 0.00 | 0.00 |
| img_is_black | max | 0.00 | 0.00 | 0.00 |

`min_ade@{0.5,1.0,2.5,5.0}s(gt)` reported N/A in all three runs.
`min_distance_to_obstacle_m` reported `inf` in all three runs.

Metric modifiers applied identically across runs: timesteps before
`eval_relevant` removed; timesteps after an offroad-or-collision event removed;
timesteps after `dist_to_gt_trajectory >= 4 m` removed; trajectories with black
images removed (none in any run).

## Observations

**Disabling Chain-of-Cognition does not cost closed-loop quality.** Every metric
in the A1.5 no-CoT run sits within run-to-run noise of the A1.5 baseline. The
largest movements are `collision_any` (0.13 to 0.11) and `dist_to_gt_location`
(18.25 to 16.83), both in the *favourable* direction, so there is no evidence
that removing reasoning degrades driving on this suite.

**A1.5 is modestly better than R1 on collisions.** `offroad_or_collision` is
0.23 vs 0.26 and `collision_any` is 0.13 vs 0.16, driven mostly by rear
collisions (0.08 vs 0.11). At-fault rates are effectively tied across all three
runs (0.15–0.16), and progress is flat at 0.72–0.73.

**The latency case for skip_cot is weaker than expected.** Measured on a smoke
run, `skip_cot` gave roughly 1.7x on the `drive` RPC (2.69 s to 1.57 s) and
about 1.6x end-to-end, not the hoped 3–5x — the diffusion trajectory decoder is
a fixed floor that removing reasoning cannot touch. Combined with the quality
results above, skip_cot is a free but bounded win.

## Caveats

- One rollout per clip; standard deviation in the underlying reports is over
  rollouts, so per-metric confidence intervals are not available here. Small
  differences (roughly under 0.02) should be treated as noise.
- The R1 run has 904 clips rather than 905, so its numbers are not computed over
  an exactly identical clip set.
- The `skip_cot` run required a bind-mount overlay of a forked `alpamayo1_5`
  package over the driver image, because the pinned upstream build silently
  ignores the `coc_text` argument. Reproducing this run needs that overlay in
  place, or the pinned dependency bumped.
