# Alpamayo 1.5 on curated_val with navigation text: smoke test, then full run

Written 2026-10-23 on ip-10-225-150-13, which could not run it (another user's jobs fill all 8 GPUs).
Run this on a different box that mounts the same `/mnt/efs`.

## What this is

Alpamayo 1.5 (A1.5) closed-loop eval on Alpasim `nurec_curated_val` (441 scenes, 1 rollout each), now
conditioned on navigation text derived from the served route ("Turn left in 30m" / "Continue straight").
The baseline is the 2026-10-20 run without navigation text:
`/mnt/efs/users/rod/repos/alpasim/alpamayo_eval_results/2026-10-20_curated_val` (12.4 h on 8 B200s).
Everything else must stay as in that run, so the only change is the navigation text.

- Code: `/mnt/efs/users/rod/repos/alpasim-e2e`, branch `rod/e2e-train-subsets`, commit `9ec3682`
  (upstream NVlabs `route_to_nav_text` ported; logs `nav_text=...` at INFO for each inference). It is on EFS,
  so no checkout or push is needed. Do not change the code or switch branches.
- Launcher: `/mnt/efs/users/rod/repos/alpasim/launch_val_nav.sh smoke|full`. Same overrides as the
  2026-10-20 run (`launch_val.sh`). Logs: `launch_val_nav.log`, `smoke_val_nav.log`, `val_nav_run.log`
  in `/mnt/efs/users/rod/repos/alpasim/`.
- A first smoke on ip-10-225-150-13 (`smoke_val_nav_ip-10-225-150-13_1023_1726`) failed only because GPU 0
  ran out of memory while the driver loaded (another user's jobs). It says nothing about the code.

## Rules

- Other people's jobs may share the GPUs. Never kill or disturb processes you did not start.
- Do not push anything to git; do not edit the code.
- Run long commands detached (`setsid nohup ... &`) so they survive the session ending.

## 1. Check the box

    hostname; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    ls /mnt/efs/users/rod/repos/alpasim-e2e >/dev/null && echo efs-ok

- Smoke needs one GPU with about 80 GB free (renderer, driver, physics share it).
- Full run needs all 8 GPUs with about 50 GB free each (2026-10-20 peaks: 46-51 GB on renderer GPUs 0-3,
  about 28 GB on driver GPUs 4-7).

## 2. Set up (skip parts that already exist)

Venv on local NVMe (takes a few minutes; needs `uv` and `cargo`):

    cd /mnt/efs/users/rod/repos/alpasim-e2e
    export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/rod/venvs/alpasim-e2e VIRTUAL_ENV=/opt/dlami/nvme/rod/venvs/alpasim-e2e UV_CACHE_DIR=/opt/dlami/nvme/rod/cache/uv
    source setup_local_env.sh </dev/null      # ends with "Setup complete"

Hugging Face token at `/opt/dlami/nvme/rod/cache/hf/token` (the launcher reads it; the A1.5 weights,
`nvidia/Alpamayo-1.5-10B`, download into `/opt/dlami/nvme/rod/cache/hf` on first use). If the token file is
missing, stop and ask rod for it.

Scene cache: needs the 441 curated_val USDZs. The default NVMe cache is
`/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts` (2026-10-20 used it on ip-10-225-141-207).
Check, and fill it if needed:

    ls /opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts/all-usdzs 2>/dev/null | wc -l
    # if < 441 (resumable; prints "sceneset ready" when done):
    cd /mnt/efs/users/rod/repos/alpasim-e2e && source setup_local_env.sh </dev/null && \
      uv run --no-sync python /mnt/efs/users/rod/repos/alpasim/warm_cache.py

Then use `export CACHE=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts` for every launcher call below.

Docker: the launcher builds or reuses the image `alpasim-base:0.89.0` from the repo. The first build takes a while.

## 3. Smoke test (3 scenes, one GPU, about 20-30 min)

Pick the GPU with the most free memory as N:

    export CACHE=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts
    SMOKE_GPU=N /mnt/efs/users/rod/repos/alpasim/launch_val_nav.sh smoke

It prints the run folder (`smoke_val_nav_<host>_<stamp>`). The smoke **passes** only if all of these hold:

1. `<run>/aggregate/metrics_results.txt` exists, with 3 rollouts.
2. The driver received navigation text:

       grep -o "nav_text=[^ ]*.*" /mnt/efs/users/rod/repos/alpasim/smoke_val_nav.log | sort | uniq -c

   Expect many lines, mostly `'Continue straight'` and some `'Turn left/right in Nm'`. If every line is
   `nav_text=None`, no route reached the driver: that is a failure, do not launch the full run.
3. No `Traceback` from `driver-0` in `smoke_val_nav.log`.

On failure: find the first service that died (`grep -n "exited with code\|Traceback\|OutOfMemory" smoke_val_nav.log`),
report it with the log excerpt, and stop. An out-of-memory error means another job took the GPU: retry on a
freer GPU once, otherwise stop and report.

## 4. If the smoke passes: full run (441 scenes, 8 GPUs, about 12-14 h)

Check again that all 8 GPUs have about 50 GB free, then launch detached:

    export CACHE=/opt/dlami/nvme/rod/datasets/alpasim/data/nre-artifacts
    setsid nohup /mnt/efs/users/rod/repos/alpasim/launch_val_nav.sh full > /dev/null 2>&1 &

The run folder is `/mnt/efs/users/rod/repos/alpasim/eval_alpamayo1_5_curated_val_nav_<host>_<stamp>`; progress
is `ls <run>/rollouts | wc -l` (441 when done) and `tail val_nav_run.log`. `launch_val_nav.log` gets
"wizard exited rc=..." at the end. Check about every hour; on a crash, report the first error as in step 3.

## 5. When the full run finishes

Report back to rod:

- run folder, host, start/end time, `rc`;
- `<run>/aggregate/metrics_results.txt` next to the baseline
  `alpamayo_eval_results/2026-10-20_curated_val/aggregate/metrics_results.txt` (both scored with the
  lateral corridor exit fix, so they compare directly): headline score, collision / offroad / corridor rates,
  number of failed rollouts;
- counts of `nav_text` values across the run (`grep -o "nav_text=..." val_nav_run.log | sort | uniq -c`).

Do not move or rename the run folder; rod will copy it into `alpamayo_eval_results/` himself.
