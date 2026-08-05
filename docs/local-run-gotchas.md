# Local run gotchas (Alpamayo 1.5, GPU pinning, HF cache)

Things that cost time running `driver=alpamayo1_5` locally and aren't in
[TUTORIAL.md](TUTORIAL.md).

```bash
export HF_HOME=/opt/dlami/nvme/<user>/cache/hf

uv run alpasim_wizard deploy=local topology=1gpu driver=alpamayo1_5 \
  '+runtime.endpoints.startup_timeout_s=900' \
  'services.renderer.gpus=[4]' 'services.driver.gpus=[4]' 'services.physics.gpus=[4]' \
  wizard.log_dir=$PWD/my_run trafficsim=catk
```

## Large drivers blow the 120s startup probe

The runtime probes each service for its version and gives up after
`startup_timeout_s`, default 120s (`src/runtime/alpasim_runtime/config.py:348`).
Alpamayo 1.5 doesn't answer in time, and the failure reads like a hang:

```
Service version probe for driver at driver-0:6005 still waiting after 110s
grpc.aio._call.AioRpcError: StatusCode.DEADLINE_EXCEEDED
```

The driver hasn't crashed — `from_pretrained(...).to(device)`
(`models/alpamayo1_5_model.py:76-80`) moves ~21 GB onto the GPU before the gRPC
server binds. Raise the deadline with `+runtime.endpoints.startup_timeout_s=900`
(the `+` is required; the key isn't in `base_config.yaml`).

`TUTORIAL.md:318` and `VIDEO_MODEL.md:99` blame startup timeouts on checkpoint
*downloads* and suggest re-running. **With a warm cache the load cost is paid on
every run and re-running changes nothing.** Check whether `Fetching N files` in
the driver log is instant (cache hit) or slow (download).

## A symlink inside the HF bind mount breaks the driver

The driver mounts `${defines.hf_cache}:/root/.cache/huggingface`
(`base_config.yaml:165`), defaulting to `$HF_HOME` or `~/.cache/huggingface`.

Docker resolves symlinks in the mount *source*, so a symlinked
`~/.cache/huggingface` is fine. A symlink **inside** the mounted tree pointing
**outside** it is not — e.g. `~/.cache/huggingface/hub -> /opt/dlami/nvme/...`.
The container resolves it against its own filesystem, where that path doesn't
exist, and the driver dies instantly:

```
os.makedirs(os.path.dirname(blob_path), exist_ok=True)
FileNotFoundError: [Errno 2] '/root/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B'
```

That traceback is diagnostic: `exist_ok=True` swallows `FileExistsError`, not
`ENOENT`, and `mkdir` only returns `ENOENT` when the parent — the dangling `hub`
— is missing. It also explains the attempted download despite a full cache.
(HF's own `snapshots/<rev>/*.safetensors -> ../../blobs/<sha>` links are relative
and stay inside the mount, so they work.)

**Fix:** point `HF_HOME` at the real directory so no link is crossed. It
relocates the whole cache root — `hub/`, `token`, `stored_tokens`, `modules/` —
so copy all of them across. Leaving `token` behind means you're authenticated on
the host and anonymous in the container, which turns the `FileNotFoundError` into
a 401 on the gated Alpamayo / Cosmos-Reason2 repos.

Then delete the old symlink. Afterwards `HF_HOME` is load-bearing: any process
that doesn't see it silently re-downloads into `$HOME`.

Note `/opt/dlami/nvme` is the ephemeral instance store — it doesn't survive a
stop/start, so the cache is re-fetchable data only.

## `gpus` is a fan-out knob, not a device selector

`services.<name>.gpus` is a list whose **length is the container count**. The
wizard cycles one container per entry (`services.py:352`), each loading its own
full copy of the model. There's no `device_map` or tensor parallelism anywhere in
`src/driver/src/alpasim_driver/models/`.

- `gpus=[3]` — one container on GPU 3. This is how you select a device.
- `gpus=[2,3]` — *two* containers, one per GPU. Not one container seeing both.
- More GPUs never makes a single model load faster or fit in less VRAM per card.

**Don't reach for `topology=2gpu` to fit a big model.** It sets the driver to
`replicas_per_container: 3` on `gpus: [0]` — three copies on one card, ~120 GB
for a 10B driver on an 80 GB A100. It's tuned for vavam. To avoid re-paying the
load cost each run, use `topology=daemon`, not more GPUs.

Also: `1gpu.yaml` sets `trafficsim.skip: true`, so
`services.trafficsim.gpus=[N]` does nothing unless you also pass
`trafficsim=catk`. The wizard's `Built N simulation containers` line confirms it
(4 without, 5 with).
