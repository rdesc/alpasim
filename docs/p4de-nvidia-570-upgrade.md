# Upgrading p4de A100 instances from NVIDIA 545 → R570

Runbook for upgrading the NVIDIA driver on the `ai-foundation-p4de-*` instances
(p4de.24xlarge, 8× A100-80GB SXM with NVSwitch, Ubuntu 20.04 DLAMI) from the
stock **545 / CUDA 12.3** to **R570 / CUDA 12.8**.

> **Why R570 and not host CUDA?** Note: some apps (like Alpasim) run CUDA inside
> containers via the NVIDIA Container Toolkit, so the host only needs a
> new-enough *driver*. Do **not** install the host CUDA toolkit — it's the thing
> that caused the worst failure below. The entire R570 branch ships CUDA 12.8
> support, so any R570 release works; this runbook used **570.211.01**.

The ⚠️ callouts explain what went wrong and why each step is
ordered the way it is.

---

## 0. Pre-flight (do not skip)

```bash
# a) Snapshot the ROOT EBS volume for rollback (AWS Console → EC2 → Volumes → Create snapshot).
#    NOTE: this does NOT capture /opt/dlami/nvme (instance-store, ephemeral).

# b) *** CRITICAL *** Inspect THIS instance's /etc/fstab for nofail mounts.
#    The set of mounts differs per instance (different EBS volumes attached to each),
#    so review whatever this specific instance has — don't assume it matches
#    another instance. The one consistent trap across these DLAMI instances is the
#    device-mapper NVMe line; extra EBS volumes (e.g. /mnt/<user>_ebs) vary.
grep -nE 'nofail' /etc/fstab
```

⚠️ **Boot-hang trap.** On Ubuntu 20.04, a **device-mapper / LVM** `/etc/fstab`
line like `/dev/mapper/vg_local_nvme-lv_ephemeral /opt/dlami/nvme ext4 defaults,nofail 0 0`
will **hang the instance forever at boot** (stuck on `systemd-tmpfiles-setup`) after a reboot — for device-mapper units `nofail` alone is *not* enough. Fix it
**before** rebooting.

> **Not every `nofail` line is dangerous.** The infinite-hang is specific to
> `/dev/mapper/...` (LVM/device-mapper) entries. A plain block device by UUID
> (e.g. an EBS volume: `UUID=... /mnt/foo ext4 defaults,nofail 0 2`) handles
> `nofail` correctly — if the device is absent it waits the systemd default
> (~90s) and then continues, no hang. So focus the fix on device-mapper lines;
> adding `x-systemd.device-timeout=0` to UUID/EBS lines is optional hardening
> only. Also note: EBS volumes are persistent and survive a stop/start, but the
> instance-store NVMe under `/opt/dlami/nvme` does **not**.

Fix it one of two ways (back up first: `sudo cp /etc/fstab /etc/fstab.bak`):

**Option A — add `x-systemd.device-timeout=0` to the device-mapper line:**

```diff
# /etc/fstab
- /dev/mapper/vg_local_nvme-lv_ephemeral /opt/dlami/nvme ext4 defaults,nofail 0 0
+ /dev/mapper/vg_local_nvme-lv_ephemeral /opt/dlami/nvme ext4 defaults,nofail,x-systemd.device-timeout=0 0 0
```

**Option B — disable the fstab line entirely and let `dlami-nvme.service` mount it**
(this is what was done on `ai-foundation-p4de-3`):

```diff
# /etc/fstab
- /dev/mapper/vg_local_nvme-lv_ephemeral /opt/dlami/nvme ext4 defaults,nofail 0 0
+ # DISABLED <date> (mounted by dlami-nvme.service): /dev/mapper/vg_local_nvme-lv_ephemeral /opt/dlami/nvme ext4 defaults,nofail,x-systemd.device-timeout=0 0 0
```

Then reload systemd's view of fstab and confirm it parses cleanly:

```bash
sudo systemctl daemon-reload
sudo systemd-analyze verify /etc/fstab 2>&1 || true   # no output = OK
```

Recovering from this hang requires force-stopping the instance (which **wipes the
NVMe instance store**), detaching the root EBS, editing fstab from another instance,
and reattaching. Treat `/opt/dlami/nvme` as throwaway; keep real work on `/mnt/efs`.

```bash
# c) Save/checkpoint and kill all GPU jobs — the kernel module won't unload otherwise.
nvidia-smi --query-compute-apps=pid --format=csv   # must be EMPTY before proceeding
```

---

## 1. Stop GPU services

```bash
sudo systemctl stop nvidia-fabricmanager
sudo systemctl stop nvidia-dcgm
```

---

## 2. Add NVIDIA's CUDA network repo

The DLAMI ships only *local* CUDA repos (12.1/12.3), which don't carry R570. Add
the network repo:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
apt-cache madison cuda-drivers-570   # sanity: should list 570.x from developer.download.nvidia.com
```

---

## 3. Purge the old 545 stack **and the unversioned `cuda`/`cuda-drivers` metas**

### 3a. First decide: do you even need the purge?

Instances differ — some carry the dangerous unversioned metas, some only have an
old driver blocking the upgrade, some install cleanly. Run this **after step 2**
(repo added) and **before installing anything**. It dry-runs the 570 install
(read-only, keying off apt's exit code) and prints one of three verdicts:

```bash
apt-get -s install cuda-drivers-570 >/tmp/dry.txt 2>&1; rc=$?
if [ $rc -ne 0 ]; then
  echo ">>> BLOCKED → cleanup needed (purge old driver; 3b, 'mild' case)"
  grep -iE 'held broken|not going to be installed|^E:' /tmp/dry.txt
elif grep -qiE '^Inst (nvidia-driver-575|libnvidia-compute-575)' /tmp/dry.txt \
   || grep -qiE '^Inst cuda.*12-9' /tmp/dry.txt; then
  echo ">>> DANGEROUS: would pull R575 / CUDA 12.9 → full purge (3b, 'dangerous' case)"
else
  echo ">>> CLEAN: cuda-drivers-570 installs directly — skip to step 4."
  echo "    Planned removals (confirm nothing protected is listed):"
  grep -E '^Remv' /tmp/dry.txt
fi
```

> **Why not just `grep 575`?** The helper packages `nvidia-modprobe` /
> `nvidia-settings` are legitimately versioned `575.x` and get pulled even by a
> *clean* 570 install, so a bare `575` match gives false positives. The check
> above keys off apt's exit code and only flags the real R575 **driver**
> (`nvidia-driver-575`, `libnvidia-compute-575`) and the CUDA-12.9 toolkit.

**Verdicts:**

- **CLEAN** → skip the rest of step 3; go straight to **step 4**. apt auto-removes
  the old 545 stack as part of the install. (Collateral monitoring tools like
  `gpustat` / `nvtop` / `python3-pynvml` may also be removed — reinstall them
  afterward if you use them. Confirm the `Remv` list contains **no** protected
  package: container-toolkit, efa-nv-peermem, gdrdrv, docker, DCGM.)
- **BLOCKED (mild)** → an old `nvidia-driver-545` is present but apt won't
  auto-resolve it. Do the purge in 3b; the meta/575 entries are harmless no-ops.
- **DANGEROUS** → unversioned `cuda`/`cuda-drivers` metas are installed
  (confirm with `dpkg -l cuda cuda-drivers | grep '^ii'`) → do the full purge in 3b.

### 3b. The purge

⚠️ **The R575/CUDA-12.9 trap.** If the unversioned `cuda` / `cuda-drivers`
meta-packages are installed, they always point at the *newest* CUDA — so just
running `apt-get install cuda-drivers-570` on top makes apt drag in **CUDA 12.9 +
the R575 driver** and then die on a file conflict, leaving a half-installed mess.
Remove the old driver **and** the unversioned metas first so nothing pulls "latest".

```bash
sudo apt-get purge -y \
  cuda 'cuda-12-9' 'cuda-*-12-9' 'libcu*-12-9' 'cuda-12-3' 'cuda-*-12-3' 'cuda-12-1' 'cuda-*-12-1' \
  'libnvidia-*-575' 'nsight-compute-2025.2.1' \
  'nvidia-*-545' 'libnvidia-*-545' 'libnvidia-compute-430' 'libnvidia-compute-418' \
  xserver-xorg-video-nvidia-545 nvtop \
  cuda-drivers cuda-drivers-545 cuda-drivers-fabricmanager-545 nvidia-fabricmanager-545
sudo dpkg --configure -a
sudo apt-get -f install -y
sudo apt-get autoremove -y
```

> Adjust the globs to whatever the instance actually has — if `apt-get purge` errors
> with "unmet dependencies" naming a reverse-dep (we hit `nvtop` and
> `xserver-xorg-video-nvidia-545`), add that package to the list and re-run.
> Simulate first with `apt-get -s purge ...` to preview.

⚠️ **Do NOT** use a blanket `apt-get purge '*nvidia*'` — that removes packages you
must keep: `nvidia-container-toolkit`, `libnvidia-container1`, `nvidia-docker2`,
`efa-nv-peermem`, `gdrdrv-dkms`, `datacenter-gpu-manager`.

**Checkpoint — clean slate.** This should show no `iU`/`ic` packages (only `rc`
config-file leftovers and the protected container/EFA packages are fine):

```bash
dpkg -l | grep -iE 'nvidia|cuda' | grep -vE '^ii'
```

---

## 4. Install the R570 driver (driver-only, versioned)

```bash
sudo apt-get install -y cuda-drivers-570
```

⚠️ Always the **versioned** `cuda-drivers-570` — never bare `cuda-drivers` or
`cuda` (they resolve to the newest branch — R575 at the time of writing). Watch the output: it builds the DKMS kernel module
against the running kernel (`nvidia`, `nvidia-modeset`, `nvidia-drm`,
`nvidia-uvm`, `nvidia-peermem`). `EFI variables are not supported … aborting` is
harmless (Secure Boot signing skipped on BIOS-boot instances).

---

## 5. Install Fabric Manager + NSCQ — **version-matched** (mandatory on NVSwitch)

⚠️ `cuda-drivers-570` does **not** pull Fabric Manager. On these NVSwitch instances,
without a *version-matched* Fabric Manager the GPUs won't initialize across
NVSwitch. The upstream version must equal the driver's; note FM uses the `-1`
Debian revision while the driver libs use `-0ubuntu1`. Ubuntu multiverse only has
an older 570.133.20 — use NVIDIA's repo build.

```bash
# Match this to the driver version installed in step 4 (check: nvidia-smi or dpkg -l | grep nvidia-driver-570)
DRV=570.211.01
sudo apt-get install -y nvidia-fabricmanager-570=${DRV}-1 libnvidia-nscq-570=${DRV}-1
```

---

## 6. Enable services

```bash
sudo systemctl enable nvidia-fabricmanager
sudo systemctl enable nvidia-persistenced
```

---

## 7. Checkpoint before reboot

```bash
dpkg -l | grep -E '^ii.*570' | awk '{print $2,$3}' | grep -iE 'driver|dkms|fabricmanager|nscq'
dkms status | grep nvidia      # nvidia, <DRV>, <kernel>: installed
```

Confirm `nvidia-driver-570`, `nvidia-dkms-570`, `nvidia-fabricmanager-570`,
`libnvidia-nscq-570` are all the **same upstream version** and the DKMS module is
`installed`. **Re-confirm `/etc/fstab` is fixed (step 0b) before rebooting.**

```bash
sudo reboot
```

---

## 8. Verify after reboot

```bash
nvidia-smi                              # Driver 570.x, CUDA 12.8, all 8 GPUs
systemctl status nvidia-fabricmanager   # active (running)
systemctl status nvidia-persistenced    # active
nvidia-smi topo -m                       # every GPU pair shows NV12 (full NVSwitch mesh)
```

End-to-end container check:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If `efa-nv-peermem` / `gdrdrv` didn't rebuild against the new driver
(`dkms status`), reinstall the EFA stack:

```bash
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
tar -xf aws-efa-installer-latest.tar.gz && cd aws-efa-installer && sudo ./efa_installer.sh -y
```

---

## 9. (Optional) Install a host CUDA toolkit — usually NOT needed

You do **not** need a host CUDA toolkit for containerized workloads (e.g. Alpasim)
— containers ship their own CUDA, the host only needs the driver. Do this step
**only** if you want host-side `nvcc`, `cuda-gdb`, Nsight, or to build CUDA code
directly on the host.

It's low-risk and much simpler than the driver: **userspace only — no reboot, no
kernel modules, no Fabric Manager, no fstab risk**, and it's purely additive
(doesn't upgrade or remove existing packages, so it leaves the R570 driver and
Fabric Manager untouched). The `cuda-toolkit-12-8` package does **not** depend on
`cuda-drivers`, so it can't disturb the driver.

⚠️ **Same meta trap as step 3.** Install the **versioned** `cuda-toolkit-12-8`.
**Never** the bare `cuda` or `cuda-toolkit` metas — they resolve to the newest
CUDA (12.9 at time of writing), which drags in R575 + `cuda-drivers` and recreates
the mess from step 3. Also match the toolkit to the driver: R570 supports CUDA
12.8, so use the **12.8** toolkit (not 12.9).

```bash
# (repo already added in step 2) — install the toolkit only
sudo apt-get install -y cuda-toolkit-12-8

# the package sets /usr/local/cuda -> cuda-12.8 but does NOT touch PATH; add it yourself:
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# verify (toolkit in; driver/FM unchanged)
nvcc --version                            # release 12.8
nvidia-smi | head -4                       # still Driver 570.x / CUDA 12.8
systemctl is-active nvidia-fabricmanager   # active
```

For all users on the instance, put the two `export` lines in
`/etc/profile.d/cuda.sh` instead of `~/.bashrc`. The CUDA apt packages never
modify `PATH` automatically (binaries live in `/usr/local/cuda-X.Y/bin` so
multiple versions can coexist), so the manual step is expected, not a bug.

---

## Summary of issues hit on instance `ai-foundation-p4de-3`

| Issue | Cause | Avoided by |
|---|---|---|
| apt pulled **R575 + CUDA 12.9**, then crashed on a file conflict | Unversioned `cuda`/`cuda-drivers` metas point at "latest" | Purge old driver **and** the unversioned metas before installing (step 3); only ever install versioned `cuda-drivers-570` |
| GPUs would not come up across NVSwitch | `cuda-drivers-570` doesn't pull Fabric Manager | Explicitly install version-matched `nvidia-fabricmanager-570` + `libnvidia-nscq-570` (step 5) |
| FM refuses to start | FM version ≠ driver version (multiverse had older 570.133.20) | Pin FM/NSCQ to the driver's upstream version from NVIDIA's repo |
| **Instance hung at boot, unreachable; NVMe data lost** | `/etc/fstab` `nofail` NVMe line missing `x-systemd.device-timeout=0` | Fix fstab in pre-flight (step 0b); never trust an EBS snapshot to cover `/opt/dlami/nvme` |
| Purge refused with "unmet dependencies" | Reverse-deps (`nvtop`, `xserver-xorg-video-nvidia-545`) not in the purge list | Add the named package and re-run; simulate with `apt-get -s purge` first |
