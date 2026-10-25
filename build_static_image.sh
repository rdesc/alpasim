#!/usr/bin/env bash
# Rebuild the base image with the `static` driver entry point baked in.
#
# The image sets UV_NO_SYNC=1 (Dockerfile:58), so the venv metadata inside it
# is frozen at build time while src/ is bind-mounted live at run time. A driver
# plugin registered in src/driver/pyproject.toml under the "alpasim.models"
# entry-point group is therefore invisible to a running container until the
# image is rebuilt. That is why the BC launchers pin defines.base_image to this
# tag instead of taking the default alpasim-base:${repo-version:}.
#
# Needed only once per change to the driver plugin registry; the launchers abort
# if the tag is missing.
set -uo pipefail

REPO=/mnt/efs/users/rod/repos/alpasim-e2e
TAG=alpasim-base:0.96.0-static
LOG=/mnt/efs/users/rod/repos/alpasim/build_static_image.log

cd "$REPO" || { echo "no repo at $REPO" >&2; exit 1; }
echo "building $TAG from $REPO (log: $LOG)"
docker build -t "$TAG" . > "$LOG" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "docker build failed (rc=$rc); see $LOG" >&2; tail -20 "$LOG" >&2; exit 1
fi

# The point of the rebuild is the entry point, so check for it rather than
# trusting that the build succeeded.
got=$(docker run --rm --entrypoint /repo/.venv/bin/python "$TAG" -c \
  'from importlib.metadata import entry_points
print(",".join(sorted(e.name for e in entry_points(group="alpasim.models"))))' 2>&1)
echo "alpasim.models entry points: $got"
case "$got" in
  *static*) echo "OK: $TAG has the static driver" ;;
  *) echo "FAIL: static not registered in $TAG" >&2; exit 1 ;;
esac
