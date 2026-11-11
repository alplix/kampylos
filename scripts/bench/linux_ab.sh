#!/bin/bash
# Linux A/B of the VBMicrolensing forms and allocators with the CPU app, whose time is the same library code as the GPU
# apps' completion step. On Linux it runs here; on macOS it runs itself in a Debian 12 container (docker, or
# DOCKER=podman). A container gives ratios between Linux builds, not Linux-against-macOS numbers: it shares the Mac's
# cores and load.
# usage: scripts/bench/linux_ab.sh [rounds, default 3] [profile, default planet]
#   env: KAMPYLOS_WORK (default ~/kampylos-work; from a Mac the container works in $KAMPYLOS_WORK/linux), CLANG=1 also
#        builds fork and strip+jemalloc with clang++, DOCKER (default docker)
# Labels: fork (thread_local library, system malloc: the GPU apps' library as boinc/README.md builds it), block
# (--block), strip (plain statics: the CPU app's library), block-jem and strip-jem (jemalloc), strip-mi (mimalloc);
# with CLANG=1 also clang-fork and clang-strip-jem. Every build should write the same result lines as fork, except
# the clang ones, which are compared with each other (another compiler differs at rounding level).
# Needs on Linux: g++ (and clang++ for CLANG=1), python3, git, patch, GNU time, libjemalloc-dev, libmimalloc-dev. On a
# Mac: docker or podman, and network access once to build the image.
set -e -o pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
ROUNDS=${1:-3}
PROFILE=${2:-planet}
if [ "$(uname)" = Darwin ]; then
  D=${DOCKER:-docker}
  W=${KAMPYLOS_WORK:-$HOME/kampylos-work}/linux
  IMAGE=kampylos-bench:linux-1      # bump the tag when the package list changes
  mkdir -p "$W/image"
  if ! $D image inspect $IMAGE > /dev/null 2>&1; then
    cat > "$W/image/Dockerfile" <<'EOF'
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends g++ clang git patch time libjemalloc-dev \
    libmimalloc-dev && rm -rf /var/lib/apt/lists/* && git config --global --add safe.directory '*'
EOF
    $D build -q -t $IMAGE "$W/image"
  fi
  exec $D run --rm -v "$repo":/src:ro -v "$W":/work -e KAMPYLOS_WORK=/work -e CLANG="${CLANG:-0}" $IMAGE \
    bash /src/scripts/bench/linux_ab.sh "$ROUNDS" "$PROFILE"
fi

W=${KAMPYLOS_WORK:-$HOME/kampylos-work}
export KAMPYLOS_WORK=$W
B=$W/bins
echo "== $(g++ --version | head -1), $(ldd --version 2>&1 | head -1)"
VBM=fork bash "$here/build_variant.sh" lx-fork
VBM=block bash "$here/build_variant.sh" lx-block
VBM=strip bash "$here/build_variant.sh" lx-strip
VBM=block ALLOC=jemalloc bash "$here/build_variant.sh" lx-block-jem
VBM=strip ALLOC=jemalloc bash "$here/build_variant.sh" lx-strip-jem
VBM=strip ALLOC=mimalloc bash "$here/build_variant.sh" lx-strip-mi
# 2026-11-05: the free lists without an extra allocator, and with history-free evaluations (switch on)
VBM=strip VBMX=pool bash "$here/build_variant.sh" lx-strip-pool
VBM=strip VBMX="reset pool" OPT="-O2 -DKAMPYLOS_RESET_STATE=1" bash "$here/build_variant.sh" lx-strip-reset-pool
LABELS="fork=$B/lx-fork/cpu block=$B/lx-block/cpu strip=$B/lx-strip/cpu block-jem=$B/lx-block-jem/cpu"
LABELS="$LABELS strip-jem=$B/lx-strip-jem/cpu strip-mi=$B/lx-strip-mi/cpu strip-pool=$B/lx-strip-pool/cpu strip-reset-pool=$B/lx-strip-reset-pool/cpu"
if [ "${CLANG:-0}" = 1 ]; then
  echo "== $(clang++ --version | head -1)"
  CXX=clang++ VBM=fork bash "$here/build_variant.sh" lx-clang-fork
  CXX=clang++ VBM=strip ALLOC=jemalloc bash "$here/build_variant.sh" lx-clang-strip-jem
  LABELS="$LABELS clang-fork=$B/lx-clang-fork/cpu clang-strip-jem=$B/lx-clang-strip-jem/cpu"
fi
bash "$here/run_bench.sh" "$PROFILE" "$ROUNDS" $LABELS
