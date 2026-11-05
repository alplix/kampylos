#!/bin/bash
# One-command A/B on a Mac (Apple Silicon or Intel with a Metal GPU): the builds of BASE_REF made as boinc/README.md
# describes them, against this working tree made the fast way, on the profiles in scripts/bench/profiles.
# usage: scripts/bench/mac_ab.sh [rounds per profile, default 3] [profiles, default "planet ogle far planet-gpu"]
#   env: BASE_REF  the baseline (default upstream/master if that remote exists, else origin/master)
#        KAMPYLOS_WORK (default ~/kampylos-work): builds in build/ and bins/, results in bench-runs/<date>-<profile>/
#        EXPERIMENTS=1  also the search changes of patches/nm-stopping.patch on planet and ogle (they change results,
#                       see docs/BENCHMARK.md): cpu-restart (restart only after a gain > 1e-3) and cpu-stall (also stop
#                       a pass after 100 iterations without a gain > 1e-4)
# Needs the Xcode command line tools, python3, the VBMicrolensing submodule (git submodule update --init) and, for the
# fast builds, Homebrew's jemalloc (brew install jemalloc). About 1 minute of building, then 15 to 30 minutes per round of
# the four profiles on an M1.
#
# Binaries (labels):
#   cpu-doc    BASE_REF, CPU app, the build lines of boinc/README.md (thread_local VBMicrolensing, system malloc)
#   cpu-fast   BASE_REF, CPU app, boinc/README.md "Faster CPU build" (--strip, jemalloc)
#   cpu-new    working tree, CPU app, the same fast build (shows whether this tree changes the CPU app's output)
#   metal-doc  BASE_REF, Metal app, the build line of boinc/README.md
#   metal-new  working tree, Metal app, VBMicrolensing --block and jemalloc (boinc/README.md "Faster GPU app build")
# The CPU labels run on planet, ogle and far; the Metal labels on planet-gpu and far.
set -e -o pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
ROUNDS=${1:-3}
PROFILES=${2:-planet ogle far planet-gpu}
W=${KAMPYLOS_WORK:-$HOME/kampylos-work}
export KAMPYLOS_WORK=$W
if [ -z "${BASE_REF:-}" ]; then
  if git -C "$repo" rev-parse -q --verify upstream/master > /dev/null; then BASE_REF=upstream/master; else BASE_REF=origin/master; fi
fi
B=$W/bins
[ -f "$repo/vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp" ] || { echo "run: git submodule update --init"; exit 2; }
git -C "$repo/vendor/VBMicrolensing" diff --quiet || { echo "vendor/VBMicrolensing has local changes: git -C vendor/VBMicrolensing checkout -- ."; exit 2; }
JEMALLOC=${JEMALLOC:-$(brew --prefix 2>/dev/null || echo /opt/homebrew)/lib/libjemalloc.a}
[ -f "$JEMALLOC" ] || { echo "no $JEMALLOC: brew install jemalloc (or set JEMALLOC)"; exit 2; }
export JEMALLOC

REF=$BASE_REF VBM=fork BACKENDS="cpu metal" bash "$here/build_variant.sh" doc
REF=$BASE_REF VBM=strip ALLOC=jemalloc BACKENDS=cpu bash "$here/build_variant.sh" fast
VBM=strip ALLOC=jemalloc BACKENDS=cpu bash "$here/build_variant.sh" newcpu
VBM=block ALLOC=jemalloc BACKENDS=metal bash "$here/build_variant.sh" newmetal

CPU="cpu-doc=$B/doc/cpu cpu-fast=$B/fast/cpu cpu-new=$B/newcpu/cpu"
if [ "${EXPERIMENTS:-0}" = 1 ]; then
  P=$here/patches/nm-stopping.patch
  OPT="-O2 -DKAMPYLOS_RESTART_MIN_GAIN=1e-3" VBM=strip ALLOC=jemalloc BACKENDS=cpu bash "$here/build_variant.sh" restart "$P"
  OPT="-O2 -DKAMPYLOS_RESTART_MIN_GAIN=1e-3 -DKAMPYLOS_NM_STALL_ITERS=100 -DKAMPYLOS_NM_STALL_TOL=1e-4" \
    VBM=strip ALLOC=jemalloc BACKENDS=cpu bash "$here/build_variant.sh" stall "$P"
fi
GPU="metal-doc=$B/doc/metal metal-new=$B/newmetal/metal"
for prof in $PROFILES; do
  case $prof in
    planet|ogle)
      if [ "${EXPERIMENTS:-0}" = 1 ]; then bash "$here/run_bench.sh" "$prof" "$ROUNDS" $CPU cpu-restart=$B/restart/cpu cpu-stall=$B/stall/cpu | tail -7
      else bash "$here/run_bench.sh" "$prof" "$ROUNDS" $CPU | tail -5; fi ;;
    far) bash "$here/run_bench.sh" "$prof" "$ROUNDS" $CPU $GPU | tail -7 ;;
    *) bash "$here/run_bench.sh" "$prof" "$ROUNDS" $GPU | tail -3 ;;
  esac
done
