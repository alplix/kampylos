#!/bin/bash
# The correctness checks of the faster GPU app (VBMicrolensing --block, jemalloc, GPU/CPU overlap in the Metal app),
# in one command on a Mac with a Metal GPU. The speed itself is mac_ab.sh's job; this one checks that nothing broke.
# usage: scripts/bench/check_change.sh
#   Builds the Metal app of BASE_REF the way boinc/README.md describes (bins/doc) and this working tree's the fast way
#   (bins/newmetal), then runs, in the copy of this tree that build_variant.sh made (build/newmetal):
#     vbm-threads  tests/test_vbm_threads.cpp with the --block library: 4 threads, every value equal to the sequential one
#     metal-fit    tests/test_metal_fit.mm on the synthetic planet: the Metal objective against the CPU one
#     resume       tests/test_boinc_resume.sh with KAMPYLOS_RESUME_BACKEND=metal: killed and restarted until it
#                  finishes, output byte-identical to an uninterrupted run (about 15 minutes; SKIP_RESUME=1 skips it)
#     threads4     profile planet-gpu with 4 completion threads, BASE_REF against this tree: the same result lines
#                  (several chunks per round, so the overlap and complete_batch(first, count) are exercised)
#   env: BASE_REF (default upstream/master if that remote exists, else origin/master), KAMPYLOS_WORK (default
#        ~/kampylos-work), JEMALLOC (default Homebrew's libjemalloc.a), CXX (default clang++)
# Prints ok or FAIL per check and PASS at the end only when every check passed; logs in $KAMPYLOS_WORK/check-<date>/.
# About 20 minutes on an M1. Run it on a quiet machine: a BOINC task on the same GPU makes the driver abort command
# buffers (KAM-024), which the app survives but which slows the resume test down.
set -u -o pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
[ "$(uname)" = Darwin ] || { echo "check_change.sh needs macOS with a Metal GPU"; exit 2; }
W=${KAMPYLOS_WORK:-$HOME/kampylos-work}
export KAMPYLOS_WORK=$W
if [ -z "${BASE_REF:-}" ]; then
  if git -C "$repo" rev-parse -q --verify upstream/master > /dev/null; then BASE_REF=upstream/master; else BASE_REF=origin/master; fi
fi
[ -f "$repo/vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp" ] || { echo "run: git submodule update --init"; exit 2; }
git -C "$repo/vendor/VBMicrolensing" diff --quiet || { echo "vendor/VBMicrolensing has local changes: git -C vendor/VBMicrolensing checkout -- ."; exit 2; }
JEMALLOC=${JEMALLOC:-$(brew --prefix 2>/dev/null || echo /opt/homebrew)/lib/libjemalloc.a}
[ -f "$JEMALLOC" ] || { echo "no $JEMALLOC: brew install jemalloc (or set JEMALLOC)"; exit 2; }
export JEMALLOC
CXX=${CXX:-clang++}
LOG=$W/check-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOG"
echo "# $(date '+%Y-%m-%d %H:%M:%S') BASE_REF $BASE_REF ($(git -C "$repo" rev-parse --short "$BASE_REF")), tree at $(git -C "$repo" rev-parse --short HEAD)$(git -C "$repo" diff --quiet HEAD -- src boinc vendor || echo ' + local changes'), $(uptime | sed 's/.*load/load/')" | tee "$LOG/summary.txt"

REF=$BASE_REF VBM=fork BACKENDS=metal bash "$here/build_variant.sh" doc || exit 1
VBM=block ALLOC=jemalloc BACKENDS=metal bash "$here/build_variant.sh" newmetal || exit 1
T=$W/build/newmetal
VB=$T/vendor/VBMicrolensing/VBMicrolensing/lib

failed=""
check() { # name command...: runs the command with its output in $LOG/<name>.log
  local name=$1; shift
  printf '%-12s ' "$name" | tee -a "$LOG/summary.txt"
  local t0=$SECONDS
  if "$@" > "$LOG/$name.log" 2>&1; then
    echo "ok    ($((SECONDS - t0)) s)" | tee -a "$LOG/summary.txt"
  else
    echo "FAIL  ($((SECONDS - t0)) s, see $LOG/$name.log)" | tee -a "$LOG/summary.txt"
    tail -5 "$LOG/$name.log" | sed 's/^/             /'
    failed="$failed $name"
  fi
}
# Each check runs in a subshell (its cd stays there); vbm.o is the --block library build_variant.sh compiled for the app
vbm_threads() (
  cd "$T" && $CXX -O2 -std=c++17 -I "$VB" tests/test_vbm_threads.cpp vbm.o -lpthread -o "$LOG/test_vbm_threads" &&
    "$LOG/test_vbm_threads"
)
metal_fit() (
  cd "$T" && $CXX -O2 -std=c++17 -fobjc-arc -I src -I "$VB" tests/test_metal_fit.mm src/kampylos_metal_fit.mm vbm.o \
    -framework Metal -framework Foundation -framework CoreGraphics -o "$LOG/test_metal_fit" &&
    "$LOG/test_metal_fit" tests/regression/synthetic_planet.dat
)
resume() (
  cd "$T" && KAMPYLOS_RESUME_BACKEND=metal CXX=$CXX sh tests/test_boinc_resume.sh "$LOG/resume-work"
)
threads4() {
  KAMPYLOS_STUB_NCPUS=4 BENCH_OUT=$LOG bash "$here/run_bench.sh" planet-gpu 1 metal-doc="$W/bins/doc/metal" \
    metal-new="$W/bins/newmetal/metal" || return 1
  grep -h '^metal-new ' "$LOG"/*-planet-gpu/summary.txt | grep -q '; same output$'
}
check vbm-threads vbm_threads
check metal-fit metal_fit
if [ "${SKIP_RESUME:-0}" = 1 ]; then echo "resume       skipped (SKIP_RESUME=1)" | tee -a "$LOG/summary.txt"; else check resume resume; fi
check threads4 threads4

if [ -z "$failed" ]; then echo PASS | tee -a "$LOG/summary.txt"; else echo "FAIL:$failed" | tee -a "$LOG/summary.txt"; exit 1; fi
