#!/bin/bash
# Builds one variant of the Kampylos BOINC wrappers against the fake BOINC runtime (tests/boinc_stub), for run_bench.sh.
# usage: build_variant.sh <name> [patch ...]
#   Copies the repository's tracked files (with the VBMicrolensing submodule) to $KAMPYLOS_WORK/build/<name>, applies the
#   patches (git diff format, -p1), prepares VBMicrolensing and builds bins/<name>/cpu (and bins/<name>/metal on macOS).
#   env: VBM    fork (default: the submodule as it is, thread_local statics), strip or block
#               (vendor/patch_vbm_thread_local.py --strip / --block)
#        ALLOC  system (default), jemalloc (JEMALLOC = static library or linker flags; default: Homebrew's
#               libjemalloc.a on macOS, -ljemalloc elsewhere) or mimalloc (MIMALLOC, default -lmimalloc: on Linux
#               the shared library replaces malloc by symbol order; on macOS give flags that really override it)
#        OPT    compiler flags (default -O2, the build lines of boinc/README.md)
#        BACKENDS  default "cpu metal" on macOS, "cpu" elsewhere
#        CXX    compiler (default clang++ on macOS, c++ elsewhere); KAMPYLOS_WORK (default ~/kampylos-work);
#        REF    build a git ref of this repository (e.g. upstream/master) instead of its working tree; the
#               VBMicrolensing submodule then comes from the working tree, which must be at the ref's commit
#        KAMPYLOS_SRC  the source tree to copy (default: this repository)
set -e -o pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo=${KAMPYLOS_SRC:-$(cd "$here/../.." && pwd)}
[ $# -ge 1 ] || { sed -n 2,17p "$0"; exit 2; }
name=$1; shift
W=${KAMPYLOS_WORK:-$HOME/kampylos-work}
VBMMODE=${VBM:-fork}; ALLOC=${ALLOC:-system}; OPT=${OPT:--O2}
if [ "$(uname)" = Darwin ]; then
  BACKENDS=${BACKENDS:-cpu metal}; CXX=${CXX:-clang++}; JEMALLOC=${JEMALLOC:-$(brew --prefix 2>/dev/null || echo /opt/homebrew)/lib/libjemalloc.a}
else
  BACKENDS=${BACKENDS:-cpu}; CXX=${CXX:-c++}; JEMALLOC=${JEMALLOC:--ljemalloc}
fi
T=$W/build/$name; out=$W/bins/$name
rm -rf "$T"; mkdir -p "$T" "$out"
if [ -n "${REF:-}" ]; then
  want=$(cd "$repo" && git ls-tree "$REF" vendor/VBMicrolensing | awk '{print $3}')
  have=$(cd "$repo" && git -C vendor/VBMicrolensing rev-parse HEAD 2>/dev/null || echo none)
  [ "$want" = "$have" ] || { echo "$REF wants VBMicrolensing $want, the checkout has $have"; exit 2; }
  (cd "$repo" && git archive "$REF") | tar -xf - -C "$T"
  (cd "$repo/vendor/VBMicrolensing" && git ls-files -z | tar --null -T - -cf -) | tar -xf - -C "$T/vendor/VBMicrolensing"
else
  (cd "$repo" && git ls-files -z --recurse-submodules | tar --null -T - -cf -) | tar -xf - -C "$T"
fi
for p in "$@"; do a="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")"; (cd "$T" && patch -p1 -s < "$a"); done
VBCPP=$T/vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp
case $VBMMODE in
  fork) ;;
  strip|block) (cd "$T" && python3 vendor/patch_vbm_thread_local.py --$VBMMODE > /dev/null) ;;
  *) echo "VBM must be fork, strip or block"; exit 2 ;;
esac
# an older patch script ignores an option it does not know (--block before 2026-11-04)
if [ $VBMMODE = strip ] && grep -q 'thread_local static' "$VBCPP"; then echo "--strip left thread_local statics"; exit 1; fi
if [ $VBMMODE = block ] && ! grep -q '_vbm_tls_t' "$VBCPP"; then echo "this tree's patch_vbm_thread_local.py has no --block"; exit 1; fi
case $ALLOC in
  system) L="" ;;
  jemalloc) L=$JEMALLOC ;;
  mimalloc) L=${MIMALLOC:--lmimalloc} ;;
  *) echo "ALLOC must be system, jemalloc or mimalloc"; exit 2 ;;
esac
VB=$T/vendor/VBMicrolensing/VBMicrolensing/lib
log=$T/build.log
$CXX $OPT -std=c++17 -I "$VB" -c "$VB/VBMicrolensingLibrary.cpp" -o "$T/vbm.o" > "$log" 2>&1 || true
[ -f "$T/vbm.o" ] || { grep -E 'error' "$log" | head; echo "VBMicrolensing did not compile, see $log"; exit 1; }
for b in $BACKENDS; do
  rm -f "$out/$b"
  case $b in
    cpu) $CXX $OPT -std=c++17 -I "$T/src" -I "$VB" -I "$T/tests/boinc_stub" "$T/boinc/main_kampylos_boinc.cpp" \
           "$T/tests/boinc_stub/boinc_stub.cpp" "$T/vbm.o" $L -lpthread -o "$out/cpu" >> "$log" 2>&1 || true ;;
    metal) $CXX $OPT -std=c++17 -fobjc-arc -I "$T/src" -I "$VB" -I "$T/tests/boinc_stub" "$T/boinc/main_kampylos_metal_boinc.cpp" \
           "$T/src/kampylos_metal_fit.mm" "$T/tests/boinc_stub/boinc_stub.cpp" "$T/vbm.o" $L \
           -framework Metal -framework Foundation -framework CoreGraphics -o "$out/metal" >> "$log" 2>&1 || true ;;
  esac
  [ -x "$out/$b" ] || { grep -E 'error' "$log" | head; echo "build of $b failed, see $log"; exit 1; }
done
echo "built $out: $(cd "$out" && ls | tr '\n' ' ')(${REF:-working tree} VBM=$VBMMODE ALLOC=$ALLOC OPT=$OPT)"
