#!/bin/bash
# Builds of the CPU app for the x86_64 platforms that a Mac cannot run natively, run under emulation to check that they
# compile, link, run to the end and write the result lines we expect: Windows (cross-compiled with MinGW-w64 GCC, run
# under Wine) and Linux x86_64 (GCC), both in an x86_64 Debian 12 container. On Apple Silicon the container runs under
# Rosetta, so the times are only a rough ratio between the builds of one platform, never a speed claim
# (docs/BENCHMARK.md, "Other platforms"). On an x86_64 Linux machine with Wine it runs natively.
# usage: scripts/bench/cross_check.sh [profile, default planet] [rounds, default 1] [forms, default "fork block strip"]
#   env: KAMPYLOS_WORK (default ~/kampylos-work; the container works in $KAMPYLOS_WORK/x86), MINGW_CXX (default
#        x86_64-w64-mingw32-g++, found on the host: brew install mingw-w64 on a Mac), DOCKER (default docker)
# Labels: win-<form> (MinGW-w64, static, run by Wine) and lx64-<form> (Linux x86_64 GCC), for each VBMicrolensing form
# (fork = thread_local library as boinc/README.md builds it, block = --block, strip = plain statics; since 2026-11-05 a
# form may add the patch script's extras with "+", e.g. strip+reset+pool, and "reset" then also builds with
# -DKAMPYLOS_RESET_STATE=1). Within one
# platform every form must write the same result lines; between platforms they may differ at rounding level (KAM-025),
# and the table shows whether they do.
# MinGW-w64 GCC implements thread_local with emulated TLS (__emutls_get_address, checked with GCC 16.2 on 2026-11-04);
# that is the case the --block form was written for.
set -e -o pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
# form "strip+reset+pool" -> VBM=strip, VBMX="reset pool", and the switch of history-free evaluations if reset is in it
form_vbm() { echo "${1%%+*}"; }
form_vbmx() { case $1 in *+*) echo "${1#*+}" | tr '+' ' ' ;; esac; }
form_opt() { case "+$1+" in *+reset+*) echo "-DKAMPYLOS_RESET_STATE=1" ;; esac; }
PROFILE=${1:-planet}
ROUNDS=${2:-1}
FORMS=${3:-fork block strip}
IMAGE=kampylos-bench:x86-1      # bump the tag when the package list changes

if [ -z "${CROSS_CHECK_INSIDE:-}" ]; then
  D=${DOCKER:-docker}
  W=${KAMPYLOS_WORK:-$HOME/kampylos-work}/x86
  MINGW_CXX=${MINGW_CXX:-x86_64-w64-mingw32-g++}
  command -v "$MINGW_CXX" > /dev/null || { echo "no $MINGW_CXX: brew install mingw-w64 (or set MINGW_CXX)"; exit 2; }
  echo "== Windows builds: $($MINGW_CXX --version | head -1)"
  # Static: no libstdc++/libgcc/libwinpthread DLLs to ship or to find under Wine
  for f in $FORMS; do
    KAMPYLOS_WORK=$W CXX=$MINGW_CXX OPT="-O2 -static $(form_opt $f)" BACKENDS=cpu VBM=$(form_vbm $f) VBMX="$(form_vbmx $f)" \
      bash "$here/build_variant.sh" win-$f
  done
  mkdir -p "$W/image"
  if ! $D image inspect $IMAGE > /dev/null 2>&1; then
    cat > "$W/image/Dockerfile" <<'EOF'
FROM --platform=linux/amd64 debian:12-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ git patch time python3 wine wine64 ca-certificates \
    && rm -rf /var/lib/apt/lists/* && git config --global --add safe.directory '*'
ENV WINEDEBUG=-all
RUN wineboot -i > /dev/null 2>&1 || true
EOF
    $D build --platform linux/amd64 -q -t $IMAGE "$W/image"
  fi
  exec $D run --rm --platform linux/amd64 -v "$repo":/src:ro -v "$W":/work -e KAMPYLOS_WORK=/work \
    -e CROSS_CHECK_INSIDE=1 $IMAGE bash /src/scripts/bench/cross_check.sh "$PROFILE" "$ROUNDS" "$FORMS"
fi

# Inside the container (or on an x86_64 Linux machine with Wine and the win-* builds already made)
W=${KAMPYLOS_WORK:-$HOME/kampylos-work}
export KAMPYLOS_WORK=$W
B=$W/bins
echo "== Linux builds: $(g++ --version | head -1), $(ldd --version 2>&1 | head -1), $(uname -m); $(wine --version 2>/dev/null)"
LABELS=""
for f in $FORMS; do
  OPT="-O2 $(form_opt $f)" VBM=$(form_vbm $f) VBMX="$(form_vbmx $f)" bash "$here/build_variant.sh" lx64-$f
  # run_bench.sh runs a label's file directly; Wine needs the .exe name and its own launcher
  mkdir -p "$W/wrap"
  cp "$B/win-$f/cpu" "$B/win-$f/cpu.exe"
  printf '#!/bin/sh\nexec wine %s "$@"\n' "$B/win-$f/cpu.exe" > "$W/wrap/win-$f"
  chmod +x "$W/wrap/win-$f"
  LABELS="$LABELS win-$f=$W/wrap/win-$f lx64-$f=$B/lx64-$f/cpu"
done
bash "$here/run_bench.sh" "$PROFILE" "$ROUNDS" $LABELS
