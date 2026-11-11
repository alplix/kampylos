#!/bin/bash
# Interleaved speed benchmark of Kampylos BOINC wrappers built against tests/boinc_stub (build_variant.sh).
# usage: run_bench.sh <profile> <rounds> <label>=<binary> [<label>=<binary> ...]
#   profile: a file in scripts/bench/profiles/ without .wu, or a path to a .wu file (photometry, input line, CPU budget)
#   Runs A B C ... A B C ... (rounds) in fresh run directories, so that load drift hits every binary alike.
#   env: BENCH_OUT (default $KAMPYLOS_WORK/bench-runs, KAMPYLOS_WORK default ~/kampylos-work), RUN_PREFIX (e.g.
#        "taskpolicy -b" or "nice -n 19"), KAMPYLOS_STUB_NCPUS (overrides the profile's ncpus), KAMPYLOS_SRC (where a
#        relative photometry path is looked up; default this repository). A label may name a wrapper script that sets
#        environment variables and execs the binary.
#   macOS and Linux (GNU time); the GPU-utilisation column only on macOS and only roughly (whole GPU, sampled every 0.5 s)
# Output: one line per run (label, round, wall s, user s, sys s, max RSS MB, exit code, hash of the result lines, mean GPU
# busy %, chi2 of the first cell), then the summary of summarize_bench.py.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
repo=${KAMPYLOS_SRC:-$(cd "$here/../.." && pwd)}
[ $# -ge 3 ] || { sed -n 2,12p "$0"; exit 2; }
PROFILE=$1; ROUNDS=$2; shift 2
WU=$PROFILE; [ -f "$WU" ] || WU=$here/profiles/$PROFILE.wu
[ -f "$WU" ] || { echo "no profile $PROFILE"; exit 2; }
PHOT=$(awk '$1=="photometry"{print $2}' "$WU"); case "$PHOT" in /*) ;; *) PHOT=$repo/$PHOT ;; esac
INLINE=$(awk '$1=="in"{sub(/^in[ \t]+/, ""); print}' "$WU")
NCPUS=${KAMPYLOS_STUB_NCPUS:-$(awk '$1=="ncpus"{print $2}' "$WU")}; NCPUS=${NCPUS:-1}
[ -f "$PHOT" ] && [ -n "$INLINE" ] || { echo "profile $WU needs 'photometry <file>' and 'in <line>'"; exit 2; }
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${BENCH_OUT:-${KAMPYLOS_WORK:-$HOME/kampylos-work}/bench-runs}/$STAMP-$(basename "$WU" .wu)
mkdir -p "$OUT"
RESULTS=$OUT/results.tsv
if [ "$(uname)" = Darwin ]; then
  TIMECMD=(/usr/bin/time -l); RSSDIV=1048576                                   # bytes
else
  TIMECMD=(/usr/bin/time -f "%e real %U user %S sys
%M maximum resident set size"); RSSDIV=1024                                    # GNU time, KiB
fi
# shasum comes with perl, which minimal Linux images lack; sha1sum gives the same hash
HASH=shasum; command -v shasum > /dev/null || HASH=sha1sum
load() { uptime 2> /dev/null | sed 's/.*load/load/' || true; [ -r /proc/loadavg ] && echo "loadavg $(cut -d' ' -f1-3 /proc/loadavg)"; }
{
  echo "# $(date '+%Y-%m-%d %H:%M:%S') profile $WU rounds $ROUNDS ncpus $NCPUS host $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
  echo "# in: $INLINE   photometry: $PHOT"
  echo "# load: $(load | head -1)"
  for lb in "$@"; do echo "# ${lb%%=*} = ${lb#*=} md5 $(md5 -q "${lb#*=}" 2>/dev/null || md5sum "${lb#*=}" | cut -d' ' -f1)"; done
} | tee "$RESULTS"

run_one() { # label binary round
  local label=$1 bin=$2 round=$3 dir=$OUT/$1-$3
  mkdir -p "$dir"
  cp "$PHOT" "$dir/photometry"
  echo "$INLINE" > "$dir/in"
  local sampler=""
  if command -v ioreg > /dev/null; then
    ( while :; do ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; sleep 0.5; done ) > "$dir/gpu_util.txt" 2> /dev/null &
    sampler=$!
  fi
  ( cd "$dir" && KAMPYLOS_STUB_NCPUS=$NCPUS "${TIMECMD[@]}" ${RUN_PREFIX:-} sh -c 'exec "$@" > stdout.txt 2> stderr.txt' sh "$bin" ) 2> "$dir/time.txt"
  echo $? > "$dir/exit_code"
  [ -n "$sampler" ] && { kill $sampler; wait $sampler; } 2> /dev/null
  local real user sys rss gpu=- h=none chi2=-
  real=$(awk '/ real /{print $1}' "$dir/time.txt"); user=$(awk '/ real /{print $3}' "$dir/time.txt"); sys=$(awk '/ real /{print $5}' "$dir/time.txt")
  rss=$(awk -v d=$RSSDIV '/maximum resident set size/{printf "%.0f", $1/d}' "$dir/time.txt")
  [ -s "$dir/gpu_util.txt" ] && gpu=$(awk '{s+=$1; n++} END {printf "%.0f", s/n}' "$dir/gpu_util.txt")
  if [ -f "$dir/out" ]; then
    # without CRs: a Windows build writes CRLF (KAM-023), which is not a difference in the results
    h=$(grep -v "^#" "$dir/out" | tr -d '\r' | $HASH | cut -c1-12)
    chi2=$(grep -v '^#' "$dir/out" | awk 'NR==1{print $3}')
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$round" "$real" "$user" "$sys" "$rss" "$(cat "$dir/exit_code")" "$h" "$gpu" "$chi2" | tee -a "$RESULTS"
  rm -f "$dir/photometry" "$dir"/kampylos_*checkpoint "$dir/stub_checkpoints"
}
for r in $(seq 1 "$ROUNDS"); do
  for lb in "$@"; do run_one "${lb%%=*}" "${lb#*=}" "$r"; done
done
python3 -B "$here/summarize_bench.py" "$RESULTS" | tee "$OUT/summary.txt"
