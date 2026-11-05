---
id: KAM-034
title: The macOS system allocator takes a large share of VBMicrolensing's time
type: gotcha
status: documented
severity: low
platforms: [macos]
area: performance
action:
next: None: link a static jemalloc (boinc/README.md); measure mimalloc and the Windows heap on Windows
blocked_by: []
pattern: PAT-032
aliases: []
reported: 2026-11-04
source: 2026-10-16 measurements and M4 benchmark 2026-11-04 (docs/BENCHMARK.md)
where: boinc/README.md (Faster CPU build) @ faba573
fixed_in:
---

## Symptom

Wasted time, no wrong result. About a third of the CPU app's time went to the macOS system allocator on an M1
(2026-10-16); a static jemalloc made the CPU app 1.13x faster on an M4 Pro with macOS 27, and the Metal app's
completion step faster as well (2026-11-04, docs/BENCHMARK.md).

## Cause

VBMicrolensing creates and deletes many small objects per magnification, many more inside the finite-source
integration (PAT-032).

## Fix

Link a static jemalloc (`libjemalloc.a`) or mimalloc; no source change, byte-identical results (boinc/README.md
"Faster CPU build", "Faster GPU app build"). With glibc it gained 10-11 % on a quiet machine (Linux arm64 container,
2026-11-04). The Windows heap and mimalloc on Windows are unmeasured.

## Test

`scripts/bench/mac_ab.sh` (cpu-fast against cpu-doc), `scripts/bench/linux_ab.sh` (strip-jem, strip-mi).

## Log

- 2026-11-04: recorded with pattern PAT-032.
