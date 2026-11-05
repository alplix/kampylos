---
id: KAM-036
title: VBMicrolensing's small objects go through malloc and free for every point and contour sample
type: bug
status: fix-proposed
severity: low
platforms: [all]
area: performance
action:
next: Release builds with vendor/patch_vbm_thread_local.py --pool (free lists, same results): 1.09-1.10x with jemalloc, 1.18-1.24x without it
blocked_by: []
pattern: PAT-032
aliases: []
reported: 2026-11-05
source: M4 benchmark 2026-11-05 (docs/BENCHMARK.md)
where: vendor/patch_vbm_thread_local.py @ fa03bd1
fixed_in:
---

## Symptom

Wasted time, no wrong result. Sampled on an Apple M4 Pro (macOS 27, Apple clang 21, fast CPU build with jemalloc, the
whole planet cell, 2026-11-05): `operator new` and `operator delete` take 5.5 % of the CPU app's samples and jemalloc's
thread lookup (`pthread_getspecific`) another 1.5 %; with the macOS system allocator the app is 1.13x slower still.

## Cause

`BinaryMag2` creates and deletes an image list for every point, and the contour integration a `_point` per contour
sample and image, `_theta`s, `_curve`s, skip lists and annuli (`vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp`,
16 `new` and 20 `delete` sites in lines 717-2450, and `_skiplist_curve::append`). Every one goes to the allocator.

## Fix

Proposed on 2026-11-05: `vendor/patch_vbm_thread_local.py --pool` adds a class-level `operator new`/`delete` with a free
list to the seven small classes (`_point`, `_curve`, `_skiplist_curve`, `_theta`, `_thetas`, `annulus`,
`_sols_for_skiplist_curve`): plain statics with `--strip`, `thread_local` in the thread-safe forms. Allocation does not
touch the arithmetic: byte-identical result lines in every pair measured. Measured on the M4 Pro (docs/BENCHMARK.md,
Results on an Apple M4 (2026-11-05), 3 interleaved rounds): CPU app 1.10x with jemalloc, 1.22x (`planet`) and 1.24x
(`ogle`) with the system allocator, which then is as fast as jemalloc; 1.18x for the `--block` library with emulated
TLS (the MinGW-w64 model); Metal app 1.09x. Not measured: the Windows heap, Linux, MinGW under Wine.

## Test

`tests/test_vbm_threads.cpp` with the `--block --pool` library (4 threads, passes); the same result lines as without
`--pool` (`scripts/bench/run_bench.sh` hash column).

## Log

- 2026-11-05: created.
- 2026-11-05: 2026-11-05: Linux arm64 container, GCC 12, glibc: free lists 1.16x against plain statics (jemalloc 1.10x), same output
