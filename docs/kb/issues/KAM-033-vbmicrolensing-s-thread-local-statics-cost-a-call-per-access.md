---
id: KAM-033
title: VBMicrolensing's thread_local statics cost a call per access on macOS and with emulated TLS
type: gotcha
status: documented
severity: low
platforms: [macos, mingw, android]
area: performance
action:
next: None: the build options --strip (CPU app) and --block (GPU apps) of vendor/patch_vbm_thread_local.py are the workaround
blocked_by: []
pattern: PAT-031
aliases: []
reported: 2026-11-04
source: 2026-10-16 measurements and M4 benchmark 2026-11-04 (docs/BENCHMARK.md)
where: vendor/patch_vbm_thread_local.py @ faba573
fixed_in:
---

## Symptom

Wasted time, no wrong result. On an M1 with macOS 15 the CPU app spent 46 % of its samples in `_tlv_get_addr`
(2026-10-16 measurements, docs/BENCHMARK.md); built with `-femulated-tls` (the model of MinGW-w64 GCC and of the
Android NDK below API 29) it was 1.2-1.3x slower still.

## Cause

The fork of VBMicrolensing makes its 577 function-local statics `thread_local` so the GPU apps' completion threads can
share the library. Every access is a call on macOS and with emulated TLS (PAT-031). On an M4 Pro with macOS 27 the
calls were cheap: plain statics gained 7 % and `--block` nothing (2026-11-04).

## Fix

Build options, no source change: `vendor/patch_vbm_thread_local.py --strip` for the single-threaded CPU app,
`--block` for the GPU apps (one TLS lookup per function call). Both give byte-identical result lines (docs/BENCHMARK.md,
boinc/README.md). MinGW-w64 GCC 16.2 still emulates TLS (checked 2026-11-04), so the Windows builds gain most.

## Test

`scripts/bench/mac_ab.sh` and the `-femulated-tls` builds of docs/BENCHMARK.md; `scripts/bench/cross_check.sh` builds and
runs the three library forms for Windows.

## Log

- 2026-11-04: recorded with pattern PAT-031 from the measurements of 2026-10-16 and 2026-11-04.
