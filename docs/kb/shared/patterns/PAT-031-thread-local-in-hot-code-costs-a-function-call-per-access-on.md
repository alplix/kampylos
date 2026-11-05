---
id: PAT-031
title: thread_local in hot code costs a function call per access on macOS and with emulated TLS (MinGW-w64 GCC, old Android)
status: active
severity: low
platforms: [macos, windows, mingw, android]
area: performance
applies_to: code that reads or writes thread_local (or __thread) variables inside a hot loop, in the app or in a vendored library
check: 'thread_local|__thread|_Thread_local' -- src vendor boinc
origin: kampylos docs/BENCHMARK.md (VBMicrolensing, 2026-10-16 and 2026-11-04)
reference_fix: kampylos vendor/patch_vbm_thread_local.py (--strip for a single-threaded build, --block for one lookup per function)
added: 2026-11-04
---

## The defect

Wasted time, no wrong result. A `thread_local` variable is not a plain memory access everywhere. On macOS every access
is a call through the variable's TLS descriptor (`_tlv_get_addr`); with emulated TLS (MinGW-w64 GCC, at least up to
version 16.2, and the Android NDK below API 29) every access is a call to `__emutls_get_address`. Code that touches
such variables in its innermost loop then spends a large share of its time in those calls. In kampylos the
`thread_local` statics of VBMicrolensing took 46 % of the CPU app's samples on an M1 with macOS 15, and the emulated
form made the app 1.2-1.3x slower than native macOS TLS.

## Why it happens

The cost is the platform's TLS model, not the code: on Linux a `thread_local` in the executable is one load relative
to the thread pointer and costs nearly nothing. The cost also depends on the OS version and CPU: on an M4 Pro with macOS
27 the same macOS calls were cheap (gathering them gained nothing there, while it gained 1.20-1.30x on the M1). A
measurement on one Mac does not settle it for volunteers' older Macs.

## How to check a repository

```
git grep -nE 'thread_local|__thread|_Thread_local' -- src vendor boinc
```

No hit: `n/a`. Hits outside hot code (set once per task, read per work unit): `not-affected`, with the place. Hits in
a loop that runs per point, per evaluation or per step: sample a run on macOS (`sample <pid> 10`) or build with
`-femulated-tls` and compare; `affected` when the TLS calls are a visible share (a few percent or more). Record the
platforms that build with emulated TLS (MinGW-w64 GCC, Android NDK below API 29) in the issue.

## How to fix

If the build is single-threaded, use plain statics (kampylos `--strip`). If threads need their own copies, gather each
function's thread-local state into one struct so a call pays one lookup instead of one per access (kampylos `--block`).
Both keep the same values and give byte-identical results. Keep threads that hold `thread_local` objects alive for the
process (PAT-020).

## How to test

The output must stay byte-identical (`cmp` of the result files). Time it with the repository's benchmark on macOS and
with `-femulated-tls` (the MinGW-w64 model), interleaved, by the rule of its docs/BENCHMARK.md.
