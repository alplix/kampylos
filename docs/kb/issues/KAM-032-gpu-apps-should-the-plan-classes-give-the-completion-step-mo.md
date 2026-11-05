---
id: KAM-032
title: GPU apps: should the plan classes give the completion step more CPU (avg_ncpus)?
type: question
status: new
severity: info
platforms: [metal, cuda, opencl]
area: performance
action:
next: Decide (maintainers) the GPU plan classes' avg_ncpus: 2-8 completion threads make caustic-crossing cells 1.7-3.4x faster for 3-9 % more CPU time
blocked_by: []
pattern:
aliases: []
reported: 2026-11-04
source: M4 benchmark 2026-11-04 (docs/BENCHMARK.md, More completion threads)
where: src/kampylos_cpu_patch_pool.h @ faba573
fixed_in:
---

## Symptom

A question for the maintainers, with measurements. On a cell where the source crosses a caustic, a GPU task spends most
of its time completing finite-source points on the CPU (Metal on the M1: GPU 13 s, CPU completion 154 s of a 172 s
run), and today's plan classes (`avg_ncpus` 0.1-0.25) give that step one thread.

## Cause

The completion pool (`src/kampylos_cpu_patch_pool.h`) has as many threads as the client's CPU budget (`init_data`
ncpus, rounded down, at least 1; since 66a1ab3), and the client sets that budget from the app version's `avg_ncpus`.
Measured on 2026-11-04 on an Apple M4 Pro (macOS 27, this branch's Metal build, profile `planet-gpu`, 60 seeds of the
planet cell, 3 interleaved runs each, docs/BENCHMARK.md "More completion threads"):

| threads | median s | CPU s | speed-up |
|---|---|---|---|
| 1 | 63.02 | 55.6 | 1.00x |
| 2 | 37.96 | 57.3 | 1.66x |
| 4 | 25.03 | 60.4 | 2.52x |
| 8 | 18.66 | 60.6 | 3.38x |

The CPU time grows by 3-9 %, so the same work finishes much sooner for nearly the same CPU. The result lines differ
between thread counts at rounding level (KAM-012), as they already do between hosts.

## Fix

Questions: should the GPU plan classes declare more CPU (for example `avg_ncpus` 2 or 4), so the client gives the task
that budget and schedules one CPU task less? Is the rounding-level difference between thread counts acceptable to the
validator (KAM-025)? No app change is needed: the app already follows the budget. CUDA and OpenCL use the same
completion step and should scale alike (not measured).

## Test

`KAMPYLOS_STUB_NCPUS=<n> scripts/bench/run_bench.sh planet-gpu 3 metal=<Metal build>` for n = 1, 2, 4, 8.

## Log

- 2026-11-04: created.
- 2026-11-04: thread scaling measured on an Apple M4 Pro (docs/BENCHMARK.md).
