---
id: KAM-040
title: CPU app: should an mt plan class run the seeds of a cell on several threads?
type: question
status: new
severity: info
platforms: [all]
area: performance
action:
next: Decide (maintainers) whether to offer a threaded CPU app version (mt plan class, avg_ncpus N): same result lines, tasks 1.9-6.5x shorter at 2-8 threads for 3-10 % more CPU
blocked_by: [KAM-030]
pattern:
aliases: []
reported: 2026-11-05
source: M4 benchmark 2026-11-05 (docs/BENCHMARK.md, Threaded CPU app)
where: src/kampylos_cpu_seed_pool.h @ fa03bd1
fixed_in:
---

## Symptom

A question for the maintainers, with measurements. A CPU task is often one cell, and a cell takes minutes on an ordinary
event and days on an event whose single-lens anchor has no baseline (96-199 CPU hours per task, docs/BENCHMARK.md,
"Measurements of 2026-10-16" §4.2); today's CPU app fits one seed at a time whatever the host has.

## Cause

Since 2026-11-05 the CPU app can fit the seeds of a cell on as many threads as the client budgets CPUs for the task
(`src/kampylos_cpu_seed_pool.h`, `boinc/main_kampylos_boinc.cpp`): each thread takes the next seed when it is free,
and the fits are merged in seed order, so progress, seed checkpoints and the result are those of one thread. That
holds only with history-free evaluations (KAM-030, `-DKAMPYLOS_RESET_STATE=1`) and the thread-safe library (the fork or
`--block`); other builds keep one thread. Measured on an Apple M4 Pro (macOS 27, whole planet cell, one run each,
docs/BENCHMARK.md "Threaded CPU app"): 312 s with today's single-thread form (`--strip`), 350 s with the thread-safe
form on one thread, 183, 96 and 54 s on 2, 4 and 8 threads, every run with the same result line, CPU time 349-384 s.

## Fix

Questions: should the server offer a threaded CPU app version under an `mt` plan class (`avg_ncpus` N, e.g. 4), at
least for the work units expected to be long (no baseline, many points)? The client then schedules it on N cores, and
the volunteer gets the result N times sooner for about the same CPU (the thread-safe library costs 12 % on one thread
here; on older Macs `--block` costs more, see KAM-033). No app change is needed beyond building it with the switch.

## Test

`KAMPYLOS_STUB_NCPUS=<n> scripts/bench/run_bench.sh planet-cell 1 ...` with a build made by
`VBM=block VBMX="reset pool" OPT="-O2 -DKAMPYLOS_RESET_STATE=1" scripts/bench/build_variant.sh mt`; the kill-and-resume
test with 4 threads: `CXX="clang++ -DKAMPYLOS_RESET_STATE=1" KAMPYLOS_STUB_NCPUS=4 sh tests/test_boinc_resume.sh` in
that build's copy of the tree.

## Log

- 2026-11-05: created.
