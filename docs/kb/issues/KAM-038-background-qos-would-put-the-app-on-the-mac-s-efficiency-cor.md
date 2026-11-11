---
id: KAM-038
title: Background QoS would put the app on the Mac's efficiency cores (3.8x slower); BOINC 8.2.11 lowers the priority with nice
type: gotcha
status: documented
severity: info
platforms: [macos]
area: performance
action:
next:
blocked_by: []
pattern:
aliases: []
reported: 2026-11-05
source: M4 benchmark 2026-11-05 (docs/BENCHMARK.md)
where: docs/BENCHMARK.md @ fa03bd1
fixed_in:
---

## Symptom

None in the apps today. On Apple Silicon a process with background QoS (`taskpolicy -b`, `PRIO_DARWIN_BG`) runs on the
efficiency cores only: the fast CPU app took 51.69 s on `planet` that way against 13.75 s plain and 13.85 s with
`nice -n 19` (Apple M4 Pro, macOS 27, 3 interleaved rounds, 2026-11-05).

## Cause

macOS schedules background-QoS work on the efficiency cores; a niced process keeps the performance cores. The BOINC
client 8.2.11 for macOS arm64 (`/Library/Application Support/BOINCManager.app/Contents/Resources/boinc`) imports
`setpriority()` and no QoS call, and its running task had nice 10 on 2026-11-05: it lowers an app's priority with nice,
so volunteers' tasks run on the performance cores and the speed numbers in docs/BENCHMARK.md apply to them. Read from
the binary's imports and the task's `ps` line, not from the client's source.

## Fix

Nothing to change. Keep it in mind if the client changes how it lowers priority, or if an app is tempted to set its own
QoS class: background QoS would make every task 3-4x slower on a Mac with efficiency cores.

## Test

`RUN_PREFIX` or wrapper labels in `scripts/bench/run_bench.sh`: `plain`, `nice -n 19`, `taskpolicy -b` on one profile.

## Log

- 2026-11-05: created.
