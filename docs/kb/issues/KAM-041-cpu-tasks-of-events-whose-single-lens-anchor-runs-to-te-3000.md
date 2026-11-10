---
id: KAM-041
title: CPU tasks of events whose single-lens anchor runs to tE = 3000 d end at the client's elapsed time limit
type: bug
status: confirmed
severity: high
platforms: [all]
area: boinc-lifecycle
action: decide
next: Decide (maintainers, work generator): a CPU bound far above 40x the estimate, an estimate per event (points, anchor at the tE bound), or such events on GPU, mt or more seed parts
blocked_by: []
pattern: PAT-034
aliases: []
reported: 2026-11-10
source: task pages 8838721, 8838769, 8838705, 8831870 (sent 2026-11-07/08); the BOINC client's job and event log of an Apple M4 Pro host (2026-10-06 to 2026-11-09)
where: the work generator (not in this repository): rsc_fpops_est 3e13 and rsc_fpops_bound 1.2e15 for every cpu work unit; src/single_lens.h:245, src/binary_fit.h:253-296 @ 2b01b38
fixed_in:
---

## Symptom

Four task pages of 2026-11-08 to 2026-11-10, app version v0.12 (`kampylos_v12_*`), all "Computation error", exit
status 197 (0xC5) `EXIT_TIME_LIMIT_EXCEEDED`, on four hosts and three platforms:

| task | work unit name | platform, client | message |
|---|---|---|---|
| 8838769 | `kampylos_ev14975_cpu_row2_q6_g10_k5p4_...` | arm64-apple-darwin, macOS 27.0, 8.2.11 | exceeded elapsed time limit 7727.36 (1200000.00G/155.29G) |
| 8838721 | `kampylos_ev14975_cpu_row2_q2_g10_k5p0_...` | arm64-apple-darwin, macOS 15.7.4, 8.2.11 | exceeded elapsed time limit 13799.82 (1200000.00G/86.96G) |
| 8838705 | `kampylos_ev14975_cpu_row2_q0_g10_k5p4_...` | x86_64-apple-darwin, macOS 12.6, 8.2.1 | exceeded elapsed time limit 74770.70 (1200000.00G/16.05G) |
| 8831870 | `kampylos_ev14966_cpu_row2_q1_g10_k5p3_...` | windows_x86_64, 8.2.11 | exceeded elapsed time limit 152562.94 (1200000.00G/7.87G) |

The stderr is the same in all four: the event loads, the single-lens anchor runs into the tE bound, the task starts
its one cell, and nothing more is written before the abort:

```
Loaded 3733 data points; grid 1x1 (log10), log10_s [-0.6000,-0.6000], log10_q [-1.6667,-1.6667]
Single-lens anchor: PSPL chi2=19269.289 FSPL chi2=19246.660 FSPL+LD chi2=19246.660 -> best=fspl t0=2455606.8825 u0=0.00152 tE=3000.0000 rho=1.619e-03 a1=0.000 chi2/dof=5.164
Resuming from cell 0/1
```

(ev14966: 1457 points, `best=fspl t0=2455735.3638 u0=0.00561 tE=3000.0000 rho=5.657e-03`, chi2/dof 2.421.) After
that the macOS tasks show "Crashed executable name: kampylos_v12_macos_..." and an `atos` line, the Windows task
"Unhandled Exception Detected ... Breakpoint Encountered (0x80000003)" and a debugger dump. These are the BOINC
library's reaction to the client's abort, not a crash of the app (shared/boinc-api.md, "Resource limits and aborts").
Valid tasks of other events were reported by the same hosts at the same time.

On the Apple M4 Pro host (BOINC 8.2.11, macOS 27.0; task 8838769 is one of its), the event log has 10 such aborts
between 2026-11-05 and 2026-11-08 and no other Kampylos error since 2026-10-26: 8 tasks of ev14966, 1 of ev14975 and
1 of ev14926. Every task of ev14966 and ev14975 the host received was aborted, none finished. Its 2120 valid CPU tasks
with the same estimate (job log, 2026-10-06 to 2026-11-09) took 129 s (median), 364 s (90th percentile), 18283 s at
most (2026-10-17); the limit on 2026-11-05 was 10558.79 s at 114.51G, on 2026-11-07 and 08 7307-7395 s at
162-164G.

## Cause

Verified from the task pages, the host's `client_state.xml`, job and event logs, and the BOINC source (boinc-api.md,
commit fc12d39); the cost of these cells was not measured here.

* **The bound.** Every CPU work unit (one cell, one of five seed parts) carries `rsc_fpops_est` 3e13 and
  `rsc_fpops_bound` 1.2e15, 40 times the estimate. The client's limit is `rsc_fpops_bound` over the host's learned
  speed, and that speed comes from the host's valid tasks, so the limit is 40 times the host's average task on every
  host: 2.1 h on the M4, 42.4 h on the slowest Windows host above (PAT-034). The more short tasks a host has done, the
  shorter the limit: on the M4 a task of 18283 s was valid on 2026-10-17, while on 2026-11-07 the limit was 7370 s.
  The aborted tasks never enter the average, and the work units are sent again and fail again.
* **The cost.** Both events' single-lens anchors end at the upper bound `tE_max = min(5 span, 3000 d)`
  (`src/single_lens.h:245`) with u0 of 1.5e-3 and 5.6e-3: the degenerate fit that the comment at `single_lens.h:153-158`
  describes (OGLE-2017-BLG-1522, "tE -> 3000 d for a 7.5 d event"). The seed families (A) and (C) of
  `kampylos_make_seeds()` start from that anchor, (C) at tE 3000 d / sqrt(mass fraction) or more
  (`src/binary_fit.h:253-296`). The same regime cost 24-50 minutes per seed and 96-199 CPU hours per task for ev25055
  (anchor u0 1.2e-12, tE 1207 d, docs/BENCHMARK.md, "Measurements of 2026-10-16" §4.2). So these work units cost more
  than 40 average tasks; how much more was not measured.
* **Not the platform.** The Windows task failed the same way as the three macOS ones, with the same ratio
  (1200000/7.87 against 1200000/155.29 is the same 40 average tasks).
* **Not fixed by the branch of 2026-11-05 (PR #4, 2b01b38).** Its same-results changes make a CPU task 1.09-1.24x
  faster (`--pool`, KAM-036); the threaded CPU app (KAM-040) shortens the wall time, but the scheduler learns that
  speed too, so the ratio stays 40; the Nelder-Mead stopping rules (KAM-037, a decision) save about 2x. None of them
  brings a task that costs more than 40 average tasks under the limit, and the app cannot raise its own limit (a
  restart keeps the elapsed time).

## Fix

Proposed, for the maintainers; the work generator and the server's settings are not in this repository.

1. At once: give CPU work units a bound far above the slowest cells, for example 1000x the estimate (3e16); the "fast"
   GPU work units already carry 5e17 for an estimate of 1.8e15 (278x). The 4-day deadline then limits the slow tasks.
   A task that really hangs would run to the deadline, but Kampylos tasks are slow, not hung.
2. Better: an estimate per event. The generator can compute the single-lens anchor before it makes the work units
   (seconds per event): an anchor at `tE_max`, or `n_peak` close to the number of points, marks an expensive event;
   its estimate (and bound) can then scale with the number of points and that class, which also fixes the runtime
   estimates and the credit of those tasks (forum thread "KAMPYLOS", 2026-11-08).
3. Send such events only to the GPU apps (since 2026-10-19 they finish overflowing candidates on the CPU, so they
   cost about what the CPU app does, boinc/README.md), to an `mt` CPU app version (KAM-040), or in more seed parts
   than 5.
4. Science, to measure first behind a switch: a cell whose anchor is at the tE bound could be seeded without the
   families that start there (the comment at `single_lens.h:155-158` calls that anchor "a useless starting point").
   It changes results, so it is a decision.

## Test

After the change, the client's event log shows no "exceeded elapsed time limit" for these events and their work units
complete (ev14966, ev14975). To size the bound: run one cpu task of such an event outside BOINC with the stub
(`tests/boinc_stub`) and time it; not done here, because the clients delete a task's input files once it is reported.

## Log

- 2026-11-10: created from the task pages and the M4 host's logs; confirmed by reading the client and scheduler code
  (fc12d39).
