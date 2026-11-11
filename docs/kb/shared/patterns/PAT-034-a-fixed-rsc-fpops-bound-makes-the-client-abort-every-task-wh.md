---
id: PAT-034
title: A fixed rsc_fpops_bound makes the client abort every task whose input costs much more than the average
status: active
severity: high
platforms: [all]
area: boinc-lifecycle
applies_to: work generators and work-unit templates that give every work unit of an app the same rsc_fpops_est and rsc_fpops_bound, for apps whose task cost depends on the input data
check: 'rsc_fpops_bound|rsc_fpops_est|fpops_bound|fpops_est'
origin: kampylos KAM-041
reference_fix:
added: 2026-11-10
---

## The defect

Every work unit of an app gets the same `rsc_fpops_est` and the same `rsc_fpops_bound`, but some inputs cost far more
than the average: an event with ten times the data points, a degenerate fit that never converges quickly, a search
whose space depends on the data. The client aborts those tasks with exit status 197, `EXIT_TIME_LIMIT_EXCEEDED`,
"exceeded elapsed time limit L (boundG/flopsG)", after hours of work, on every host they are sent to. The volunteer
sees a computation error (on Windows with a debugger dump, on macOS with "Crashed executable name", which are the
library's reaction to the abort, not a crash). Tasks of other work units sent at the same time are valid, so it
looks like a random or host-specific fault. The work unit is reissued, fails again and, after the server's error
limit, is never computed.

## Why it happens

The client's limit is `rsc_fpops_bound / flops`, and `flops` is the scheduler's speed for this app version on this
host, learned from the host's own validated tasks as 1 / (average of elapsed time / `rsc_fpops_est`)
([boinc-api.md](../boinc-api.md#resource-limits-and-aborts)). With one `rsc_fpops_est` for all work units, the limit
is therefore `rsc_fpops_bound / rsc_fpops_est` times the host's average task: a fast host gets a short limit, a slow
host a long one, and a task that costs more than that ratio of average tasks fails on both. The failed tasks never
enter the average, so nothing corrects it; the more short tasks a host has done, the tighter it gets.

## How to check a repository

1. Find where the work units get their bounds: `git grep -nE 'rsc_fpops_bound|rsc_fpops_est|fpops_bound|fpops_est'`
   (work-unit templates, `create_work` calls, work generators). If the generator is not in the repository, say so
   in Evidence and look at the documented task lengths instead.
2. Find the ratio `rsc_fpops_bound / rsc_fpops_est` (or `delay_bound` if the bound is huge) and whether either
   depends on the input (size, kind of event, number of points, parts).
3. Compare with the measured spread of task cost for the same estimate (`docs/BENCHMARK.md`, the BOINC client's
   `job_log_<project>.txt`, field `et`, per input): the slowest valid task over the median.
4. Look for real reports: "exceeded elapsed time limit" in task pages or a client's event log, exit status 197.

* `affected`: one estimate for inputs whose cost spread (slowest / median) is near or above the bound ratio, or
  reports of 197 aborts; give the ratio, the spread and a task id in Evidence.
* `not-affected`: the estimate and bound scale with the input, or the bound ratio is well above the measured spread
  (give both).
* `n/a`: no work units with a CPU-intensive app (a non-CPU-intensive app is exempt from the limit), or the repository
  has neither a generator nor any task-length data; say which.

## How to fix

In the work generator, not in the app: the app cannot raise its limit, and a restart keeps the elapsed time.

* Scale `rsc_fpops_est` with what drives the cost (points, cells, parts, a cheap pre-check of the input) and keep
  `rsc_fpops_bound` a large multiple of that estimate, at least the measured slowest / median of the same class.
* Where the cost cannot be predicted, set `rsc_fpops_bound` far above the estimate (hundreds of times) and let the
  deadline limit the task; a task that really hangs then costs one deadline instead of every long task being lost.
* Send the inputs known to be expensive to the hosts and app versions that can finish them (GPU, multi-threaded), or
  split them into more work units of the same kind.

## How to test

From a client's `job_log_<project>.txt`: the median elapsed time of the app's tasks on that host times
`rsc_fpops_bound / rsc_fpops_est` is the limit; every input class whose tasks there need more must be gone after the
fix (the event log shows no "exceeded elapsed time limit" for them). On the server: the error rate of the long input
class before and after, and that its work units complete.
