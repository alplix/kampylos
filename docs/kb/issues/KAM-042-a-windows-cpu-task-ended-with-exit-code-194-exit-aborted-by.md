---
id: KAM-042
title: A Windows CPU task ended with exit code 194 (EXIT_ABORTED_BY_CLIENT) that the client had not asked for
type: bug
status: new
severity: low
platforms: [windows, boinc-client]
area: boinc-lifecycle
action: verify
next: Look at the host's event log around the end of task 8830679 and at the other tasks of its work unit; count exit-194 errors of other apps on Windows hosts
blocked_by: []
pattern:
aliases: []
reported: 2026-11-10
source: task page 8830679 (sent 2026-11-07); maintainer's first guess (2026-11-10): the Windows version
where: not in the app: boinc/main_kampylos_boinc.cpp @ 2b01b38 exits only through boinc_finish(0|1)
fixed_in:
---

## Symptom

Task 8830679, `kampylos_ev14964_cpu_row2_q7_g10_k5p3_...`, windows_x86_64, app v0.12, BOINC 8.2.11: sent
2026-11-07 18:54 UTC, reported 20:29 UTC, "Computation error", exit status 194 (0xC2) `EXIT_ABORTED_BY_CLIENT`, peak
working set 10.68 MB. The message is Windows' text for error 194 with a garbled insert:

```
<message>
The operating system cannot run L;0��.
 (0xc2) - exit code 194 (0xc2)</message>
<stderr_txt>
Loaded 127 data points; grid 1x1 (log10), log10_s [-0.6000,-0.6000], log10_q [-1.1111,-1.1111]
Single-lens anchor: PSPL chi2=1103.912 FSPL chi2=1097.563 FSPL+LD chi2=1097.269 -> best=fspl_ld t0=2455713.3488 u0=0.01447 tE=69.9737 rho=3.671e-02 a1=1.000 chi2/dof=9.144
  fixed-tE PSPL profile point (extra binary seeds): t0=2455713.5418 u0=0.03869 tE=30.0000 chi2=1122.139
  fixed-tE PSPL profile point (extra binary seeds): t0=2455713.2979 u0=0.17749 tE=10.0000 chi2=1230.913
Resuming from cell 0/1

</stderr_txt>
```

## Cause

Not known. What was checked (2026-11-10):

* The app never exits with 194: the CPU main ends with `boinc_finish(0)` or `boinc_finish(1)` (`git grep` of
  `boinc/` and `src/` at 2b01b38 and at ee44a13).
* 194 comes from the BOINC library after an abort message, or from the Windows client's `TerminateProcess` when an
  abort is not obeyed in 60 s and when it cleans up the recorded descendants and other processes of a task that has
  exited (shared/boinc-api.md, "Resource limits and aborts"). When the client asks for an abort it reports its own
  reason, as in the time-limit aborts of KAM-041. Here it reported only the exit code, so it saw the process end with
  194 while it still counted it as running.
* No library trace in stderr: an abort message outside a critical section makes the Windows library print a debugger
  dump first (task 8831870, KAM-041). That fits a `TerminateProcess` from outside better than the library's own
  abort path. Who did it is a guess: another BOINC process or a tool on the host, or the client's clean-up of another
  task's descendants hitting a reused process id. Not checked.
* Not the memory bound (10 MB here, peak working set 10.68 MB): the client does not enforce `rsc_memory_bound`.
* Not KAM-041: 127 points, an ordinary anchor (tE 70 d).

## Fix

None yet. If the host's event log shows another task's exit or an abort time-out at that moment, the cause is in the
client (then a report to BOINC); if the same work unit was valid on other hosts, nothing in the app needs to change.

## Test

None.

## Log

- 2026-11-10: created from the task page.
