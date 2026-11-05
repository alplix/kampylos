---
id: KAM-037
title: Decide the Nelder-Mead stopping rules: about 2x less CPU on whole cells with the same chi2 once evaluations are history-free
type: question
status: new
severity: info
platforms: [all]
area: science
action:
next: Decide (maintainers) on nm-stopping.patch's rules (restart after a gain > 1e-3, stop after 100 iterations without 1e-4), after the regression suite's real events
blocked_by: []
pattern:
aliases: []
reported: 2026-11-05
source: M4 benchmark 2026-11-05 (docs/BENCHMARK.md)
where: scripts/bench/patches/nm-stopping.patch @ fa03bd1
fixed_in:
---

## Symptom

A question for the maintainers, with measurements. 71 % of the objective evaluations are in the two restart passes and
a pass often runs to the 400-iteration cap while gaining nothing (docs/BENCHMARK.md, M1 2026-11-04). Two rules cut
that: restart again only after a restart gained more than 1e-3, and stop a pass whose best chi2 has not improved by
1e-4 in 100 iterations (`scripts/bench/patches/nm-stopping.patch`, switches off by default).

## Cause

Measured on an Apple M4 Pro on 2026-11-05 with history-free evaluations on (KAM-030; docs/BENCHMARK.md, "Search changes
on whole cells"), whole cells of 240 seeds, fast CPU build:

| rules | planet cell | OGLE cell | far cell |
|---|---|---|---|
| none | 326 s, chi2 411.2228 | 169 s, chi2 2810.4678 | 247 s, chi2 714.1706 |
| restart rule | 271 s, same line | 138 s, same line | 213 s, same line |
| restart and stall rules | 168 s, same line | 76 s, same line | 122 s, same chi2, other digits |

Without the reset (M1, 2026-11-04) the same rules moved the OGLE cell by -0.94 and a 10-seed profile by +21.6, through
the state one seed leaves the next (KAM-030).

## Fix

Questions: may the next app versions use the two rules (a change of the search, so of some results), and which
thresholds? Before deciding, run the regression suite's real events (`tests/regression/run_regression.py --suite real`,
which needs the server's `kampylos_analyze.py`) with and without the rules.

## Test

`OPT="-O2 -DKAMPYLOS_RESET_STATE=1 -DKAMPYLOS_RESTART_MIN_GAIN=1e-3 -DKAMPYLOS_NM_STALL_ITERS=100 -DKAMPYLOS_NM_STALL_TOL=1e-4"
VBM=strip VBMX=reset ALLOC=jemalloc scripts/bench/build_variant.sh st scripts/bench/patches/nm-stopping.patch`, then
`scripts/bench/run_bench.sh planet-cell 1 ...` against the same build without the patch.

## Log

- 2026-11-05: created.
- 2026-11-05: 2026-11-05: Metal app with both rules and the reset, planet-gpu: 27.20 s against 67.10 s (2.47x), chi2 -0.0005
