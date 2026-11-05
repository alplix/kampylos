---
id: KAM-039
title: A cell's best fit can be reported in either mirror form, (u0, alpha) or (-u0, -alpha), which describe the same light curve
type: design-note
status: documented
severity: low
platforms: [all]
area: determinism
action:
next:
blocked_by: []
pattern:
aliases: []
reported: 2026-11-05
source: M4 benchmark 2026-11-05 (docs/BENCHMARK.md, half the seeds)
where: src/binary_fit.h @ fa03bd1
fixed_in:
---

## Symptom

Two builds that differ only in their seeds reported the planet cell's best fit as t0 2455999.995101, u0 -0.100075,
alpha 6.147547 and as t0 2455999.995101, u0 +0.100075, alpha 0.135638, with the same chi2 411.2228 (Apple M4 Pro,
2026-11-05, docs/BENCHMARK.md, "half the seeds").

## Cause

The source trajectory y1 = u0 sin(alpha) - tn cos(alpha), y2 = -u0 cos(alpha) - tn sin(alpha) keeps y1 and flips y2
under (u0, alpha) -> (-u0, -alpha), and the binary lens is symmetric about its axis (`BinaryMag2` uses |y2|): the two
parameter sets give the same light curve. `kampylos_make_seeds()` starts from both members of every such pair
(`src/binary_fit.h`, "MIRROR PAIRS"; `tests/test_mirror_seeds.cpp`), so which form wins a cell depends on which seed of
a pair ends lowest, and that can change with the backend, the compiler, a search change or a seed-split part.

## Fix

Nothing to change in the apps. A validator or analysis that compares fits of the same cell (KAM-025) should compare
(u0, alpha) modulo the mirror: (u0, alpha) and (-u0, 2 pi - alpha) are one solution. Keeping only one member of each
pair (`-DKAMPYLOS_SEEDS_HALF=1`) halves the work but lost the best minimum on two of three cells (+23 and +35 in chi2):
the two runs of a pair are independent attempts.

## Test

`tests/test_mirror_seeds.cpp` (the pairing, and equal chi2 at the two starts of each pair).

## Log

- 2026-11-05: created.
