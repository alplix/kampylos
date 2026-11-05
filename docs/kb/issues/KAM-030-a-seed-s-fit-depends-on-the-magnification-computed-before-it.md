---
id: KAM-030
title: A seed's fit depends on the magnification computed before it, so a resumed or seed-split task can end a seed in another minimum
type: bug
status: fix-proposed
severity: medium
platforms: [all]
area: determinism
action: fix
next: Before each seed in fit_binary_local_steps(), compute two fixed magnifications so every seed starts from the same VBMicrolensing state; check seed 96 of the OGLE cell alone and in sequence
blocked_by: []
pattern:
aliases: []
reported: 2026-11-04
source: speed benchmark 2026-11-04 (docs/BENCHMARK.md), reproduced with a Nelder-Mead trace harness
where: src/binary_fit.h:66 @ 908f47f
fixed_in:
---

## Symptom

The same seed ends in different minima depending on what the process computed before it. Bundled OGLE light curve
(`tests/sample_ogle_blg1900.dat`), cell `mag 0.3 0.3 1 -2.5 -2.5 1 log10`, CPU code path (`fit_binary_local_steps`),
fast CPU build, 2026-11-04:

| run | seed 96, first pass | after the restarts |
|---|---|---|
| seeds 0, 24, 48, 72, 96 ... in one process (`part 0 24`) | 2833.6074, 677 evaluations | 2833.4503 |
| seeds 72, 96 ... | 2833.6074 | 2833.4503 |
| seed 96 alone (what a resume from a seed checkpoint at seed 96 computes) | 2855.3127, 651 evaluations | 2855.3126 |

In the `part 0 24` work unit seed 96 is the best seed, so a task killed after seed 72 and resumed would report a
cell chi2 about 22 higher than an uninterrupted one. The same happened when a change elsewhere altered the
preceding calls only: with a restart rule that skips some second restarts of seeds 0-72 (an experiment, see
`docs/BENCHMARK.md`), seed 96's first pass, which the rule does not touch, ended at 2855.31 instead of 2833.61. On
the whole 240-seed cell another seed holds the minimum (2811.6732) and the cell result did not change.

## Cause

* The root solver starts from the previous call's roots (`cmplx_roots_gen(zr, ...)`, a function-local static,
  `vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp:1535`; KAM-012), so the first magnifications of
  a seed differ in their last bits with the seed that ran before. Verified by the run above.
* Nelder-Mead on this objective amplifies a last-bit difference into another path: the objective is noisy at the
  level of the finite-source integration's accuracy (`Tol` 1e-2 in the magnification) and many passes stop at the
  400-iteration cap (606 of 1440 passes on the planet and OGLE cells), so two paths that part early end in different
  local minima. Inferred from the traces, not proven in general.
* KAM-012 describes the same state for the GPU completion threads and calls the effect rounding-level; this case
  shows it can change the minimum a seed reaches.

## Fix

Proposed, tried in the trace harness only: before each seed, compute two fixed magnifications at the cell's s and q
(`BinaryMag2(s, q, 0, 0, 0.01)`, which runs the contour integration, then `BinaryMag2(s, q, 10, 10, 0.001)`, a
point-source one), so every seed starts from the same library state whatever ran before. With it, seed 96 gives
2833.6074 (677 evaluations) alone, after seed 72 and in the 10-seed sequence. Cost: two magnifications per seed. It
changes the result of every seed whose predecessor left another state, so it needs a new app version; in exchange a
resumed task, a seed-split part and the whole cell give each seed the same fit, and experiments with the restart
rule become predictable. For the GPU apps the same reset before each candidate's completion would make the result
independent of the completion-thread count (KAM-012); not tried.

Proposed on 2026-11-05, behind a switch that is off by default (history-free evaluations,
`src/kampylos_vbm_state.h`): `vendor/patch_vbm_thread_local.py --reset` adds `ResetRootWarmStart()` to the library
(it clears the root solver's warm start `zr` in `NewImages`, the one array that carries state from one solve to the
next), and with `-DKAMPYLOS_RESET_STATE=1` every candidate evaluation, final recomputation and cell finalization starts
from it, on every backend. Exact by construction (a fresh thread's state, not "two fixed magnifications that happen to
be enough") and free: on a fixed set of 300 candidates the objective and the GPU completion step take the same time
with and without it. Measured on an Apple M4 Pro (docs/BENCHMARK.md, Results on an Apple M4 (2026-11-05)): seed 96
of the OGLE cell gives 2833.4503 alone and after seeds 0-72; the whole planet, OGLE and far cells write the same result
lines as without the switch; the 10-seed `far` profile moves (chi2 -0.11, a better minimum) because its seeds no longer
inherit their predecessors' state; the Metal app writes the same line with 1, 2, 4 and 8 completion threads (KAM-012).

## Test

`tests/test_vbm_state_reset.cpp` (2026-11-05, build line in its header; needs the `--reset` library): 300 candidates
of the planet cell in both orders through the CPU objective and through the GPU completion step must give the same
bits, the same loop without the reset must not (153 of 300 differ), and seed 96 of the OGLE cell alone and after seeds
0-72 must give the same fit. Passes with the `--strip` and `--block` libraries.

## Log

- 2026-11-04: created from the speed benchmark; reproduced with the trace harness (fast CPU build, Apple M1).
- 2026-11-05: 2026-11-05: history-free evaluations behind -DKAMPYLOS_RESET_STATE=1 (vendor/patch_vbm_thread_local.py --reset, src/kampylos_vbm_state.h), tests/test_vbm_state_reset.cpp, measured on the M4 (docs/BENCHMARK.md)
