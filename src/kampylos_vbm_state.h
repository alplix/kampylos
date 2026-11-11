#pragma once
// History-free evaluations (2026-11-05; KAM-030, KAM-012). Switch: -DKAMPYLOS_RESET_STATE=1, off by default.
//
// VBMicrolensing's binary-lens root solver starts every solve from the roots of the previous one (`zr` in
// NewImages, a function-local static; cmplx_roots_gen(..., use_roots_as_starting_points = true)). So a
// magnification differs in its last bits with the call that ran before it, and Nelder-Mead turns such a
// difference into another path: a seed fitted after other seeds can end in another minimum than the same
// seed fitted alone, which is what a resumed task or a seed-split part computes (KAM-030, measured on the
// OGLE cell: 2833.61 against 2855.31). In the GPU apps the same state makes the result depend on the number
// of completion threads and on how a round is cut into chunks (KAM-012).
//
// With the switch on, kampylos_vbm_prepare() (below) is called before every candidate evaluation, every final
// recomputation of a fit and every cell finalization, and puts the warm start back to what a fresh thread
// starts with (every root 0). A candidate's chi2 is then a function of its parameters alone: the same in
// every process, on every completion thread and in every order. The points of one evaluation still warm-start
// from each other in time order, as before. Cost: nothing measurable (five complex numbers cleared per
// evaluation; the first point's solve starts cold). It changes every result at rounding level once (the
// first point of each evaluation no longer starts from the previous evaluation's roots), so it needs a new
// app version, and it needs the library prepared with `vendor/patch_vbm_thread_local.py --reset` (any form:
// fork, --strip, --block), which adds ResetRootWarmStart() and defines VBM_KAMPYLOS_RESET.
#include "../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.h"

#ifndef KAMPYLOS_RESET_STATE
#define KAMPYLOS_RESET_STATE 0
#endif
#if KAMPYLOS_RESET_STATE && !defined(VBM_KAMPYLOS_RESET)
#error "KAMPYLOS_RESET_STATE=1 needs VBMicrolensing prepared with vendor/patch_vbm_thread_local.py --reset"
#endif

// Relative accuracy of the finite-source integration (2026-11-05, an experiment; a science change, off by
// default). VBMicrolensing refines a source contour until the error of the magnification is below its absolute
// Tol (1e-2) or below RelTol times the magnification; RelTol is 0 by default, so only the absolute goal counts,
// which at a magnification of 30 is 3e-4 relative. -DKAMPYLOS_VBM_RELTOL=1e-3 lets finite-source points brighter
// than a magnification of 10 stop at 1e-3 relative. It does not touch the point-source test (that uses Tol only),
// on any backend. Measured 2026-11-05: no change on the planet, OGLE and far cells, whose finite-source points are
// fainter than that (docs/BENCHMARK.md); a high-magnification event was not tried.

// Puts the instance into the state every candidate evaluation starts from: the warm start of the root solver
// (KAMPYLOS_RESET_STATE) and the accuracy settings (KAMPYLOS_VBM_RELTOL). Called before every candidate
// evaluation, every final recomputation of a fit and every cell finalization, on every backend. With both
// switches off it does nothing.
inline void kampylos_vbm_prepare(VBMicrolensing& vbm) {
#if KAMPYLOS_RESET_STATE
    vbm.ResetRootWarmStart();
#endif
#ifdef KAMPYLOS_VBM_RELTOL
    vbm.RelTol = KAMPYLOS_VBM_RELTOL;
#endif
    (void)vbm;
}
