#pragma once
// Plain data types shared between kampylos_gpu_kernel.cu (the CUDA kernel) and
// kampylos_gpu_complete.h (the host-side completion step that follows it) -- no CUDA keywords
// here, so a plain C++ compiler (building main_kampylos_boinc.cpp's eventual GPU wrapper, not
// nvcc) can include this and kampylos_gpu_complete.h without also needing to understand
// __global__/__device__. Only the kernel .cu file itself needs those.

struct LightCurvePoint {
    double t;
    double flux;
    double sigma;
};

// Matches fit_binary_local's x[] order (t0, u0, log_tE, alpha, log_rho) -- s and q are fixed for
// the whole grid cell (passed separately, not per-candidate) since every candidate in one kernel
// launch belongs to the same (log_s, log_q) cell.
struct Candidate {
    double t0;
    double u0;
    double log_tE;
    double alpha;
    double log_rho;
};

// Fixed per-candidate capacity for bad-point indices. First set to 64 on the assumption that
// needing the finite-source fallback at all would be a rare, small-cluster occurrence; tests
// against a realistic light curve (tests/test_kampylos_gpu_kernel.cu, 1744 points, dense
// near-peak sampling) overflowed that on 532/2001 candidates -- not rare, and the real in-event
// cluster is wider than 64 points. Raised to 512, confirmed by the same test to clear every
// candidate with zero overflow (max observed: see that test's own printed max_bad). `overflow`
// stays as the safety valve for the genuinely pathological case beyond that -- the caller's
// choice how to handle it (e.g. fall back to this candidate's GPU-only sums as an approximation,
// or redo it entirely on the CPU, or raise this constant further), not something this file
// decides.
#define KAMPYLOS_MAX_BAD_POINTS 512

struct CandidateResult {
    // Six sufficient statistics from the fast-path-OK points only -- the caller (see
    // kampylos_gpu_complete.h) completes these by adding each listed bad point's real
    // contribution, then derives fs, fb, chi2 in closed form (see kampylos_gpu_kernel.cu's top
    // comment for the identity). Not meaningful on their own when poisoned/overflow are set.
    double S_AA, S_A1, S_11, S_Ay, S_1y, S_yy;
    int n_bad;
    int overflow;             // 1 if more than KAMPYLOS_MAX_BAD_POINTS points needed fallback
    int bad_idx[KAMPYLOS_MAX_BAD_POINTS];
    int poisoned;             // 1 if any point gave a non-finite/non-positive magnification even
                               // at the point-source level (degenerate root classification right
                               // at a critical curve) -- genuinely different from needing
                               // finite-source fallback, and not something BinaryMagDark fixes
                               // either; the reference objective() treats this the same way
                               // (immediate chi2=1e18 for the whole candidate), so this does too.
                               // Also set (as a repurposed "dead candidate" signal) when tE/rho
                               // were out of fit_binary_local's valid range to begin with.
};
