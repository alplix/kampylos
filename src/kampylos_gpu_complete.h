#pragma once
// Host-side completion of a CandidateResult from kampylos_gpu_kernel.cu: adds each listed bad
// point's real (VBMicrolensing-exact, finite-source-aware) contribution to the kernel's partial
// sums, then derives the whole light curve's fs, fb, chi2 via the closed-form identity documented
// in that kernel's own top comment. Plain host C++ -- no CUDA keywords, so it compiles under the
// same plain compiler as the rest of Kampylos's CPU code (this is meant to end up called from
// main_kampylos_boinc.cpp's eventual GPU-enabled build, not just this header's own test).
#include <cmath>
#include "kampylos_gpu_types.h"
#include "VBMicrolensingLibrary.h"

struct CompletedFit {
    double chi2, fs, fb;
    bool dead; // poisoned (degenerate point-source root) or overflow (more bad points than
                // KAMPYLOS_MAX_BAD_POINTS) -- caller should treat this candidate as rejected,
                // same as fit_binary_local's objective() returning 1e18.
};

inline CompletedFit kampylos_complete_candidate(
    VBMicrolensing& vbm,
    const CandidateResult& r,
    const Candidate& c,
    const LightCurvePoint* data,
    double s, double q
) {
    CompletedFit out;
    if (r.poisoned || r.overflow) {
        out.chi2 = 1e18; out.fs = 0; out.fb = 0; out.dead = true;
        return out;
    }

    double S_AA = r.S_AA, S_A1 = r.S_A1, S_11 = r.S_11, S_Ay = r.S_Ay, S_1y = r.S_1y, S_yy = r.S_yy;
    double tE = exp(c.log_tE), tE_inv = 1.0 / tE;
    double rho = exp(c.log_rho);
    double salpha = sin(c.alpha), calpha = cos(c.alpha);

    for (int k = 0; k < r.n_bad; k++) {
        int i = r.bad_idx[k];
        double t = data[i].t;
        double tn = (t - c.t0) * tE_inv;
        double y1 = c.u0 * salpha - tn * calpha;
        double y2 = -c.u0 * calpha - tn * salpha;
        // The real, finite-source-aware value -- this is exactly the computation this whole
        // design exists to avoid doing on the GPU (BinaryMag2 internally calls BinaryMagDark's
        // adaptive contour integration for points like this one), done here on the CPU where the
        // already-correct VBMicrolensing implementation already exists.
        double A = vbm.BinaryMag2(s, q, y1, y2, rho);
        if (!(A > 0) || !std::isfinite(A)) {
            out.chi2 = 1e18; out.fs = 0; out.fb = 0; out.dead = true;
            return out;
        }
        double sigma = data[i].sigma;
        double w = 1.0 / (sigma * sigma);
        double flux = data[i].flux;
        S_AA += w * A * A;
        S_A1 += w * A;
        S_11 += w;
        S_Ay += w * A * flux;
        S_1y += w * flux;
        S_yy += w * flux * flux;
    }

    // Same ridge regularization as lightcurve.h's linear_flux_fit() -- this is a SEPARATE
    // reimplementation of the same closed-form fs/fb solve (needed here since the GPU's partial
    // sums arrive already split from the CPU-patched bad-point sums, not through that function),
    // so it needs the identical fix: an absolute 1e-300 degeneracy threshold never trips in
    // practice, letting a near-singular design (the model magnification barely varying across the
    // light curve -- common when a candidate's trial parameters are far from the real event)
    // through to a raw Cramer's-rule divide that blows fs/fb up to huge, meaningless values. See
    // lightcurve.h for the full analysis (confirmed directly: this exact gap caused the GPU
    // backends to report wildly wrong fs/fb -- e.g. in the hundreds of thousands -- for harder
    // light curves where the CPU reference, going through the already-fixed linear_flux_fit(),
    // got the correct answer on the identical candidate).
    double lambda = 1e-3 * S_11;
    double S_AA_r = S_AA + lambda, S_11_r = S_11 + lambda;
    double det = S_AA_r * S_11_r - S_A1 * S_A1;
    double fs = (S_11_r * S_Ay - S_A1 * S_1y) / det;
    double fb = (S_AA_r * S_1y - S_A1 * S_Ay) / det;
    // chi2 = S_yy - fs*S_Ay - fb*S_1y (see kampylos_gpu_kernel.cu's top comment for the identity).
    out.chi2 = S_yy - fs * S_Ay - fb * S_1y;
    out.fs = fs;
    out.fb = fb;
    out.dead = false;
    return out;
}
