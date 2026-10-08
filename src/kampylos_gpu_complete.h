#pragma once
// Host-side completion of a CandidateResult from kampylos_gpu_kernel.cu: adds each listed bad
// point's real (VBMicrolensing-exact, finite-source-aware) contribution to the kernel's partial
// sums, then derives the whole light curve's fs, fb, chi2 via the closed-form identity documented
// in that kernel's own top comment. Plain host C++ -- no CUDA keywords, so it compiles under the
// same plain compiler as the rest of Kampylos's CPU code (this is meant to end up called from
// main_kampylos_boinc.cpp's eventual GPU-enabled build, not just this header's own test).
#include <cmath>
#include <algorithm>
#include "kampylos_gpu_types.h"
#include "lightcurve.h" // flux_fit_solve(): the one fs/fb solve shared with the CPU path
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
    double s, double q,
    double fs_bound = 1e300,   // flux_fit_bound() of the light curve (backstop, see lightcurve.h)
    double fb_offset = 0.0     // constant subtracted from every flux before upload (GPU paths
                               // centre the light curve -- see gpu_upload_light_curve())
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

    // The closed-form fs/fb solve is lightcurve.h's flux_fit_solve() -- the SAME function the
    // CPU path uses (2026-10-08: this used to be a separate copy of the old eps_reg=1e-2 ridge
    // solve, which biased chi2 upward by hundreds to thousands; see lightcurve.h). The data
    // arrive flux-centred (flux - fb_offset), which leaves fs and chi2 unchanged and shifts fb
    // by -fb_offset; the backstop test is applied to the uncentred fb so it trips exactly where
    // the CPU path's does (flux_fit_solve's fb_offset argument).
    FluxFit ff = flux_fit_solve(S_AA, S_A1, S_11, S_Ay, S_1y, fs_bound, fb_offset);
    double fs = ff.fs, fb = ff.fb;
    // chi2 = S_yy - fs*S_Ay - fb*S_1y (see kampylos_gpu_kernel.cu's top comment for the identity
    // -- valid at the exact least-squares fs/fb; for the rare tiny-ridge/backstop cases the
    // general quadratic form below is used instead, which is exact for ANY fs/fb).
    out.chi2 = (ff.degenerate == 0)
        ? S_yy - fs * S_Ay - fb * S_1y
        : S_yy - 2.0 * fs * S_Ay - 2.0 * fb * S_1y + fs * fs * S_AA + 2.0 * fs * fb * S_A1 + fb * fb * S_11;
    out.fs = fs;
    out.fb = fb;
    out.dead = false;
    return out;
}
