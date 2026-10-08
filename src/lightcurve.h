#pragma once
// Photometry data loading and the flux/chi-squared machinery shared by both the single-lens
// pre-fit and the binary-lens grid search.
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

struct DataPoint {
    double t;      // time, HJD (or HJD - 2450000, whatever the source file uses -- kept
                    // consistent within one event, that's all that matters for fitting)
    double flux;
    double sigma;  // flux uncertainty, same units as flux
};

// Loads "time value error" (whitespace-separated, '#'-prefixed comment lines skipped). If
// input_is_mag is true, value/error are magnitude/mag-error and get converted to flux via
// flux = 10^(-0.4*(mag - zp)), sigma_flux = flux * ln(10) * 0.4 * mag_err -- the standard
// linear-error-propagation approximation, valid for the sub-0.1-mag errors typical of these
// survey alert-system light curves. zp is arbitrary (only relative flux matters for fitting,
// since fs/fb are fitted anyway) -- 18.0 is just a convenient round number, not a real
// zeropoint from any specific instrument.
inline std::vector<DataPoint> load_photometry(const std::string& path, bool input_is_mag) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<DataPoint> out;
    std::string line;
    const double zp = 18.0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double t, v, e;
        if (!(ss >> t >> v >> e)) continue;
        DataPoint dp;
        dp.t = t;
        if (input_is_mag) {
            dp.flux = pow(10.0, -0.4 * (v - zp));
            dp.sigma = dp.flux * log(10.0) * 0.4 * e;
        } else {
            dp.flux = v;
            dp.sigma = e;
        }
        if (dp.sigma > 0 && std::isfinite(dp.flux)) out.push_back(dp);
    }
    return out;
}

struct FluxFit {
    double fs, fb;   // source flux, blend flux
    double chi2;
    int degenerate;  // 0 = exact least squares; 1 = near-singular design, tiny-ridge fallback
                     // used; 2 = fs/fb sanity backstop tripped (constant-flux model returned)
};

// Relative-determinant threshold below which the 2x2 normal equations count as genuinely
// near-singular. rel_det = det / (S_AA*S_11) = 1 - <A>^2/<A^2> (weighted), i.e. the weighted
// variance of the magnification across the data divided by its mean square. Any light curve
// that actually samples an event has rel_det >= ~1e-6 (a 0.1% magnification change on a few
// points already gives that); 1e-10 means A is constant to ~1e-5 relative across EVERY point --
// a trial so far from the data (tE >> span, or the event entirely between points) that fs and
// fb are genuinely not separable.
#define KAMPYLOS_FLUX_RELDET_MIN 1e-10
// Size of the fallback ridge, relative to each diagonal entry. Only ever applied below
// KAMPYLOS_FLUX_RELDET_MIN, where the model prediction fs*A+fb is (to ~1e-5) the same along
// the whole degenerate fs/fb direction, so this cannot change chi2 by more than a rounding
// error -- it only picks a finite point on that flat valley.
#define KAMPYLOS_FLUX_RIDGE 1e-8

// The one closed-form (fs, fb) solve shared by every backend: linear_flux_fit() below (CPU)
// and kampylos_complete_candidate() (CUDA/OpenCL/Metal host completion) both call this, so
// the backends cannot drift apart in how fs/fb are chosen.
//
// History (2026-10-08 fix): this used to scale BOTH diagonal entries by 1.01 unconditionally
// (eps_reg = 1e-2 Tikhonov damping), added because optimizers walked into near-singular
// designs and rode them. That damping biases fs/fb for EVERY trial, and chi2 was then evaluated
// at the biased fs/fb -- verified on MOA-2003-BLG-004 to inflate chi2 by hundreds to thousands
// (worse for high-magnification trials, where fs carries most of the model), and by a
// different amount for the binary grid cells than for the single-lens anchor, so delta_chi2
// was meaningless. Now: exact unregularized least squares whenever the design is not
// genuinely singular (rel_det >= KAMPYLOS_FLUX_RELDET_MIN); a tiny ridge only below that; and
// the independent fs/fb sanity backstop (|fs| or |fb| > fs_bound -> constant-flux model, which
// can only make chi2 WORSE, so it never rewards an optimizer for approaching it).
// fb_offset: constant that was subtracted from every flux before the sums were formed (the GPU
// paths centre the light curve); the backstop is applied to the uncentred fb, fb + fb_offset.
inline FluxFit flux_fit_solve(double S_AA, double S_A1, double S_11, double S_Ay, double S_1y,
                              double fs_bound, double fb_offset = 0.0) {
    FluxFit out;
    out.chi2 = 0;
    out.degenerate = 0;
    double diag = S_AA * S_11;
    double det = diag - S_A1 * S_A1;
    if (!(diag > 0) || !(det > KAMPYLOS_FLUX_RELDET_MIN * diag)) {
        double a = S_AA * (1.0 + KAMPYLOS_FLUX_RIDGE), b = S_11 * (1.0 + KAMPYLOS_FLUX_RIDGE);
        det = a * b - S_A1 * S_A1;
        if (det > 0) {
            out.fs = (b * S_Ay - S_A1 * S_1y) / det;
            out.fb = (a * S_1y - S_A1 * S_Ay) / det;
        } else {
            out.fs = 0;
            out.fb = S_11 > 0 ? S_1y / S_11 : 0;
        }
        out.degenerate = 1;
    } else {
        out.fs = (S_11 * S_Ay - S_A1 * S_1y) / det;
        out.fb = (S_AA * S_1y - S_A1 * S_Ay) / det;
    }
    // Backstop (unchanged in spirit from the original design): a real fit's fs/fb are flux
    // contributions in the data's own units and cannot legitimately run to 1000x the observed
    // flux range. The bound is the light curve's flux RANGE (max - min), which is invariant
    // under a constant flux offset -- the GPU paths centre the flux before upload, and this
    // keeps the bound identical on every backend.
    if (!std::isfinite(out.fs) || !std::isfinite(out.fb) ||
        std::fabs(out.fs) > fs_bound || std::fabs(out.fb + fb_offset) > fs_bound) {
        out.fs = 0;
        out.fb = S_11 > 0 ? S_1y / S_11 : 0;
        out.degenerate = 2;
    }
    return out;
}

// 1000x the light curve's flux range -- see flux_fit_solve(). Computed once per light curve.
inline double flux_fit_bound(const std::vector<DataPoint>& data) {
    if (data.empty()) return 1e300;
    double lo = data[0].flux, hi = data[0].flux;
    for (const auto& d : data) { lo = std::min(lo, d.flux); hi = std::max(hi, d.flux); }
    return 1e3 * std::max(hi - lo, 1e-12);
}

// Given a magnification model evaluated at every data point, finds the best-fit (fs, fb) in
// closed form (model flux = fs*A + fb is linear in fs, fb for fixed A) and returns the
// resulting chi-squared, evaluated directly from the residuals at exactly those fs/fb.
// fs_bound: pass flux_fit_bound(data) when calling this in a loop; <= 0 computes it here.
inline FluxFit linear_flux_fit(const std::vector<DataPoint>& data, const std::vector<double>& mag,
                               double fs_bound = -1.0) {
    double S_AA = 0, S_A1 = 0, S_11 = 0, S_Ay = 0, S_1y = 0;
    size_t n = data.size();
    for (size_t i = 0; i < n; i++) {
        double w = 1.0 / (data[i].sigma * data[i].sigma);
        double A = mag[i];
        S_AA += w * A * A;
        S_A1 += w * A;
        S_11 += w;
        S_Ay += w * A * data[i].flux;
        S_1y += w * data[i].flux;
    }
    if (fs_bound <= 0) fs_bound = flux_fit_bound(data);
    FluxFit out = flux_fit_solve(S_AA, S_A1, S_11, S_Ay, S_1y, fs_bound);
    double chi2 = 0;
    for (size_t i = 0; i < n; i++) {
        double resid = (data[i].flux - (out.fs * mag[i] + out.fb)) / data[i].sigma;
        chi2 += resid * resid;
    }
    out.chi2 = chi2;
    return out;
}
