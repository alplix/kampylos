#pragma once
// Single-lens anchor models for Kampylos (2026-10-08): point-source point-lens (PSPL), uniform
// finite-source point-lens (FSPL) and linearly limb-darkened FSPL, each fitted once per work
// unit with multi-start seeding. The binary-lens grid is judged against the BEST of these
// (the "anchor"): a binary cell only counts as an improvement if it beats a single lens that
// was also allowed a finite source -- the vetted false positive MOA-2003-BLG-004 gained ~270
// chi2 from finite-source effects alone, which a PSPL-only anchor attributed to the binary.
//
// Everything here is self-contained (no VBMicrolensing calls): VBMicrolensing's own ESPL
// routines need an external lookup table file (ESPL.tbl) that a BOINC app does not ship, so
// the finite-source magnification is computed directly:
//
//   uniform disk:  A_U(u, rho) = F(rho) / (pi rho^2), with the magnified flux inside the disk
//                  F = int dphi [ r sqrt(r^2+4) / 2 ]_{r1(phi)}^{r2(phi)}
//                  (the radial integral of the point-lens magnification (r^2+2)/sqrt(r^2+4)
//                  is closed form), leaving one smooth angular integral done by Gauss-Legendre;
//   linear LD:     I(r) = 1 - a + a*sqrt(1 - r^2/rho^2). Integrating by parts in the disk
//                  radius gives A_LD = [(1-a) A_U(u,rho) + a int_0^1 (1-m^2) A_U(u, rho
//                  sqrt(1-m^2)) dm] / (1 - a/3), another smooth 1-D integral.
// Far from the lens (u > 10 rho) the quadrupole expansion A + <r^2>/4 * Laplacian(A) is used
// (relative error < ~2e-5 there). Validated against VBMicrolensing's table-based ESPLMag2 /
// ESPLMagDark in tests/test_single_lens.cpp.
#include <vector>
#include <cmath>
#include <algorithm>
#include "simplex.h"
#include "lightcurve.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------------------------
// Gauss-Legendre nodes/weights on [0, 1], computed once (function-local statics are initialised
// thread-safely in C++11).
struct KGaussLegendre {
    std::vector<double> x, w;
    explicit KGaussLegendre(int n) : x(n), w(n) {
        for (int i = 0; i < n; i++) {
            double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), pp = 0;
            for (int it = 0; it < 100; it++) {
                double p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < n; j++) {
                    double p3 = p2; p2 = p1;
                    p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
                }
                pp = n * (z * p1 - p2) / (z * z - 1.0);
                double z1 = z;
                z = z1 - p1 / pp;
                if (std::fabs(z - z1) < 1e-15) break;
            }
            x[i] = 0.5 * (1.0 - z);
            w[i] = 1.0 / ((1.0 - z * z) * pp * pp);  // = 2/((1-z^2)pp^2) * 1/2 for [0,1]
        }
    }
};
inline const KGaussLegendre& kgl_angle() { static const KGaussLegendre g(48); return g; }
inline const KGaussLegendre& kgl_ld() { static const KGaussLegendre g(20); return g; }

inline double pspl_mag(double u) {
    u = std::fabs(u);
    if (u < 1e-12) u = 1e-12;
    double u2 = u * u;
    return (u2 + 2.0) / (u * std::sqrt(u2 + 4.0));
}

// Laplacian of the point-lens magnification, used by the quadrupole expansion.
inline double pspl_lapl(double u) {
    double u2 = u * u, s = std::sqrt(u2 + 4.0), s3 = s * (u2 + 4.0), s5 = s3 * (u2 + 4.0);
    return 8.0 / (u2 * u * s3) + 24.0 / (u * s5);
}

static inline double kfs_prim(double r) { return r * std::sqrt(r * r + 4.0); } // 2 * int A r dr

// Uniform-disk finite-source magnification.
inline double fspl_mag_uniform(double u, double rho) {
    u = std::fabs(u);
    if (rho <= 0) return pspl_mag(u);
    if (u > 10.0 * rho) return pspl_mag(u) + rho * rho / 8.0 * pspl_lapl(u);
    const KGaussLegendre& g = kgl_angle();
    double F = 0;
    if (u < rho) {
        // lens inside the disk: phi in [0, pi], r from 0 to r2. Split at pi/2 so the u -> rho
        // kink in r2 sits on a panel boundary.
        double u2 = u * u, rho2 = rho * rho;
        for (int half = 0; half < 2; half++) {
            for (size_t i = 0; i < g.x.size(); i++) {
                double phi = (half + g.x[i]) * (M_PI / 2.0);
                double sp = std::sin(phi), cp = std::cos(phi);
                double r2 = u * cp + std::sqrt(std::max(rho2 - u2 * sp * sp, 0.0));
                F += g.w[i] * (M_PI / 2.0) * kfs_prim(std::max(r2, 0.0));
            }
        }
        F *= 0.5 * 2.0; // [r sqrt(r^2+4)/2], phi in [0,pi] covers half the disk -> x2
    } else {
        // lens outside: sin(phi) = k sin(psi), k = rho/u, psi in [0, pi/2] removes the
        // square-root endpoint singularity at phi_max.
        double k = rho / u;
        for (size_t i = 0; i < g.x.size(); i++) {
            double psi = g.x[i] * (M_PI / 2.0);
            double sps = std::sin(psi), cps = std::cos(psi);
            double cphi = std::sqrt(std::max(1.0 - k * k * sps * sps, 1e-300));
            double r1 = u * cphi - rho * cps, r2 = u * cphi + rho * cps;
            double jac = k * cps / cphi;
            F += g.w[i] * (M_PI / 2.0) * (kfs_prim(r2) - kfs_prim(std::max(r1, 0.0))) * jac;
        }
        F *= 0.5 * 2.0;
    }
    return F / (M_PI * rho * rho);
}

// Linearly limb-darkened finite source, I(r) = 1 - a + a sqrt(1 - r^2/rho^2).
inline double fspl_mag_ld(double u, double rho, double a) {
    if (a <= 0) return fspl_mag_uniform(u, rho);
    u = std::fabs(u);
    if (u > 10.0 * rho) {
        double r2mean = 2.0 * rho * rho * ((1.0 - a) / 4.0 + 2.0 * a / 15.0) / (1.0 - a / 3.0);
        return pspl_mag(u) + r2mean / 4.0 * pspl_lapl(u);
    }
    const KGaussLegendre& g = kgl_ld();
    double I = 0;
    for (size_t i = 0; i < g.x.size(); i++) {
        double m = g.x[i];
        double f = 1.0 - m * m;
        I += g.w[i] * f * fspl_mag_uniform(u, rho * std::sqrt(f));
    }
    return ((1.0 - a) * fspl_mag_uniform(u, rho) + a * I) / (1.0 - a / 3.0);
}

// ---------------------------------------------------------------------------------------------
enum SingleLensKind { SL_PSPL = 0, SL_FSPL = 1, SL_FSPL_LD = 2 };

struct SingleLensFit {
    int kind = SL_PSPL;
    double t0 = 0, u0 = 0, tE = 1, rho = 0, ld_a1 = 0;
    double fs = 0, fb = 0, chi2 = 1e300;
};

inline double single_lens_mag(const SingleLensFit& m, double t) {
    double tau = (t - m.t0) / m.tE;
    double u = std::sqrt(m.u0 * m.u0 + tau * tau);
    if (m.kind == SL_PSPL) return pspl_mag(u);
    if (m.kind == SL_FSPL) return fspl_mag_uniform(u, m.rho);
    return fspl_mag_ld(u, m.rho, m.ld_a1);
}

// Everything the work unit needs from the single-lens stage: the three fits, which one is the
// anchor, the anchor's per-point model flux (for the per-cell delta-chi2 diagnostics) and a
// goodness-of-fit summary (written to the result header).
struct SingleLensAnchor {
    SingleLensFit pspl, fspl, fspl_ld;
    SingleLensFit best;                 // lowest chi2 of the three
    // PSPL fits at FIXED timescales (tE = 3, 10, 30, 100 d) that differ from the best one by > 2x,
    // lowest chi2 first, at most 2. Used only as extra binary seeds: for strongly blended high-magnification
    // events the global PSPL minimum often runs off along the tE-u0-fs degeneracy (A ~ 1/u, only
    // u0*tE and fs*tE are constrained) to a huge tE, which is a fine single-lens chi2 reference
    // but a useless starting point for the binary search (OGLE-2017-BLG-1522: tE -> 3000 d for
    // a 7.5 d event).
    std::vector<SingleLensFit> pspl_alt;
    std::vector<double> model_flux;     // best.fs * A_best(t_i) + best.fb, one per data point
    std::vector<double> chi2_pt;        // per-point chi2 of the anchor
    int npts = 0, n_nights = 0, dof = 0;
    double chi2dof = 0;                 // best.chi2 / dof
    double worst1pct_frac = 0;          // fraction of best.chi2 carried by the worst 1% points
    int n_peak = 0;                     // points with |t - t0| < 2 tE
    double chi2dof_peak = 0, chi2dof_base = 0, chi2_peak = 0;
    // Data-driven fallback anchor (peak time + excess width), kept for the binary seeding --
    // see pspl_prefit.h for why a second, optimizer-free anchor is useful.
    double t0_raw = 0, tE_raw = 1;
};

// Night index for every data point (sorted by time, consecutive points < 0.3 d apart are the
// same night, a night is capped at 0.6 d so continuous multi-site data still splits into
// night-sized chunks). Survey-agnostic: OGLE (Chile) nights do not straddle a JD boundary but
// MOA (New Zealand) nights do, so a plain floor(JD) would split every MOA night in two.
inline std::vector<int> night_index(const std::vector<DataPoint>& data, int* n_nights_out = nullptr) {
    size_t n = data.size();
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return data[a].t < data[b].t; });
    std::vector<int> night(n, 0);
    int cur = -1;
    double prev_t = -1e300, start_t = -1e300;
    for (size_t k = 0; k < n; k++) {
        double t = data[order[k]].t;
        if (t - prev_t > 0.3 || t - start_t > 0.6) { cur++; start_t = t; }
        night[order[k]] = cur;
        prev_t = t;
    }
    if (n_nights_out) *n_nights_out = cur + 1;
    return night;
}

namespace kampylos_sl_detail {

// chi2 of one single-lens parameter vector; mag is scratch space.
inline double eval(const std::vector<DataPoint>& data, SingleLensFit& m, std::vector<double>& mag,
                   double fs_bound) {
    for (size_t i = 0; i < data.size(); i++) {
        double a = single_lens_mag(m, data[i].t);
        if (!std::isfinite(a) || a <= 0) return 1e18;
        mag[i] = a;
    }
    FluxFit ff = linear_flux_fit(data, mag, fs_bound);
    m.fs = ff.fs; m.fb = ff.fb; m.chi2 = ff.chi2;
    return ff.chi2;
}

// Robust peak time: maximum of a 5-point running median (time-ordered), so one or two outlier
// points (saturation, cosmic rays -- exactly what fooled the vetting of MOA-2003-BLG-004)
// cannot pull the t0 seed away from the real event.
inline double robust_peak_time(const std::vector<DataPoint>& data) {
    size_t n = data.size();
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return data[a].t < data[b].t; });
    double best = -1e300, best_t = data[order[n / 2]].t;
    for (size_t k = 2; k + 2 < n; k++) {
        double v[5];
        for (int j = 0; j < 5; j++) v[j] = data[order[k + j - 2]].flux;
        std::sort(v, v + 5);
        if (v[2] > best) { best = v[2]; best_t = data[order[k]].t; }
    }
    return best_t;
}

} // namespace kampylos_sl_detail

// Fits PSPL, FSPL and FSPL+LD with multi-start Nelder-Mead and fills the anchor summary.
// Deterministic (no RNG), so every host computing a WU of the same event gets the same anchor.
inline SingleLensAnchor fit_single_lens_anchor(const std::vector<DataPoint>& data) {
    using namespace kampylos_sl_detail;
    SingleLensAnchor out;
    const size_t n = data.size();
    out.npts = (int)n;
    std::vector<double> mag(n);
    const double fs_bound = flux_fit_bound(data);

    double t_lo = data[0].t, t_hi = data[0].t, t_maxflux = data[0].t, maxflux = data[0].flux;
    for (const auto& d : data) {
        t_lo = std::min(t_lo, d.t); t_hi = std::max(t_hi, d.t);
        if (d.flux > maxflux) { maxflux = d.flux; t_maxflux = d.t; }
    }
    const double span = std::max(t_hi - t_lo, 1.0);
    const double tE_max = std::min(5.0 * span, 3000.0);
    const double t0_robust = robust_peak_time(data);

    // ---- PSPL: x = (t0, |u0|, ln tE) -------------------------------------------------------
    auto pspl_obj = [&](const std::vector<double>& x) -> double {
        SingleLensFit m; m.kind = SL_PSPL;
        m.t0 = x[0]; m.u0 = std::fabs(x[1]); m.tE = std::exp(x[2]);
        if (m.tE < 0.05 || m.tE > tE_max || m.u0 < 1e-5 || m.u0 > 5.0) return 1e18;
        return eval(data, m, mag, fs_bound);
    };
    std::vector<double> t0_seeds = { t0_robust };
    if (std::fabs(t_maxflux - t0_robust) > 0.5) t0_seeds.push_back(t_maxflux);
    const double u0_seeds[] = { 0.003, 0.02, 0.07, 0.2, 0.6 };
    const double tE_seeds[] = { 3.0, 10.0, 30.0, 100.0 };
    struct Cand { std::vector<double> x; double f; };
    std::vector<Cand> pspl_cands;
    for (double t0s : t0_seeds)
        for (double u0s : u0_seeds)
            for (double tEs : tE_seeds) {
                std::vector<double> x0 = { t0s, u0s, std::log(tEs) };
                std::vector<double> st = { std::max(0.3, 0.2 * tEs), 0.5 * u0s, 0.4 };
                SimplexResult r = nelder_mead(pspl_obj, x0, st, 600, 1e-9, 1e-9);
                for (int k = 0; k < 2; k++) {
                    std::vector<double> st2 = { std::max(0.05, 0.05 * std::exp(r.x[2])), 0.2 * std::fabs(r.x[1]) + 1e-4, 0.1 };
                    SimplexResult r2 = nelder_mead(pspl_obj, r.x, st2, 600, 1e-10, 1e-10);
                    if (r2.fval < r.fval) r = r2; else break;
                }
                pspl_cands.push_back({ r.x, r.fval });
            }
    std::sort(pspl_cands.begin(), pspl_cands.end(), [](const Cand& a, const Cand& b) { return a.f < b.f; });
    {
        SingleLensFit m; m.kind = SL_PSPL;
        m.t0 = pspl_cands[0].x[0]; m.u0 = std::fabs(pspl_cands[0].x[1]); m.tE = std::exp(pspl_cands[0].x[2]);
        eval(data, m, mag, fs_bound);
        out.pspl = m;
    }
    // Fixed-timescale profile: the best PSPL at tE = 3, 10, 30, 100 d (t0, u0 free). Along the
    // blending degeneracy there is usually no separate local minimum at the physical timescale
    // (every free start slides to large tE), so the profile is what provides seeds at sensible
    // timescales. Keep the two lowest-chi2 profile points whose tE differs from the global
    // minimum (and from each other) by more than a factor 2.
    {
        std::vector<SingleLensFit> prof;
        for (double tEf : tE_seeds) {
            if (tEf > tE_max) continue;
            auto obj = [&](const std::vector<double>& x) -> double {
                SingleLensFit m; m.kind = SL_PSPL;
                m.t0 = x[0]; m.u0 = std::fabs(x[1]); m.tE = tEf;
                if (m.u0 < 1e-5 || m.u0 > 5.0) return 1e18;
                return eval(data, m, mag, fs_bound);
            };
            SingleLensFit bestp; bestp.chi2 = 1e300;
            for (double u0s : { 0.01, 0.1, 0.5 }) {
                SimplexResult r = nelder_mead(obj, { t0_robust, u0s }, { std::max(0.3, 0.2 * tEf), 0.5 * u0s }, 500, 1e-9, 1e-9);
                SimplexResult r2 = nelder_mead(obj, r.x, { std::max(0.05, 0.05 * tEf), 0.2 * std::fabs(r.x[1]) + 1e-4 }, 500, 1e-10, 1e-10);
                if (r2.fval < r.fval) r = r2;
                if (r.fval < bestp.chi2) {
                    bestp.kind = SL_PSPL; bestp.t0 = r.x[0]; bestp.u0 = std::fabs(r.x[1]); bestp.tE = tEf;
                    eval(data, bestp, mag, fs_bound);
                }
            }
            if (bestp.chi2 < 1e17) prof.push_back(bestp);
        }
        std::sort(prof.begin(), prof.end(), [](const SingleLensFit& a, const SingleLensFit& b) { return a.chi2 < b.chi2; });
        for (const auto& m : prof) {
            if (out.pspl_alt.size() >= 2) break;
            bool distinct = std::fabs(std::log(m.tE / out.pspl.tE)) > 0.7;
            for (const auto& a : out.pspl_alt) distinct = distinct && std::fabs(std::log(m.tE / a.tE)) > 0.7;
            if (distinct) out.pspl_alt.push_back(m);
        }
    }
    // Up to three distinct PSPL minima seed the finite-source fits.
    std::vector<Cand> distinct;
    for (const auto& c : pspl_cands) {
        bool dup = false;
        for (const auto& d : distinct)
            if (std::fabs(c.x[0] - d.x[0]) < 0.5 && std::fabs(std::fabs(c.x[1]) - std::fabs(d.x[1])) < 0.2 * std::fabs(d.x[1]) + 1e-3 &&
                std::fabs(c.x[2] - d.x[2]) < 0.1) { dup = true; break; }
        if (!dup) distinct.push_back(c);
        if (distinct.size() >= 3) break;
    }

    // ---- FSPL (uniform): x = (t0, |u0|, ln tE, ln rho) -------------------------------------
    auto fspl_obj = [&](const std::vector<double>& x) -> double {
        SingleLensFit m; m.kind = SL_FSPL;
        m.t0 = x[0]; m.u0 = std::fabs(x[1]); m.tE = std::exp(x[2]); m.rho = std::exp(x[3]);
        if (m.tE < 0.05 || m.tE > tE_max || m.u0 > 5.0 || m.rho < 1e-5 || m.rho > 1.0) return 1e18;
        return eval(data, m, mag, fs_bound);
    };
    std::vector<double> best_fx; double best_ff = 1e300;
    for (const auto& c : distinct) {
        double u0c = std::fabs(c.x[1]);
        std::vector<double> rho_seeds = { 1e-3, 1e-2, 0.05 };
        if (u0c > 1e-4 && u0c < 0.5) rho_seeds.push_back(std::max(1e-4, 1.5 * u0c));
        for (double rs : rho_seeds) {
            std::vector<double> x0 = { c.x[0], u0c, c.x[2], std::log(rs) };
            std::vector<double> st = { std::max(0.05, 0.05 * std::exp(c.x[2])), 0.3 * u0c + 1e-3, 0.1, 0.7 };
            SimplexResult r = nelder_mead(fspl_obj, x0, st, 800, 1e-9, 1e-9);
            for (int k = 0; k < 2; k++) {
                std::vector<double> st2 = { std::max(0.02, 0.02 * std::exp(r.x[2])), 0.1 * std::fabs(r.x[1]) + 1e-4, 0.05, 0.2 };
                SimplexResult r2 = nelder_mead(fspl_obj, r.x, st2, 800, 1e-10, 1e-10);
                if (r2.fval < r.fval) r = r2; else break;
            }
            if (r.fval < best_ff) { best_ff = r.fval; best_fx = r.x; }
        }
    }
    {
        SingleLensFit m; m.kind = SL_FSPL;
        m.t0 = best_fx[0]; m.u0 = std::fabs(best_fx[1]); m.tE = std::exp(best_fx[2]); m.rho = std::exp(best_fx[3]);
        eval(data, m, mag, fs_bound);
        // FSPL contains PSPL as rho -> 0; never report a finite-source fit worse than the PSPL one
        // just because the optimizer stalled.
        if (m.chi2 > out.pspl.chi2) { m = out.pspl; m.kind = SL_FSPL; m.rho = 1e-5; eval(data, m, mag, fs_bound); }
        out.fspl = m;
    }

    // ---- FSPL + linear limb darkening, a1 free in [0, 1]: x = (t0, |u0|, ln tE, ln rho, a1) ---
    auto ld_obj = [&](const std::vector<double>& x) -> double {
        SingleLensFit m; m.kind = SL_FSPL_LD;
        m.t0 = x[0]; m.u0 = std::fabs(x[1]); m.tE = std::exp(x[2]); m.rho = std::exp(x[3]); m.ld_a1 = x[4];
        if (m.tE < 0.05 || m.tE > tE_max || m.u0 > 5.0 || m.rho < 1e-5 || m.rho > 1.0 || m.ld_a1 < 0 || m.ld_a1 > 1.0) return 1e18;
        return eval(data, m, mag, fs_bound);
    };
    {
        const SingleLensFit& f = out.fspl;
        SingleLensFit bestld = f; bestld.kind = SL_FSPL_LD; bestld.ld_a1 = 0; eval(data, bestld, mag, fs_bound);
        for (double a0 : { 0.3, 0.6 }) {
            std::vector<double> x0 = { f.t0, f.u0, std::log(f.tE), std::log(f.rho), a0 };
            std::vector<double> st = { std::max(0.02, 0.02 * f.tE), 0.1 * f.u0 + 1e-4, 0.05, 0.2, 0.15 };
            SimplexResult r = nelder_mead(ld_obj, x0, st, 800, 1e-10, 1e-10);
            SimplexResult r2 = nelder_mead(ld_obj, r.x, { std::max(0.01, 0.01 * f.tE), 0.05 * f.u0 + 1e-4, 0.02, 0.1, 0.05 }, 800, 1e-10, 1e-10);
            if (r2.fval < r.fval) r = r2;
            if (r.fval < bestld.chi2) {
                bestld.t0 = r.x[0]; bestld.u0 = std::fabs(r.x[1]); bestld.tE = std::exp(r.x[2]);
                bestld.rho = std::exp(r.x[3]); bestld.ld_a1 = r.x[4];
                eval(data, bestld, mag, fs_bound);
            }
        }
        out.fspl_ld = bestld;
    }

    out.best = out.pspl;
    if (out.fspl.chi2 < out.best.chi2) out.best = out.fspl;
    if (out.fspl_ld.chi2 < out.best.chi2) out.best = out.fspl_ld;

    // ---- per-point anchor model and goodness-of-fit summary ---------------------------------
    out.model_flux.resize(n);
    out.chi2_pt.resize(n);
    double chi2_peak = 0, chi2_base = 0;
    int n_peak = 0;
    for (size_t i = 0; i < n; i++) {
        double a = single_lens_mag(out.best, data[i].t);
        out.model_flux[i] = out.best.fs * a + out.best.fb;
        double r = (data[i].flux - out.model_flux[i]) / data[i].sigma;
        out.chi2_pt[i] = r * r;
        if (std::fabs(data[i].t - out.best.t0) < 2.0 * out.best.tE) { chi2_peak += r * r; n_peak++; }
        else chi2_base += r * r;
    }
    int n_par = (out.best.kind == SL_PSPL) ? 5 : (out.best.kind == SL_FSPL ? 6 : 7);
    out.dof = std::max(1, (int)n - n_par);
    out.chi2dof = out.best.chi2 / out.dof;
    {
        std::vector<double> c = out.chi2_pt;
        std::sort(c.begin(), c.end(), std::greater<double>());
        size_t k = std::max<size_t>(1, (n + 99) / 100);
        double top = 0;
        for (size_t i = 0; i < k && i < n; i++) top += c[i];
        out.worst1pct_frac = out.best.chi2 > 0 ? top / out.best.chi2 : 0;
    }
    out.n_peak = n_peak;
    out.chi2_peak = chi2_peak;
    out.chi2dof_peak = n_peak > 0 ? chi2_peak / n_peak : 0;
    out.chi2dof_base = ((int)n - n_peak) > 0 ? chi2_base / ((int)n - n_peak) : 0;
    night_index(data, &out.n_nights);
    return out;
}
