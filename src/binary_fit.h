#pragma once
// The core of the actual search: for a binary lens with (s, q) FIXED at one grid point, find
// the best-fitting (t0, u0, tE, alpha, rho) by local optimization, and report the resulting
// chi-squared. The outer grid search calls this once per grid cell, which is the part of
// parameter space that genuinely can't be locally optimized away -- different (s, q) give
// qualitatively different caustic topologies, so the fit landscape isn't smooth in those two
// dimensions the way it is in the other five.
//
// CONVENTION (read this first): every log_s/log_q passed INTO a function in this file is a
// NATURAL log, because that is what VBMicrolensing's BinaryLightCurve expects (it does
// s = exp(pr[0]), q = exp(pr[1])). Work units may describe their grid in log10 (the "log10"
// marker, see kampylos_wu.h); the caller converts with kampylos_grid_to_ln() before calling in
// here and writes the grid value, not the ln value, to the result file.
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include "simplex.h"
#include "lightcurve.h"
#include "../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.h"

// M_PI is a common extension (glibc, etc.), not standard C++ -- MSVC/mingw only define it
// when _USE_MATH_DEFINES is set before <cmath> is first included, which isn't guaranteed given
// the include order here. Defining it directly, guarded, works on every platform regardless.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct BinaryFitResult {
    // log_s/log_q: as passed in (natural log) by the fit functions below; the work-unit driver
    // overwrites them with the cell's GRID coordinates before writing the result line.
    double log_s, log_q, t0, u0, tE, alpha, rho;
    double fs, fb, chi2;
};

// One local optimization run from an explicit starting point/simplex step.
// x = (t0, u0, ln tE, alpha, ln rho); pr[] order matches VBMicrolensing's BinaryLightCurve:
// [ln s, ln q, u0, alpha, ln rho, ln tE, t0]. n_restarts controls how many
// restart-from-converged passes follow the initial simplex run.
inline BinaryFitResult fit_binary_local_steps(
    VBMicrolensing& vbm,
    const std::vector<DataPoint>& data,
    double log_s, double log_q,
    const std::vector<double>& x0, const std::vector<double>& step,
    int max_iter = 400, int n_restarts = 2
) {
    std::vector<double> mag(data.size());
    const double fs_bound = flux_fit_bound(data);

    auto objective = [&](const std::vector<double>& x) -> double {
        double t0 = x[0], u0 = x[1], log_tE = x[2], alpha = x[3], log_rho = x[4];
        double tE = exp(log_tE);
        if (tE < 0.01 || tE > 10000.0) return 1e18;
        double rho = exp(log_rho);
        if (rho < 1e-6 || rho > 1.0) return 1e18;

        double pr[7] = { log_s, log_q, u0, alpha, log_rho, log_tE, t0 };
        for (size_t i = 0; i < data.size(); i++) {
            double a = vbm.BinaryLightCurve(pr, data[i].t);
            if (!std::isfinite(a) || a <= 0) return 1e18;
            mag[i] = a;
        }
        return linear_flux_fit(data, mag, fs_bound).chi2;
    };

    SimplexResult res = nelder_mead(objective, x0, step, max_iter, 1e-9, 1e-8);
    // Nelder-Mead's own convergence test (simplex has shrunk enough) can fire while still
    // sitting in a shallow local trap, especially near a caustic where the chi2 surface has
    // real structure on scales smaller than the initial step size. Restarting once from the
    // converged point with fresh (smaller) steps costs little and reliably breaks out of that
    // kind of premature convergence without needing a different algorithm.
    for (int restart = 0; restart < n_restarts; restart++) {
        std::vector<double> step2 = { 0.3, 0.02, 0.08, 0.15, 0.2 };
        SimplexResult res2 = nelder_mead(objective, res.x, step2, max_iter, 1e-10, 1e-9);
        if (res2.fval < res.fval) res = res2; else break;
    }

    BinaryFitResult out;
    out.log_s = log_s; out.log_q = log_q;
    out.t0 = res.x[0]; out.u0 = res.x[1]; out.tE = exp(res.x[2]);
    out.alpha = res.x[3]; out.rho = exp(res.x[4]);
    out.chi2 = res.fval;

    double pr[7] = { log_s, log_q, out.u0, out.alpha, log(out.rho), log(out.tE), out.t0 };
    for (size_t i = 0; i < data.size(); i++) mag[i] = vbm.BinaryLightCurve(pr, data[i].t);
    FluxFit ff = linear_flux_fit(data, mag, fs_bound);
    out.fs = ff.fs; out.fb = ff.fb;
    return out;
}

// One local optimization run from one starting point (legacy signature, natural-log s/q).
inline BinaryFitResult fit_binary_local(
    VBMicrolensing& vbm,
    const std::vector<DataPoint>& data,
    double log_s, double log_q,
    double t0_seed, double u0_seed, double tE_seed, double alpha_seed, double rho_seed,
    int max_iter = 400, int n_restarts = 2
) {
    std::vector<double> x0 = { t0_seed, u0_seed, log(tE_seed), alpha_seed, log(rho_seed) };
    std::vector<double> step = { std::max(0.5, tE_seed * 0.05), 0.05, 0.2, 0.3, 0.5 };
    return fit_binary_local_steps(vbm, data, log_s, log_q, x0, step, max_iter, n_restarts);
}

// Multistart: tries every combination of alpha seed x u0 sign x u0 magnitude x rho seed, each
// with the FULL restart-backed local optimization (fit_binary_local's default n_restarts=2).
//
// A cheap-screen-then-refine version was tried first (rank all seeds with a short, no-restart
// run, only fully refine the best few) to cut cost -- it was faster but unreliable: on this
// same caustic-crossing test case it landed on chi2=523 instead of the true 386, because a
// short run isn't long enough to tell a real basin from a false one near a caustic (the same
// reason fit_binary_local itself needs restarts to escape premature convergence in the first
// place). So this stays brute-force: every seed gets the full treatment, nothing is filtered
// out based on an unreliable cheap proxy.
//
// u0's magnitude is seeded from multiple scales, not just the single-lens pre-fit's own
// estimate: PSPL fits to a genuinely binary, caustic-crossing light curve are known to often
// lock onto the wrong impact-parameter scale entirely.
//
// Kept for the unit tests; the work-unit drivers use kampylos_make_seeds()/fit_binary_seeds().
inline BinaryFitResult fit_binary_multistart(
    VBMicrolensing& vbm,
    const std::vector<DataPoint>& data,
    double log_s, double log_q,
    double t0_anchor, double u0_anchor, double tE_anchor,
    double rho_seed = 1e-3,
    int n_alpha_seeds = 8
) {
    const double rho_seeds[] = { rho_seed, 1e-4, 1e-2 };
    const double u0_mag_seeds[] = { fabs(u0_anchor), 0.02 };

    BinaryFitResult best;
    best.chi2 = 1e300;
    for (int i = 0; i < n_alpha_seeds; i++) {
        double alpha_seed = 2.0 * M_PI * i / n_alpha_seeds;
        for (double u0_mag : u0_mag_seeds) {
            for (double u0_sign : { 1.0, -1.0 }) {
                for (double rs : rho_seeds) {
                    BinaryFitResult r = fit_binary_local(
                        vbm, data, log_s, log_q,
                        t0_anchor, u0_sign * u0_mag, tE_anchor, alpha_seed, rs
                    );
                    if (r.chi2 < best.chi2) best = r;
                }
            }
        }
    }
    return best;
}

// Calls fit_binary_multistart twice -- once from the single-lens pre-fit's optimized
// (t0, u0, tE) anchor, once from the independent, purely data-driven fallback anchor (peak time
// + excess-region width, u0 fixed at a generic 0.3 guess) -- and keeps whichever converges to
// the lower chi2. Kept for the unit tests; superseded by kampylos_make_seeds() (which contains
// both of these seed grids plus the single-lens-reproducing seeds).
inline BinaryFitResult fit_binary_multistart_dual_anchor(
    VBMicrolensing& vbm,
    const std::vector<DataPoint>& data,
    double log_s, double log_q,
    double t0_anchor, double u0_anchor, double tE_anchor,
    double t0_anchor_fallback, double tE_anchor_fallback,
    double rho_seed = 1e-3,
    int n_alpha_seeds = 8
) {
    BinaryFitResult r1 = fit_binary_multistart(vbm, data, log_s, log_q, t0_anchor, u0_anchor, tE_anchor, rho_seed, n_alpha_seeds);
    BinaryFitResult r2 = fit_binary_multistart(vbm, data, log_s, log_q, t0_anchor_fallback, 0.3, tE_anchor_fallback, rho_seed, n_alpha_seeds);
    return (r1.chi2 < r2.chi2) ? r1 : r2;
}

// ---------------------------------------------------------------------------------------------
// 2026-10-08: one seed list shared by EVERY backend (CPU fit_binary_seeds() below, and the
// CUDA/OpenCL/Metal batched drivers), so the backends search exactly the same starting points
// and can be compared cell by cell. A seed is a 5-vector x0 = (t0, u0, ln tE, alpha, ln rho)
// in VBMicrolensing's BinaryLightCurve convention (origin at the centre of mass, primary at
// x = -q s/(1+q), secondary at x = +s/(1+q), tE and rho in units of the TOTAL-mass Einstein
// radius) plus its initial simplex step.
struct BinarySeedSet {
    std::vector<std::vector<double>> x0, step;
    void add(const std::vector<double>& x, const std::vector<double>& st) { x0.push_back(x); step.push_back(st); }
    size_t size() const { return x0.size(); }
};

struct BinarySeedAnchors {
    // (A) optimized single-lens anchor (best of PSPL/FSPL/FSPL+LD)
    double t0, u0, tE, rho;
    // (B) optimizer-free fallback anchor (peak time + excess width), see pspl_prefit.h
    double t0_raw, tE_raw;
    // (C) the uniform-source single-lens solution, re-expressed below as binary-lens
    // parameters that reproduce it (rho may be ~0 when the source is effectively a point)
    double sl_t0, sl_u0, sl_tE, sl_rho;
    // (D) other distinct single-lens minima (different timescale), see SingleLensAnchor::pspl_alt
    std::vector<std::array<double, 3>> alt;   // (t0, u0, tE)
};

// Bug fix 4 (2026-10-08): on the vetted event MOA-2003-BLG-004, 44 of 100 grid cells ended
// with chi2 WORSE than the PSPL anchor -- impossible for a converged fit wherever the binary
// can mimic a single lens (small q, or s far from 1: the companion's caustic is then small and
// can be avoided). The old seeds all started the source trajectory relative to the CENTRE OF
// MASS with the single-lens tE, which for a wide binary is the wrong place (the event happens
// around one component, offset from the centre of mass) and the wrong timescale (one
// component's Einstein radius is sqrt(m_c/M) of the total). Seed set (C) maps the single-lens
// solution exactly onto each component:
//     tE_b = tE_1L / sqrt(m_c),  rho_b = rho_1L * sqrt(m_c),  u0_c = +-u0_1L * sqrt(m_c),
//     u0_b = u0_c + x_c sin(alpha),  t0_b = t0_1L + tE_b x_c cos(alpha)
// (m_c = mass fraction, x_c = component position on the binary axis), for 8 trajectory angles,
// so seeds start AT the single-lens light curve with the companion's caustic off the
// trajectory, and the optimizer can only go down from there. (Near s ~ 1 with q >~ 0.1 the
// resonant caustic cannot be avoided, so there a binary genuinely may not reproduce a single
// lens; those cells legitimately stay above the anchor.)
inline BinarySeedSet kampylos_make_seeds(const BinarySeedAnchors& a, double s, double q, int n_alpha_seeds = 8) {
    BinarySeedSet set;
    // (A) + (B): the original multistart grid around each anchor -- 8 alpha x 2 |u0| x 2 signs
    // x 3 rho = 96 seeds each, the same as fit_binary_multistart_dual_anchor() except that the
    // first rho seed comes from the finite-source anchor when it measured one.
    double rho_seed = (a.rho >= 1e-4 && a.rho <= 0.5) ? a.rho : 1e-3;
    struct Anc { double t0, u0, tE; };
    const Anc ancs[2] = { { a.t0, a.u0, a.tE }, { a.t0_raw, 0.3, a.tE_raw } };
    for (const Anc& an : ancs) {
        const double rho_seeds[] = { rho_seed, 1e-4, 1e-2 };
        const double u0_mag_seeds[] = { std::fabs(an.u0), 0.02 };
        for (int i = 0; i < n_alpha_seeds; i++) {
            double alpha_seed = 2.0 * M_PI * i / n_alpha_seeds;
            for (double u0_mag : u0_mag_seeds)
                for (double u0_sign : { 1.0, -1.0 })
                    for (double rs : rho_seeds)
                        set.add({ an.t0, u0_sign * u0_mag, log(an.tE), alpha_seed, log(rs) },
                                { std::max(0.5, an.tE * 0.05), 0.05, 0.2, 0.3, 0.5 });
        }
    }
    // (C): single-lens-reproducing seeds around the primary (always) and the secondary (only
    // for q >= 0.1 -- below that a secondary-only event would need tE_b > 3.3x the observed tE
    // with the source far from the primary, which seeds (A)/(B) already reach).
    double rho_1l = std::min(std::max(a.sl_rho, 1e-5), 0.5);
    for (int comp = 0; comp < 2; comp++) {
        if (comp == 1 && q < 0.1) continue;
        double m_c = comp == 0 ? 1.0 / (1.0 + q) : q / (1.0 + q);
        double x_c = comp == 0 ? -q * s / (1.0 + q) : s / (1.0 + q);
        double th = std::sqrt(m_c);
        double tE_b = a.sl_tE / th, rho_b = std::max(rho_1l * th, 2e-6);
        for (int i = 0; i < n_alpha_seeds; i++) {
            double alpha = 2.0 * M_PI * (i + 0.5) / n_alpha_seeds; // between (A)'s angles
            for (double sgn : { 1.0, -1.0 }) {
                double u0_c = sgn * std::fabs(a.sl_u0) * th;
                double u0_b = u0_c + x_c * sin(alpha);
                double t0_b = a.sl_t0 + tE_b * x_c * cos(alpha);
                set.add({ t0_b, u0_b, log(tE_b), alpha, log(rho_b) },
                        { std::max(0.05, 0.02 * tE_b), std::max(0.005, 0.2 * std::fabs(u0_c)), 0.1, 0.1, 0.5 });
            }
        }
    }
    // (D): 16 seeds (8 alpha x 2 u0 signs) around each alternative single-lens minimum.
    for (const auto& al : a.alt) {
        for (int i = 0; i < n_alpha_seeds; i++) {
            double alpha_seed = 2.0 * M_PI * i / n_alpha_seeds;
            for (double u0_sign : { 1.0, -1.0 })
                set.add({ al[0], u0_sign * std::max(std::fabs(al[1]), 1e-4), log(al[2]), alpha_seed, log(rho_seed) },
                        { std::max(0.5, al[2] * 0.05), 0.05, 0.2, 0.3, 0.5 });
        }
    }
    return set;
}

// CPU driver for a seed set: every seed gets the full restart-backed local optimization
// (fit_binary_local_steps), best chi2 wins. ln_s/ln_q are natural logs.
inline BinaryFitResult fit_binary_seeds(
    VBMicrolensing& vbm, const std::vector<DataPoint>& data,
    double ln_s, double ln_q, const BinarySeedSet& seeds
) {
    BinaryFitResult best;
    best.chi2 = 1e300;
    for (size_t k = 0; k < seeds.x0.size(); k++) {
        BinaryFitResult r = fit_binary_local_steps(vbm, data, ln_s, ln_q, seeds.x0[k], seeds.step[k]);
        if (r.chi2 < best.chi2) best = r;
    }
    return best;
}
