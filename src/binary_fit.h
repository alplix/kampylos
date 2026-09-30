#pragma once
// The core of the actual search: for a binary lens with (s, q) FIXED at one grid point, find
// the best-fitting (t0, u0, tE, alpha, rho) by local optimization, and report the resulting
// chi-squared. The outer grid search (grid_search.h) calls this once per (log s, log q) cell,
// which is the part of parameter space that genuinely can't be locally optimized away --
// different (s, q) give qualitatively different caustic topologies, so the fit landscape isn't
// smooth in those two dimensions the way it is in the other five.
#include <vector>
#include <cmath>
#include <algorithm>
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
    double log_s, log_q, t0, u0, tE, alpha, rho;
    double fs, fb, chi2;
};

// One local optimization run from one starting point. pr[] order matches VBMicrolensing's
// BinaryLightCurve convention: [log(s), log(q), u0, alpha, log(rho), log(tE), t0].
// n_restarts controls how many restart-from-converged passes follow the initial simplex run
// (see the comment below) -- 0 for a cheap coarse screen, 2 for a full refine.
inline BinaryFitResult fit_binary_local(
    VBMicrolensing& vbm,
    const std::vector<DataPoint>& data,
    double log_s, double log_q,
    double t0_seed, double u0_seed, double tE_seed, double alpha_seed, double rho_seed,
    int max_iter = 400, int n_restarts = 2
) {
    std::vector<double> mag(data.size());

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
        return linear_flux_fit(data, mag).chi2;
    };

    std::vector<double> x0 = { t0_seed, u0_seed, log(tE_seed), alpha_seed, log(rho_seed) };
    std::vector<double> step = { std::max(0.5, tE_seed * 0.05), 0.05, 0.2, 0.3, 0.5 };
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
    FluxFit ff = linear_flux_fit(data, mag);
    out.fs = ff.fs; out.fb = ff.fb;
    return out;
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
// out based on an unreliable cheap proxy. Slower, but every seed that finds the true minimum
// keeps it, rather than being screened away before it gets the chance to.
//
// u0's magnitude is seeded from multiple scales, not just the single-lens (PSPL) pre-fit's own
// estimate: PSPL fits to a genuinely binary, caustic-crossing light curve are known to often
// lock onto the wrong impact-parameter scale entirely, so trusting that one value as the only
// u0 seed silently loses the true solution whenever that happens (this is exactly what caused
// the very first version of this function to miss the true minimum when anchored from a real
// automated PSPL pre-fit instead of a hand-picked good guess).
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
