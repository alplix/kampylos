// Validates NelderMeadStepper against simplex.h's own blocking nelder_mead(): for the same
// (x0, step, max_iter, ftol, xtol) and the same objective, does driving the stepper round-by-round
// reach the identical (x, fval, converged) result as the original? This is a refactor of a proven
// algorithm (same reflect/expand/contract/shrink logic, same constants), not a new one -- the
// thing that actually needs checking is that re-expressing it as a step machine didn't change its
// behavior, not whether Nelder-Mead itself works.
//
// Build: g++ -O2 -std=c++17 -I ../src -I ../vendor/VBMicrolensing/VBMicrolensing/lib
//   test_nelder_mead_stepper.cpp ../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp
//   -o test_nelder_mead_stepper
#include <cstdio>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vector>
#include <random>
#include <functional>
#include "simplex.h"
#include "nelder_mead_stepper.h"
#include "lightcurve.h"
#include "binary_fit.h"
#include "VBMicrolensingLibrary.h"

// Drives a stepper to completion against a plain objective function, round by round.
SimplexResultLite drive(const std::function<double(const std::vector<double>&)>& f, std::vector<double> x0, std::vector<double> step, int max_iter, double ftol, double xtol) {
    NelderMeadStepper stepper(x0, step, max_iter, ftol, xtol);
    while (!stepper.done()) {
        const auto& pts = stepper.pending_points();
        std::vector<double> fvals(pts.size());
        for (size_t i = 0; i < pts.size(); i++) fvals[i] = f(pts[i]);
        stepper.submit_results(fvals);
    }
    auto r = stepper.result();
    return r;
}

int n_checked = 0, n_mismatch = 0;

void check(const char* label, const std::function<double(const std::vector<double>&)>& f,
           std::vector<double> x0, std::vector<double> step, int max_iter = 2000, double ftol = 1e-10, double xtol = 1e-10) {
    SimplexResult ref = nelder_mead(f, x0, step, max_iter, ftol, xtol);
    SimplexResultLite got = drive(f, x0, step, max_iter, ftol, xtol);
    n_checked++;
    double dx = 0;
    for (size_t j = 0; j < ref.x.size(); j++) { double d = ref.x[j] - got.x[j]; dx += d * d; }
    dx = sqrt(dx);
    double df = fabs(ref.fval - got.fval);
    bool ok = (dx < 1e-9) && (df < 1e-9) && (ref.converged == got.converged);
    if (!ok) {
        n_mismatch++;
        printf("MISMATCH [%s]: ref.fval=%.12f got.fval=%.12f |dx|=%.3e ref.converged=%d got.converged=%d ref.iter=%d got.iter=%d\n",
            label, ref.fval, got.fval, dx, ref.converged, got.converged, ref.iterations, got.iterations);
    } else {
        printf("ok [%s]: fval=%.8f converged=%d ref.iter=%d got.iter=%d\n", label, ref.fval, ref.converged, ref.iterations, got.iterations);
    }
}

int main() {
    // 1) Standard analytic test functions (sphere, Rosenbrock-ish in 5D) -- cheap, exercises
    // reflect/expand/contract/shrink across many trajectories without needing real photometry.
    auto sphere = [](const std::vector<double>& x) {
        double s = 0; for (double v : x) s += v * v; return s;
    };
    auto rosenbrock5 = [](const std::vector<double>& x) {
        double s = 0;
        for (size_t i = 0; i + 1 < x.size(); i++) {
            double a = x[i + 1] - x[i] * x[i];
            double b = 1 - x[i];
            s += 100 * a * a + b * b;
        }
        return s;
    };
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-3, 3);
    for (int i = 0; i < 20; i++) {
        std::vector<double> x0 = { u(rng), u(rng), u(rng), u(rng), u(rng) };
        std::vector<double> step = { 0.5, 0.5, 0.5, 0.5, 0.5 };
        check(("sphere#" + std::to_string(i)).c_str(), sphere, x0, step);
    }
    for (int i = 0; i < 20; i++) {
        std::vector<double> x0 = { u(rng), u(rng), u(rng), u(rng), u(rng) };
        std::vector<double> step = { 0.3, 0.3, 0.3, 0.3, 0.3 };
        check(("rosenbrock5#" + std::to_string(i)).c_str(), rosenbrock5, x0, step, 5000);
    }

    // 2) The real thing: fit_binary_local's actual objective, same synthetic light curve
    // approach used in the GPU kernel tests, same multistart seeding pattern -- this is the
    // function the stepper will actually drive once wired to fit_binary_multistart.
    VBMicrolensing vbm;
    const double true_log_s = 0.05, true_log_q = -2.3;
    const double true_t0 = 750.0, true_u0 = 0.08, true_tE = 25.0, true_alpha = 1.2, true_rho = 0.01;
    const double true_fs = 1.0, true_fb = 0.2;
    std::mt19937 rng2(42);
    std::uniform_real_distribution<double> cadence(0.2, 1.5);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::vector<DataPoint> data;
    double t = 0.0;
    while (t < 1500.0) {
        t += cadence(rng2);
        double pr[7] = { true_log_s, true_log_q, true_u0, true_alpha, log(true_rho), log(true_tE), true_t0 };
        double mag = vbm.BinaryLightCurve(pr, t);
        double sigma = 0.02;
        DataPoint dp; dp.t = t; dp.flux = true_fs * mag + true_fb + noise(rng2) * sigma; dp.sigma = sigma;
        data.push_back(dp);
    }
    printf("\nReal objective: %zu points\n", data.size());

    std::vector<double> mag_scratch(data.size());
    auto objective = [&](const std::vector<double>& x) -> double {
        double t0 = x[0], u0 = x[1], log_tE = x[2], alpha = x[3], log_rho = x[4];
        double tE = exp(log_tE);
        if (tE < 0.01 || tE > 10000.0) return 1e18;
        double rho = exp(log_rho);
        if (rho < 1e-6 || rho > 1.0) return 1e18;
        double pr[7] = { true_log_s, true_log_q, u0, alpha, log_rho, log_tE, t0 };
        for (size_t i = 0; i < data.size(); i++) {
            double a = vbm.BinaryLightCurve(pr, data[i].t);
            if (!std::isfinite(a) || a <= 0) return 1e18;
            mag_scratch[i] = a;
        }
        return linear_flux_fit(data, mag_scratch).chi2;
    };

    // Same seed pattern fit_binary_multistart uses: 8 alpha seeds x 2 u0 magnitudes x 2 signs x
    // 3 rho seeds, each run through the same x0/step/max_iter/n_restarts=0 (checking the base
    // nelder_mead call here, restarts are just repeated calls of the same primitive).
    int n_alpha_seeds = 8;
    double rho_seeds[] = { 1e-3, 1e-4, 1e-2 };
    double u0_mag_seeds[] = { fabs(true_u0), 0.02 };
    int idx = 0;
    for (int i = 0; i < n_alpha_seeds; i++) {
        double alpha_seed = 2.0 * M_PI * i / n_alpha_seeds;
        for (double u0_mag : u0_mag_seeds) {
            for (double u0_sign : { 1.0, -1.0 }) {
                for (double rs : rho_seeds) {
                    std::vector<double> x0 = { true_t0, u0_sign * u0_mag, log(true_tE), alpha_seed, log(rs) };
                    std::vector<double> step = { std::max(0.5, true_tE * 0.05), 0.05, 0.2, 0.3, 0.5 };
                    check(("kampylos_seed#" + std::to_string(idx++)).c_str(), objective, x0, step, 400, 1e-9, 1e-8);
                }
            }
        }
    }

    printf("\n=== %d checked, %d mismatches ===\n", n_checked, n_mismatch);
    printf("%s\n", n_mismatch == 0 ? "PASS" : "FAIL");
    return n_mismatch == 0 ? 0 : 1;
}
