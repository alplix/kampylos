// Validates the Metal objective (kampylos_metal_fit.mm) against the CPU objective point by
// point, on a real light curve and at one grid cell:
//   * device setup must pass the double-single self-test (fast math off);
//   * at 1 400 candidates -- a seed->solution line plus a random box around the planet
//     solution -- Metal must reject no candidate the CPU accepts, and agree in chi2 to a small
//     fraction of a unit (the fit's Nelder-Mead tolerance is 1e-9, so systematic noise here is
//     what used to pull the Metal fit into the wrong basin: 691 vs 411 on this very cell).
// With --cell it also runs the whole cell through metal_fit_binary_seeds() and compares the
// finalised chi2 with the CPU's fit_binary_seeds() (minutes on an M1).
//
// Build (from the repo root):
//   clang++ -O2 -std=c++17 -fobjc-arc -I src -I vendor/VBMicrolensing/VBMicrolensing/lib tests/test_metal_fit.mm \
//       src/kampylos_metal_fit.mm vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp \
//       -framework Metal -framework Foundation -o test_metal_fit
//   ./test_metal_fit tests/regression/synthetic_planet.dat [--cell]
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include "kampylos_wu.h"
#include "kampylos_gpu_types.h"
#include "kampylos_metal_fit.h"

static int fails = 0;
static void check(bool ok, const char* what) {
    printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) fails++;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <photometry (flux)> [--cell]\n", argv[0]); return 2; }
    bool full_cell = argc > 2 && !strcmp(argv[2], "--cell");
    std::vector<DataPoint> data = load_photometry(argv[1], false);
    printf("loaded %zu points\n", data.size());
    // the synthetic planet's cell: log10 s = 0.2, log10 q = -3; true (t0, u0, tE, alpha, rho)
    const double s = pow(10.0, 0.2), q = 1e-3;
    const double truth[5] = { 2456000.0, 0.1, log(20.0), 0.135, log(1e-4) };

    printf("== device\n");
    MetalFitContext ctx;
    std::string err;
    bool dev = metal_fit_select_device(ctx, &err);
    check(dev, "metal_fit_select_device (includes the double-single self-test)");
    if (!dev) { printf("   %s\n", err.c_str()); printf("\nFAIL\n"); return 1; }
    metal_fit_upload_light_curve(ctx, data);

    printf("== objective vs CPU\n");
    std::vector<Candidate> cands;
    const double seed[5] = { 2455999.991, 0.112916, log(20.7488), 0.0, log(1e-3) };   // a seed-(A)-like start
    for (int i = 0; i <= 400; i++) {
        double f = i / 400.0; Candidate c;
        c.t0 = seed[0] + f * (truth[0] - seed[0]); c.u0 = seed[1] + f * (truth[1] - seed[1]);
        c.log_tE = seed[2] + f * (truth[2] - seed[2]); c.alpha = seed[3] + f * (truth[3] - seed[3]);
        c.log_rho = seed[4] + f * (truth[4] - seed[4]);
        cands.push_back(c);
    }
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> U(0, 1);
    for (int i = 0; i < 1000; i++) {
        Candidate c;
        c.t0 = truth[0] + (U(rng) - 0.5) * 0.4; c.u0 = 0.05 + 0.1 * U(rng); c.log_tE = log(15.0 + 10.0 * U(rng));
        c.alpha = truth[3] + (U(rng) - 0.5) * 0.6; c.log_rho = -13.0 + 9.0 * U(rng);
        cands.push_back(c);
    }
    std::vector<double> metal = metal_fit_eval_chi2(ctx, s, q, cands);

    VBMicrolensing vbm;
    std::vector<double> mag(data.size());
    const double fs_bound = flux_fit_bound(data);
    int dead_cpu = 0, dead_metal_only = 0, n_live = 0;
    double worst = 0, sum = 0;
    for (size_t k = 0; k < cands.size(); k++) {
        const Candidate& c = cands[k];
        double pr[7] = { log(s), log(q), c.u0, c.alpha, c.log_rho, c.log_tE, c.t0 };
        double cpu = 1e18;
        bool ok = true;
        for (size_t i = 0; i < data.size() && ok; i++) {
            mag[i] = vbm.BinaryLightCurve(pr, data[i].t);
            if (!std::isfinite(mag[i]) || mag[i] <= 0) ok = false;
        }
        if (ok) cpu = linear_flux_fit(data, mag, fs_bound).chi2;
        bool dm = metal[k] >= 1e17, dc = cpu >= 1e17;
        dead_cpu += dc;
        if (dm && !dc) {
            dead_metal_only++;
            if (dead_metal_only <= 5) printf("   Metal rejects, CPU accepts: cand %zu cpu chi2 %.3f (t0=%.3f u0=%.4f tE=%.3f alpha=%.3f rho=%.2e)\n",
                                             k, cpu, c.t0, c.u0, exp(c.log_tE), c.alpha, exp(c.log_rho));
        }
        if (!dm && !dc) { n_live++; double d = fabs(metal[k] - cpu); sum += d; worst = std::max(worst, d); }
    }
    printf("   %zu candidates: dead on CPU %d, dead on Metal only %d; live %d: mean |dchi2| %.4f, worst %.4f\n",
           cands.size(), dead_cpu, dead_metal_only, n_live, n_live ? sum / n_live : 0.0, worst);
    check(dead_metal_only == 0, "no candidate rejected by Metal that the CPU accepts");
    check(n_live > 1000, "the test candidates are mostly live");
    check(n_live && sum / n_live < 0.05, "mean |chi2_metal - chi2_cpu| < 0.05");
    check(worst < 0.5, "worst |chi2_metal - chi2_cpu| < 0.5");
    printf("   command buffers retried so far: %ld\n", metal_fit_buffers_retried(ctx));

    if (full_cell) {
        printf("== whole cell (Metal vs CPU, same seeds)\n");
        KampylosWuAnchor anchor = kampylos_compute_anchor(data);
        BinarySeedSet seeds = kampylos_make_seeds(anchor.seeds, s, q);
        BinaryFitResult rm = metal_fit_binary_seeds(ctx, data, log(s), log(q), seeds);
        KampylosCellDiag dm = kampylos_finalize_cell(vbm, data, anchor, rm);
        BinaryFitResult rc = fit_binary_seeds(vbm, data, log(s), log(q), seeds);
        KampylosCellDiag dc = kampylos_finalize_cell(vbm, data, anchor, rc);
        printf("   Metal: chi2 %.4f (t0=%.6f u0=%.6f tE=%.6f alpha=%.6f rho=%.3e)\n", rm.chi2, rm.t0, rm.u0, rm.tE, rm.alpha, rm.rho);
        printf("   CPU:   chi2 %.4f (t0=%.6f u0=%.6f tE=%.6f alpha=%.6f rho=%.3e)\n", rc.chi2, rc.t0, rc.u0, rc.tE, rc.alpha, rc.rho);
        (void)dm; (void)dc;
        check(fabs(rm.chi2 - rc.chi2) <= 1.0 + 0.002 * rc.chi2, "cell chi2 agrees with the CPU within the regression tolerance");
    }

    printf("\n%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
