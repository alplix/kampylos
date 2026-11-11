// History-free evaluations (src/kampylos_vbm_state.h, KAM-030, KAM-012): with -DKAMPYLOS_RESET_STATE=1 a
// candidate's chi2 must not depend on what the process computed before it. Checks, on the synthetic planet
// cell (log10 s = 0.2, log10 q = -3) and the bundled OGLE light curve:
//   objective    300 candidates around the cell's seeds through kampylos_binary_chi2() (the CPU app's
//                objective), in index order and in reverse order: bitwise equal chi2;
//   sensitivity  the same loop without the reset must differ in at least one candidate, or the test above
//                could not have failed (the warm start of the root solver is what it guards against);
//   completion   the GPU apps' host completion (kampylos_complete_candidate) of the same candidates, from
//                partial sums made as a kernel would make them, in both orders: bitwise equal chi2;
//   seed 96      KAM-030's case: seed 96 of the OGLE cell (log10 s = 0.3, log10 q = -2.5) fitted alone and
//                after seeds 0, 24, 48 and 72 by fit_binary_local_steps(): bitwise equal result.
// Prints PASS and exits 0 only when every check passed. About 15 s on an Apple M4.
//
// Build: the library must be prepared with vendor/patch_vbm_thread_local.py --reset (any form), and the
// src/ headers include it through ../vendor, so build in a copy of the tree, e.g. the one that
// scripts/bench/build_variant.sh leaves in $KAMPYLOS_WORK/build/<name> when run with VBMX=reset:
//   cd ~/kampylos-work/build/<name>; VB=vendor/VBMicrolensing/VBMicrolensing/lib
//   clang++ -O2 -std=c++17 -DKAMPYLOS_RESET_STATE=1 -I src -I $VB tests/test_vbm_state_reset.cpp vbm.o -o test_vbm_state_reset
//   ./test_vbm_state_reset            (from the tree's root: it reads tests/regression/synthetic_planet.dat)
// or in place after `python3 vendor/patch_vbm_thread_local.py --strip --reset` (restore the submodule with
// `git -C vendor/VBMicrolensing checkout -- .` afterwards).
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "kampylos_wu.h"
#include "kampylos_gpu_complete.h"

#if !KAMPYLOS_RESET_STATE
#error "build with -DKAMPYLOS_RESET_STATE=1 (see the build line above)"
#endif

static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// Deterministic on every platform (std::*_distribution is not specified bit for bit).
struct Lcg {
    unsigned long long s;
    double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return (double)(s >> 11) / 9007199254740992.0; }
};

// The candidates: the cell's seeds, each moved by up to half its simplex step in every coordinate.
static std::vector<std::vector<double>> make_candidates(const BinarySeedSet& seeds, int n) {
    Lcg r{ 20261105ULL };
    std::vector<std::vector<double>> c;
    for (int k = 0; k < n; k++) {
        const auto& x0 = seeds.x0[k % seeds.size()];
        const auto& st = seeds.step[k % seeds.size()];
        std::vector<double> x = x0;
        for (size_t j = 0; j < x.size(); j++) x[j] += (r.next() - 0.5) * st[j];
        c.push_back(x);
    }
    return c;
}

// The objective without the reset, as the apps compute it with the switch off.
static double chi2_without_reset(VBMicrolensing& vbm, const std::vector<DataPoint>& data, double ls, double lq,
                                 const std::vector<double>& x, std::vector<double>& mag, double fs_bound) {
    double tE = exp(x[2]), rho = exp(x[4]);
    if (tE < 0.01 || tE > 10000.0 || rho < 1e-6 || rho > 1.0) return 1e18;
    double pr[7] = { ls, lq, x[1], x[3], x[4], x[2], x[0] };
    for (size_t i = 0; i < data.size(); i++) {
        double a = vbm.BinaryLightCurve(pr, data[i].t);
        if (!std::isfinite(a) || a <= 0) return 1e18;
        mag[i] = a;
    }
    return linear_flux_fit(data, mag, fs_bound).chi2;
}

// What a GPU kernel hands to the host for candidate x: the six sums over the points that pass BinaryMag2's
// point-source test, and the list of the others (here computed on the CPU with a separate instance, so the
// instance that completes the candidates sees only the completion's own calls).
static CandidateResult kernel_like(VBMicrolensing& k, const std::vector<LightCurvePoint>& d, double s, double q, const Candidate& c) {
    CandidateResult r{};
    double tE = exp(c.log_tE), rho = exp(c.log_rho), rho2 = rho * rho, salpha = sin(c.alpha), calpha = cos(c.alpha);
    if (tE < 0.01 || tE > 10000.0 || rho < 1e-6 || rho > 1.0) { r.poisoned = 1; return r; }
    for (int i = 0; i < (int)d.size(); i++) {
        double tn = (d[i].t - c.t0) / tE;
        double y1 = c.u0 * salpha - tn * calpha, y2 = -c.u0 * calpha - tn * salpha;
        double A = k.BinaryMag0(s, q, y1, fabs(y2));
        if (!(A > 0) || !std::isfinite(A)) { r.poisoned = 1; return r; }
        bool fast = k.corrquad * 6 * (rho2 + 1.e-4 * k.Tol) < k.Tol && k.corrquad2 * 256 * (rho2 + 1.e-8) < 1 && k.safedist > 4 * rho2;
        if (!fast) {
            if (r.n_bad < KAMPYLOS_MAX_BAD_POINTS) r.bad_idx[r.n_bad++] = i; else r.overflow = 1;
            continue;
        }
        double w = 1.0 / (d[i].sigma * d[i].sigma), f = d[i].flux;
        r.S_AA += w * A * A; r.S_A1 += w * A; r.S_11 += w; r.S_Ay += w * A * f; r.S_1y += w * f; r.S_yy += w * f * f;
    }
    return r;
}

static int count_differences(const std::vector<double>& a, const std::vector<double>& b) {
    int n = 0;
    for (size_t i = 0; i < a.size(); i++) n += !same_bits(a[i], b[i]);
    return n;
}

int main() {
    int failures = 0;
    std::vector<DataPoint> planet = load_photometry("tests/regression/synthetic_planet.dat", false);
    const double ls = log(pow(10.0, 0.2)), lq = log(pow(10.0, -3.0)), s = exp(ls), q = exp(lq);
    KampylosWuAnchor a = kampylos_compute_anchor(planet);
    BinarySeedSet seeds = kampylos_make_seeds(a.seeds, s, q);
    const int n = 300;
    std::vector<std::vector<double>> cand = make_candidates(seeds, n);
    const double fs_bound = flux_fit_bound(planet);
    std::vector<double> mag(planet.size());

    {   // objective, both orders, with the reset
        VBMicrolensing vbm;
        std::vector<double> fwd(n), rev(n);
        for (int k = 0; k < n; k++) fwd[k] = kampylos_binary_chi2(vbm, planet, ls, lq, cand[k], mag, fs_bound);
        for (int k = n - 1; k >= 0; k--) rev[k] = kampylos_binary_chi2(vbm, planet, ls, lq, cand[k], mag, fs_bound);
        int d = count_differences(fwd, rev);
        printf("objective:   %d of %d candidates differ between the two orders (want 0)  %s\n", d, n, d ? "FAIL" : "ok");
        failures += d != 0;
    }
    {   // sensitivity: the same without the reset
        VBMicrolensing vbm;
        std::vector<double> fwd(n), rev(n);
        for (int k = 0; k < n; k++) fwd[k] = chi2_without_reset(vbm, planet, ls, lq, cand[k], mag, fs_bound);
        for (int k = n - 1; k >= 0; k--) rev[k] = chi2_without_reset(vbm, planet, ls, lq, cand[k], mag, fs_bound);
        int d = count_differences(fwd, rev);
        printf("sensitivity: %d of %d candidates differ without the reset (want > 0)     %s\n", d, n, d ? "ok" : "FAIL");
        failures += d == 0;
    }
    {   // the GPU apps' completion step
        std::vector<LightCurvePoint> lc;
        for (const auto& p : planet) lc.push_back({ p.t, p.flux, p.sigma });
        VBMicrolensing kern, host;
        std::vector<Candidate> cs;
        std::vector<CandidateResult> rs;
        int n_bad = 0;
        for (const auto& x : cand) {
            Candidate c{ x[0], x[1], x[2], x[3], x[4] };
            cs.push_back(c);
            rs.push_back(kernel_like(kern, lc, s, q, c));
            n_bad += rs.back().n_bad;
        }
        std::vector<double> fwd(n), rev(n);
        for (int k = 0; k < n; k++) fwd[k] = kampylos_complete_candidate(host, rs[k], cs[k], lc.data(), (int)lc.size(), s, q, fs_bound).chi2;
        for (int k = n - 1; k >= 0; k--) rev[k] = kampylos_complete_candidate(host, rs[k], cs[k], lc.data(), (int)lc.size(), s, q, fs_bound).chi2;
        int d = count_differences(fwd, rev);
        printf("completion:  %d of %d candidates differ between the two orders (want 0; %d finite-source points)  %s\n",
               d, n, n_bad, d ? "FAIL" : "ok");
        failures += d != 0;
    }
    {   // KAM-030: seed 96 of the OGLE cell, alone and after seeds 0, 24, 48, 72
        std::vector<DataPoint> ogle = load_photometry("tests/sample_ogle_blg1900.dat", true);
        const double ols = log(pow(10.0, 0.3)), olq = log(pow(10.0, -2.5));
        KampylosWuAnchor oa = kampylos_compute_anchor(ogle);
        BinarySeedSet os = kampylos_make_seeds(oa.seeds, exp(ols), exp(olq));
        VBMicrolensing v1, v2;
        BinaryFitResult alone = fit_binary_local_steps(v1, ogle, ols, olq, os.x0[96], os.step[96]);
        for (int k : { 0, 24, 48, 72 }) fit_binary_local_steps(v2, ogle, ols, olq, os.x0[k], os.step[k]);
        BinaryFitResult after = fit_binary_local_steps(v2, ogle, ols, olq, os.x0[96], os.step[96]);
        bool same = same_bits(alone.chi2, after.chi2) && same_bits(alone.t0, after.t0) && same_bits(alone.u0, after.u0) &&
                    same_bits(alone.tE, after.tE) && same_bits(alone.alpha, after.alpha) && same_bits(alone.rho, after.rho);
        printf("seed 96:     alone chi2 %.4f, after seeds 0-72 chi2 %.4f (want the same bits)  %s\n", alone.chi2, after.chi2,
               same ? "ok" : "FAIL");
        failures += !same;
    }
    printf(failures ? "FAIL (%d)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
