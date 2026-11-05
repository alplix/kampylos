// The seed set of kampylos_make_seeds() is made of mirror pairs (src/binary_fit.h, "MIRROR PAIRS", plan of
// 2026-11-05 section 4.1): (t0, u0, ln tE, alpha, ln rho) and (t0, -u0, ln tE, -alpha, ln rho) give the same
// light curve. Checks, on the synthetic planet cell (log10 s = 0.2, log10 q = -3) and the bundled OGLE cell
// (0.3, -2.5):
//   pairs     every seed's mirror is another seed of the set (an involution without fixed points), and the
//             set built with -DKAMPYLOS_SEEDS_HALF=1 holds exactly one seed of each pair;
//   same      the objective (kampylos_binary_chi2) at the two starting points of every pair agrees to rounding
//             (relative difference below 1e-9; alpha and 2 pi - alpha differ in the last bit).
// It does not say that the two Nelder-Mead runs of a pair end in the same minimum: their simplex steps are not
// mirrored, so they do not (see docs/BENCHMARK.md for the measured effect of dropping one of them).
// Prints PASS and exits 0 only when every check passed. A few seconds.
//
// Build (from the repo root; any VBMicrolensing form):
//   VB=vendor/VBMicrolensing/VBMicrolensing/lib
//   clang++ -O2 -std=c++17 -I src -I $VB tests/test_mirror_seeds.cpp $VB/VBMicrolensingLibrary.cpp -o test_mirror_seeds
// Built a second time with -DKAMPYLOS_SEEDS_HALF=1, it checks the half set instead: no two of its seeds are
// mirrors of each other.
#include <cstdio>
#include <cmath>
#include <vector>
#include "kampylos_wu.h"

static double wrap(double a) {   // alpha in [0, 2 pi)
    double t = std::fmod(a, 2 * M_PI);
    return t < 0 ? t + 2 * M_PI : t;
}
static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(a)); }

static int check_cell(const char* name, const char* path, bool mag, double log10_s, double log10_q) {
    std::vector<DataPoint> data = load_photometry(path, mag);
    const double ls = log10_s * log(10.0), lq = log10_q * log(10.0);
    KampylosWuAnchor a = kampylos_compute_anchor(data);
    BinarySeedSet set = kampylos_make_seeds(a.seeds, exp(ls), exp(lq));
    const int n = (int)set.size();
    int failures = 0;
#if KAMPYLOS_SEEDS_HALF
    // the half set: no two seeds may be mirrors of each other
    int pairs_inside = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            const auto &x = set.x0[i], &y = set.x0[j];
            if (near(x[0], y[0], 1e-12) && near(x[1], -y[1], 1e-12) && near(x[2], y[2], 1e-12) && near(x[4], y[4], 1e-12) &&
                std::fabs(wrap(x[3]) - wrap(-y[3])) < 1e-9)
                pairs_inside++;
        }
    printf("%s: half set of %d seeds, %d mirror pairs inside it (want 0)  %s\n", name, n, pairs_inside, pairs_inside ? "FAIL" : "ok");
    return pairs_inside != 0;
#else
    std::vector<int> mirror(n, -1);
    for (int i = 0; i < n; i++) {
        const auto& x = set.x0[i];
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            const auto& y = set.x0[j];
            if (near(x[0], y[0], 1e-12) && near(x[1], -y[1], 1e-12) && near(x[2], y[2], 1e-12) && near(x[4], y[4], 1e-12) &&
                std::fabs(wrap(x[3]) - wrap(-y[3])) < 1e-9 && set.step[i] == set.step[j]) {
                if (mirror[i] >= 0) { printf("%s: seed %d has two mirrors (%d, %d)\n", name, i, mirror[i], j); failures++; }
                mirror[i] = j;
            }
        }
    }
    int unpaired = 0, not_involution = 0, plus_half = 0;
    for (int i = 0; i < n; i++) {
        if (mirror[i] < 0) { unpaired++; continue; }
        if (mirror[mirror[i]] != i) not_involution++;
        plus_half += set.x0[i][1] > 0;
    }
    printf("%s: %d seeds, %d without a mirror, %d not paired both ways, %d with u0 > 0 (want 0, 0, %d)  %s\n", name, n,
           unpaired, not_involution, plus_half, n / 2, (unpaired || not_involution || plus_half != n / 2) ? "FAIL" : "ok");
    failures += unpaired || not_involution || plus_half != n / 2;

    VBMicrolensing vbm;
    std::vector<double> mag_buf(data.size());
    const double fs_bound = flux_fit_bound(data);
    double worst = 0;
    for (int i = 0; i < n; i++) {
        if (mirror[i] < i) continue;
        double f1 = kampylos_binary_chi2(vbm, data, ls, lq, set.x0[i], mag_buf, fs_bound);
        double f2 = kampylos_binary_chi2(vbm, data, ls, lq, set.x0[mirror[i]], mag_buf, fs_bound);
        worst = std::max(worst, std::fabs(f1 - f2) / std::max(1.0, std::fabs(f1)));
    }
    printf("%s: largest relative chi2 difference at the two starts of a pair %.2e (want < 1e-9)  %s\n", name, worst,
           worst < 1e-9 ? "ok" : "FAIL");
    failures += !(worst < 1e-9);
    return failures;
#endif
}

int main() {
    int f = check_cell("planet", "tests/regression/synthetic_planet.dat", false, 0.2, -3.0);
    f += check_cell("ogle", "tests/sample_ogle_blg1900.dat", true, 0.3, -2.5);
    printf(f ? "FAIL (%d)\n" : "PASS\n", f);
    return f ? 1 : 0;
}
