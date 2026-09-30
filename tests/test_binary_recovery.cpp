// End-to-end recovery test for the binary-lens local fit (binary_fit.h): inject a known
// caustic-crossing binary lens light curve with realistic noise, then run the multi-start
// local optimizer AT THE TRUE (log s, log q) grid point and confirm it recovers a fit at least
// as good as the injected truth. This validates the piece that grid_search.h will call
// thousands of times per event -- if this doesn't work, nothing built on top of it can be
// trusted either.
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include "../src/binary_fit.h"

static int failures = 0;
static void check(bool ok, const char *what) {
    if (!ok) { printf("FAIL: %s\n", what); failures++; }
    else printf("OK:   %s\n", what);
}

int main() {
    VBMicrolensing vbm;

    // Ground truth: a close, comparable-mass binary with a real central-caustic crossing (the
    // same regime as test_vbm_sanity's caustic case, so we already know it produces genuine
    // structure and isn't just a PSPL curve in disguise).
    const double log_s_true = log(1.0), log_q_true = log(0.5);
    const double t0_true = 5.0, u0_true = 0.02, tE_true = 25.0, alpha_true = 0.35, rho_true = 1e-3;
    const double fs_true = 800.0, fb_true = 150.0;

    double pr_true[7] = { log_s_true, log_q_true, u0_true, alpha_true, log(rho_true), log(tE_true), t0_true };

    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 1.0);

    std::vector<DataPoint> data;
    for (double t = -50; t <= 60; t += 0.3) {
        double A = vbm.BinaryLightCurve(pr_true, t);
        double flux_true = fs_true * A + fb_true;
        double sigma = 4.0 + 0.015 * flux_true;
        data.push_back(DataPoint{ t, flux_true + sigma * noise(rng), sigma });
    }
    printf("Synthetic binary light curve: %zu points\n", data.size());
    printf("Injected: log_s=%.3f log_q=%.3f t0=%.3f u0=%.4f tE=%.3f alpha=%.3f rho=%.4f\n",
           log_s_true, log_q_true, t0_true, u0_true, tE_true, alpha_true, rho_true);

    // chi2 at the exact injected parameters, as the benchmark the fit must beat or match.
    std::vector<double> mag_at_truth(data.size());
    for (size_t i = 0; i < data.size(); i++) mag_at_truth[i] = vbm.BinaryLightCurve(pr_true, data[i].t);
    double chi2_at_truth = linear_flux_fit(data, mag_at_truth).chi2;

    // A single-lens (PSPL) anchor is what a real pipeline would have at this point (fit before
    // ever touching the binary grid) -- deliberately imprecise/wrong-ish the way a PSPL fit to
    // genuinely binary data often is, to make sure multistart's seed spread actually matters.
    double t0_anchor = 4.0, u0_anchor = 0.05, tE_anchor = 20.0;

    BinaryFitResult best = fit_binary_multistart(
        vbm, data, log_s_true, log_q_true, t0_anchor, u0_anchor, tE_anchor, /*rho_seed=*/1e-2, /*n_alpha_seeds=*/8
    );

    printf("Fit at true grid point: t0=%.4f u0=%.5f tE=%.4f alpha=%.4f rho=%.5f chi2=%.3f\n",
           best.t0, best.u0, best.tE, best.alpha, best.rho, best.chi2);
    printf("chi2 at injected truth: %.3f\n", chi2_at_truth);

    int ndof = (int)data.size() - 7;
    check(best.chi2 <= chi2_at_truth + 1e-3, "multistart found chi2 at or below injected truth");
    check(best.chi2 / ndof < 1.5 && best.chi2 / ndof > 0.5, "reduced chi2 near 1 at the true grid point");

    // Now check the grid actually discriminates: fit the SAME data at a clearly wrong (log s,
    // log q) grid point (a wide-separation, small-mass-ratio binary -- essentially a
    // perturbed single lens, a very different caustic topology from the true close/comparable
    // case) and confirm it fits noticeably worse. If it didn't, the grid dimension wouldn't be
    // doing anything and a single local fit would suffice -- which would contradict the whole
    // reason for grid-searching (log s, log q) in the first place.
    BinaryFitResult wrong = fit_binary_multistart(
        vbm, data, log(3.0), log(0.01), t0_anchor, u0_anchor, tE_anchor, /*rho_seed=*/1e-2, /*n_alpha_seeds=*/8
    );
    printf("Fit at a deliberately wrong grid point (s=3, q=0.01): chi2=%.3f\n", wrong.chi2);
    check(wrong.chi2 > best.chi2 + 20.0, "wrong (s,q) topology fits meaningfully worse than the true one -- the grid dimension matters");

    printf("\n%s\n", failures == 0 ? "All binary recovery checks passed." : "SOME CHECKS FAILED.");
    return failures == 0 ? 0 : 1;
}
