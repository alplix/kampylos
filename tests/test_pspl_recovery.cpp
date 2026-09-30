// End-to-end recovery test for the single-lens (PSPL) fitting machinery: simplex.h +
// lightcurve.h's linear flux fit + VBMicrolensing's PSPLMag, all together. Generates a
// synthetic light curve from known parameters plus realistic noise, fits it blind, and checks
// the fit recovers the injected parameters. This has to work before the binary-lens grid
// search (which reuses all the same machinery, just with a bigger parameter vector) is worth
// trusting.
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include "../src/simplex.h"
#include "../src/lightcurve.h"
#include "../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.h"

static int failures = 0;
static void check(bool ok, const char *what) {
    if (!ok) { printf("FAIL: %s\n", what); failures++; }
    else printf("OK:   %s\n", what);
}

int main() {
    VBMicrolensing vbm;

    // Ground truth.
    const double t0_true = 15.0, u0_true = 0.15, tE_true = 22.0;
    const double fs_true = 1000.0, fb_true = 200.0;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 1.0);

    std::vector<DataPoint> data;
    for (double t = -60; t <= 90; t += 0.5) {
        double u = sqrt(u0_true * u0_true + ((t - t0_true) / tE_true) * ((t - t0_true) / tE_true));
        double A = vbm.PSPLMag(u);
        double flux_true = fs_true * A + fb_true;
        double sigma = 5.0 + 0.02 * flux_true; // simple realistic-ish noise model
        double flux_obs = flux_true + sigma * noise(rng);
        data.push_back(DataPoint{ t, flux_obs, sigma });
    }
    printf("Synthetic PSPL light curve: %zu points, injected t0=%.3f u0=%.4f tE=%.3f\n",
           data.size(), t0_true, u0_true, tE_true);

    auto objective = [&](const std::vector<double>& p) -> double {
        double t0 = p[0], u0 = fabs(p[1]), tE = fabs(p[2]);
        if (tE < 0.1) return 1e18;
        std::vector<double> mag(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            double u = sqrt(u0 * u0 + ((data[i].t - t0) / tE) * ((data[i].t - t0) / tE));
            mag[i] = vbm.PSPLMag(u);
        }
        return linear_flux_fit(data, mag).chi2;
    };

    // Deliberately off-truth starting point, to make sure the optimizer is actually doing
    // the work and not just sitting near a lucky initial guess.
    std::vector<double> x0 = { 10.0, 0.3, 15.0 };
    std::vector<double> step = { 2.0, 0.05, 3.0 };
    SimplexResult res = nelder_mead(objective, x0, step, 5000, 1e-12, 1e-10);

    double t0_fit = res.x[0], u0_fit = fabs(res.x[1]), tE_fit = fabs(res.x[2]);
    printf("Fit result (%d iters, converged=%d): t0=%.4f u0=%.5f tE=%.4f chi2=%.3f (ndof=%zu)\n",
           res.iterations, res.converged, t0_fit, u0_fit, tE_fit, res.fval, data.size() - 5);

    double chi2_at_truth = objective({ t0_true, u0_true, tE_true });
    printf("chi2 at injected truth: %.3f vs chi2 at fit optimum: %.3f\n", chi2_at_truth, res.fval);

    // u0 and tE are known to be strongly covariant in single-lens fits (the peak
    // amplitude/width combination is what's actually well constrained, not each parameter
    // individually) -- a single noisy realization can land noticeably off the injected values
    // in each one separately while still being the correct chi2 minimum for that dataset. The
    // real check for "did the optimizer work" is whether it found a chi2 at least as good as
    // the injected truth, not whether it landed exactly on the truth.
    check(fabs(t0_fit - t0_true) < 0.1, "t0 recovered within 0.1");
    check(res.fval <= chi2_at_truth + 1e-6, "optimizer found chi2 at or below the injected truth (true minimization, not a stuck/buggy search)");
    check(res.fval / (data.size() - 5) < 1.5 && res.fval / (data.size() - 5) > 0.5, "reduced chi2 near 1 (good fit, not overfit/underfit)");
    check(fabs(u0_fit * tE_fit - u0_true * tE_true) / (u0_true * tE_true) < 0.15, "u0*tE (the actually well-constrained combination) recovered within 15%");

    printf("\n%s\n", failures == 0 ? "All recovery checks passed." : "SOME CHECKS FAILED.");
    return failures == 0 ? 0 : 1;
}
