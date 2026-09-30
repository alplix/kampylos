// Smoke test against a real downloaded event (not synthetic): confirms load_photometry() and
// the PSPL pre-fit don't choke on real survey data -- gaps, outliers, whatever quirks a real
// alert-system light curve has that a synthetic test with clean Gaussian noise wouldn't catch.
// Not a correctness check on the fit result itself (we don't know this event's true parameters
// without an independent reference), just "does the pipeline run and produce something sane."
#include <cstdio>
#include <cmath>
#include "../src/lightcurve.h"
#include "../src/pspl_prefit.h"
#include "../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.h"

int main() {
    VBMicrolensing vbm;
    auto data = load_photometry("tests/sample_ogle_blg1900.dat", /*input_is_mag=*/true);
    printf("Loaded %zu real data points\n", data.size());
    if (data.empty()) { printf("FAIL: no data loaded\n"); return 1; }

    double tmin = data[0].t, tmax = data[0].t;
    double fmin = data[0].flux, fmax = data[0].flux;
    for (auto& d : data) {
        tmin = std::min(tmin, d.t); tmax = std::max(tmax, d.t);
        fmin = std::min(fmin, d.flux); fmax = std::max(fmax, d.flux);
        if (!std::isfinite(d.t) || !std::isfinite(d.flux) || !std::isfinite(d.sigma) || d.sigma <= 0) {
            printf("FAIL: bad data point t=%.4f flux=%.4f sigma=%.4f\n", d.t, d.flux, d.sigma);
            return 1;
        }
    }
    printf("Time range: %.2f to %.2f (span %.1f days)\n", tmin, tmax, tmax - tmin);
    printf("Flux range: %.2f to %.2f\n", fmin, fmax);

    PsplFit fit = pspl_prefit(vbm, data);
    printf("PSPL pre-fit: t0=%.4f u0=%.5f tE=%.4f chi2=%.3f (ndof=%zu, reduced=%.3f)\n",
           fit.t0, fit.u0, fit.tE, fit.chi2, data.size() - 3, fit.chi2 / (data.size() - 3));

    bool ok = std::isfinite(fit.chi2) && fit.tE > 0 && fit.u0 > 0 &&
              fit.t0 > tmin - 500 && fit.t0 < tmax + 500;
    printf("\n%s\n", ok ? "Smoke test passed (pipeline runs cleanly on real data)." : "FAIL: pre-fit produced nonsense.");
    return ok ? 0 : 1;
}
