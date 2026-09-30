// Sanity check for the vendored VBMicrolensing library before building anything on top of it:
// 1. PSPLMag(u) must match the closed-form Paczynski (1986) point-source point-lens formula.
// 2. BinaryLightCurve with a vanishingly small mass ratio q must reduce to the same PSPL curve
//    (a binary lens with one component carrying ~all the mass should look like a single lens).
// 3. A textbook-scale binary lens (comparable-mass, projected separation near the Einstein
//    radius) must produce a light curve with real caustic-crossing structure: magnification
//    swinging well above the single-lens level, not just a smooth Paczynski bump.
#include <cstdio>
#include <cmath>
#include "../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.h"

static int failures = 0;

static void check(bool ok, const char *what) {
    if (!ok) { printf("FAIL: %s\n", what); failures++; }
    else printf("OK:   %s\n", what);
}

static double paczynski_A(double u) {
    double u2 = u * u;
    return (u2 + 2.0) / (u * sqrt(u2 + 4.0));
}

int main() {
    VBMicrolensing vbm;

    // 1. PSPL sanity.
    double max_pspl_err = 0;
    for (double u = 0.05; u < 3.0; u += 0.05) {
        double a_lib = vbm.PSPLMag(u);
        double a_ref = paczynski_A(u);
        double err = fabs(a_lib - a_ref) / a_ref;
        if (err > max_pspl_err) max_pspl_err = err;
    }
    printf("PSPL max relative error vs Paczynski formula: %.3e\n", max_pspl_err);
    check(max_pspl_err < 1e-6, "PSPLMag matches closed-form Paczynski formula");

    // 2. Binary with tiny q should reduce to PSPL. Parameter order (see
    // VBMicrolensingLibrary.cpp's BinaryLightCurve): [log(s), log(q), u0, alpha, log(rho),
    // log(tE), t0].
    {
        double pr[7] = { log(2.0), log(1e-8), 0.3, 0.7, log(1e-4), log(20.0), 0.0 };
        double max_err = 0;
        for (double t = -40; t <= 40; t += 1.0) {
            double u = sqrt(pr[2] * pr[2] + (t / 20.0) * (t / 20.0));
            double a_ref = paczynski_A(u);
            double a_lib = vbm.BinaryLightCurve(pr, t);
            double err = fabs(a_lib - a_ref) / a_ref;
            if (err > max_err) max_err = err;
        }
        printf("Binary(q=1e-8) vs PSPL max relative error: %.3e\n", max_err);
        check(max_err < 1e-3, "binary lens with q->0 reduces to the single-lens curve");
    }

    // 3. A real caustic-crossing case: comparable masses (q=0.5), separation near the
    // Einstein ring (s=1.0, where the binary caustic is largest), a source trajectory that
    // actually crosses through the central caustic region (u0 small).
    {
        double pr[7] = { log(1.0), log(0.5), 0.01, 0.3, log(1e-3), log(20.0), 0.0 };
        double max_mag = 0, min_mag = 1e9;
        int n = 2001;
        for (int i = 0; i < n; i++) {
            double t = -60.0 + 120.0 * i / (n - 1);
            double a = vbm.BinaryLightCurve(pr, t);
            if (a > max_mag) max_mag = a;
            if (a < min_mag) min_mag = a;
            if (!(a > 0) || !std::isfinite(a)) {
                printf("FAIL: non-finite/non-positive magnification %.6g at t=%.3f\n", a, t);
                failures++;
            }
        }
        printf("Caustic-crossing case: min mag=%.4f max mag=%.4f\n", min_mag, max_mag);
        // A central-caustic crossing at these parameters should push magnification well into
        // double digits at minimum; this is a loose sanity bound, not a precision check.
        check(max_mag > 10.0, "comparable-mass close binary shows real caustic amplification");
        check(min_mag > 0.9 && min_mag < 1.1, "far from the caustic, magnification relaxes back near baseline");
    }

    printf("\n%s\n", failures == 0 ? "All sanity checks passed." : "SOME CHECKS FAILED.");
    return failures == 0 ? 0 : 1;
}
