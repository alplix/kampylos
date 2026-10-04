// Standalone CLI: evaluates a fitted binary-lens model's magnitude at a dense grid of times,
// for kampylos_lightcurve.php's light-curve plot (public, one per flagged/reviewed event --
// shows the real photometry against the fitted model so a visitor can see for themselves
// whether a claimed signal is actually traced out by real data points, not just asserted by a
// chi2 number). Kept as a separate CLI rather than linked into the PHP process directly since
// VBMicrolensing is a C++ library with no PHP binding -- PHP shells out to this, same pattern as
// every other "real physics needs real C++" boundary in this project.
#include <cstdio>
#include <cmath>
#include "VBMicrolensingLibrary.h"

int main(int argc, char** argv) {
    if (argc < 13) {
        fprintf(stderr, "usage: %s t_min t_max n_points log_s log_q u0 alpha log_rho log_tE t0 fs fb\n", argv[0]);
        return 1;
    }
    double t_min = atof(argv[1]), t_max = atof(argv[2]);
    int n = atoi(argv[3]);
    double log_s = atof(argv[4]), log_q = atof(argv[5]);
    double u0 = atof(argv[6]), alpha = atof(argv[7]);
    double log_rho = atof(argv[8]), log_tE = atof(argv[9]), t0 = atof(argv[10]);
    double fs = atof(argv[11]), fb = atof(argv[12]);
    const double zp = 18.0; // same convention as lightcurve.h's load_photometry()

    VBMicrolensing vbm;
    double pr[7] = { log_s, log_q, u0, alpha, log_rho, log_tE, t0 };
    for (int i = 0; i < n; i++) {
        double t = (n > 1) ? t_min + (t_max - t_min) * i / (n - 1) : t_min;
        double A = vbm.BinaryLightCurve(pr, t);
        double flux = fs * A + fb;
        double mag = (flux > 0) ? (zp - 2.5 * log10(flux)) : 99.0;
        printf("%.6f %.6f\n", t, mag);
    }
    return 0;
}
