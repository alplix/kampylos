// Tiny helper: writes the same synthetic binary-lens light curve used in
// test_binary_recovery.cpp to a plain "time flux fluxerr" file, so fit_event's CLI plumbing
// (file loading, grid looping, output writing) can be exercised the same way a real BOINC
// work unit would use it. Not itself a correctness test -- test_binary_recovery.cpp already
// validates the fit; this just validates the I/O path.
#include <cstdio>
#include <cmath>
#include <random>
#include "../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.h"

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <out_file>\n", argv[0]); return 1; }
    VBMicrolensing vbm;
    const double log_s_true = log(1.0), log_q_true = log(0.5);
    const double t0_true = 5.0, u0_true = 0.02, tE_true = 25.0, alpha_true = 0.35, rho_true = 1e-3;
    const double fs_true = 800.0, fb_true = 150.0;
    double pr[7] = { log_s_true, log_q_true, u0_true, alpha_true, log(rho_true), log(tE_true), t0_true };

    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 1.0);

    FILE* f = fopen(argv[1], "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    fprintf(f, "# synthetic binary lens, injected log_s=%.4f log_q=%.4f t0=%.3f u0=%.4f tE=%.3f alpha=%.3f rho=%.4f\n",
            log_s_true, log_q_true, t0_true, u0_true, tE_true, alpha_true, rho_true);
    for (double t = -50; t <= 60; t += 0.3) {
        double A = vbm.BinaryLightCurve(pr, t);
        double flux_true = fs_true * A + fb_true;
        double sigma = 4.0 + 0.015 * flux_true;
        fprintf(f, "%.4f %.4f %.4f\n", t, flux_true + sigma * noise(rng), sigma);
    }
    fclose(f);
    fprintf(stderr, "wrote synthetic data to %s\n", argv[1]);
    return 0;
}
