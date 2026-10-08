// Writes the synthetic planetary light curve used by tests/regression/run_regression.py
// (tests/regression/synthetic_planet.dat, committed, so the regression run needs no generator):
// a wide planet with log10 s = 0.2 (s = 1.585) and log10 q = -3 (q = 1e-3), i.e. EXACTLY on the
// log10 grid cell (0.2, -3). The source passes just outside the planetary caustic (a ~30-sigma bump)
// ~19 days after the peak (no caustic crossing) with a small source (rho = 1e-4),
// so almost every point stays on the point-source fast path and a grid cell fits in minutes.
// If a WU's grid were read with the wrong logarithm, the cell "0.2 0.2 1 -3 -3 1" would be fitted
// at s = e^0.2 = 1.22, q = e^-3 = 0.05 and could not reproduce the bump -- that is the
// end-to-end check of the grid-convention fix.
//
// Build (from vendor/kampylos):
//   g++ -O2 -std=c++17 -I vendor/VBMicrolensing/VBMicrolensing/lib tests/gen_synthetic_planet.cpp \
//       vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp -o gen_synthetic_planet
//   ./gen_synthetic_planet tests/regression/synthetic_planet.dat
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include "VBMicrolensingLibrary.h"

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <out_file>\n", argv[0]); return 1; }
    VBMicrolensing vbm;
    const double s = pow(10.0, 0.2), q = 1e-3;
    const double t0 = 2456000.0, u0 = 0.1, tE = 20.0, alpha = 0.135, rho = 1e-4;
    const double fs = 1.0, fb = 0.3;
    double pr[7] = { log(s), log(q), u0, alpha, log(rho), log(tE), t0 };

    std::vector<double> ts;
    for (double t = t0 - 60; t <= t0 + 60; t += 0.4) ts.push_back(t);                 // survey cadence
    for (double t = t0 + 17.0; t <= t0 + 21.0; t += 0.04) ts.push_back(t + 0.003);     // dense on the bump
    std::sort(ts.begin(), ts.end());

    std::mt19937 rng(11);
    std::normal_distribution<double> noise(0.0, 1.0);
    FILE* f = fopen(argv[1], "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    fprintf(f, "# synthetic planet: log10 s=0.2 log10 q=-3 t0=%.1f u0=%.4f tE=%.1f alpha=%.2f rho=%.0e fs=%.1f fb=%.1f (flux)\n",
            t0, u0, tE, alpha, rho, fs, fb);
    for (double t : ts) {
        double A = vbm.BinaryLightCurve(pr, t);
        double flux = fs * A + fb;
        double sigma = 0.004 + 0.01 * flux;
        fprintf(f, "%.5f %.6f %.6f\n", t, flux + sigma * noise(rng), sigma);
    }
    fclose(f);
    fprintf(stderr, "wrote %zu points to %s\n", ts.size(), argv[1]);
    return 0;
}
