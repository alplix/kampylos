// Validates single_lens.h's finite-source magnifications against two independent references:
//   1. VBMicrolensing's own table-based ESPLMag2 (uniform) and ESPLMagDark (linear limb
//      darkening, a1) -- needs the ESPL.tbl table shipped with VBMicrolensing (data/ESPL.tbl);
//      At u = 0 the closed-form limit is used instead (VBMicrolensing is ~1% off there);
//   2. a brute-force 2-D midpoint integration of the point-lens magnification over the disk
//      (polar grid centred on the SOURCE, fine enough to make the 1/r singularity harmless).
// Also checks the linear flux fit is the exact least-squares solution (bug fix 2): a perturbed
// (fs, fb) must never give a lower chi2 than the returned one.
//
// Build (from vendor/kampylos):
//   g++ -O2 -std=c++17 -I src -I vendor/VBMicrolensing/VBMicrolensing/lib tests/test_single_lens.cpp \
//       vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp -o test_single_lens
// Run from vendor/kampylos (so the table path below resolves). Exit code 0 = pass.
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include "single_lens.h"
#include "VBMicrolensingLibrary.h"

static double brute_uniform(double u, double rho, double a1) {
    // polar grid around the source centre
    const int NR = 1500, NP = 1500;
    double num = 0, den = 0;
    for (int i = 0; i < NR; i++) {
        double r = (i + 0.5) / NR * rho;
        double mu = std::sqrt(std::max(0.0, 1.0 - (r / rho) * (r / rho)));
        double I = 1.0 - a1 + a1 * mu;
        for (int j = 0; j < NP; j++) {
            double ph = (j + 0.5) / NP * 2.0 * M_PI;
            double x = u + r * std::cos(ph), y = r * std::sin(ph);
            double d = std::sqrt(x * x + y * y);
            num += I * pspl_mag(d) * r;
            den += I * r;
        }
    }
    return num / den;
}

int main() {
    int fails = 0;
    VBMicrolensing vbm;
    VBMicrolensing::SetESPLtablefile((char*)"vendor/VBMicrolensing/VBMicrolensing/data/ESPL.tbl");
    vbm.Tol = 1e-5;
    vbm.RelTol = 1e-5;

    const double rhos[] = { 0.001, 0.01, 0.05, 0.2 };
    const double zs[] = { 0.0, 0.3, 0.8, 0.99, 1.01, 1.5, 3.0, 9.0, 12.0, 40.0 };
    double worst_u = 0, worst_ld = 0, worst_bf = 0;
    for (double rho : rhos) {
        for (double z : zs) {
            double u = z * rho;
            if (u == 0) u = 1e-9;
            double mine = fspl_mag_uniform(u, rho);
            vbm.a1 = 0;
            double ref = vbm.ESPLMag2(u, rho);
            double e = std::fabs(mine / ref - 1);
            worst_u = std::max(worst_u, e);
            for (double a1 : { 0.4, 0.7 }) {
                vbm.a1 = a1;
                // At u = 0 VBMicrolensing's annulus integrator is itself ~1% off; compare with the
                // closed form there instead: A_LD(0) = [(1-a) A_U(0,rho) + a int_0^1 (1-m^2)
                // sqrt(1 + 4/(rho^2 (1-m^2))) dm] / (1 - a/3), integral done with 200k midpoints.
                double ref_ld;
                if (z == 0.0) {
                    double I = 0; const int N = 200000;
                    for (int k = 0; k < N; k++) { double m = (k + 0.5) / N, f = 1 - m * m; I += f * std::sqrt(1 + 4 / (rho * rho * f)) / N; }
                    ref_ld = ((1 - a1) * std::sqrt(1 + 4 / (rho * rho)) + a1 * I) / (1 - a1 / 3);
                } else {
                    ref_ld = vbm.ESPLMagDark(u, rho);
                }
                double mine_ld = fspl_mag_ld(u, rho, a1);
                worst_ld = std::max(worst_ld, std::fabs(mine_ld / ref_ld - 1));
            }
            if (rho == 0.05 && (z == 0.3 || z == 0.99 || z == 1.5 || z == 12.0)) {
                for (double a1 : { 0.0, 0.6 }) {
                    double bf = brute_uniform(u, rho, a1);
                    double mine2 = fspl_mag_ld(u, rho, a1);
                    worst_bf = std::max(worst_bf, std::fabs(mine2 / bf - 1));
                    printf("  rho=%.3f u/rho=%5.2f a1=%.1f  mine=%.6f brute=%.6f\n", rho, z, a1, mine2, bf);
                }
            }
        }
    }
    printf("max rel. difference vs VBMicrolensing ESPLMag2 (uniform): %.2e\n", worst_u);
    printf("max rel. difference vs VBMicrolensing ESPLMagDark (LD):   %.2e\n", worst_ld);
    printf("max rel. difference vs brute-force 2-D integration:       %.2e\n", worst_bf);
    if (worst_u > 2e-3) { printf("FAIL uniform\n"); fails++; }
    if (worst_ld > 2e-3) { printf("FAIL limb-darkened\n"); fails++; }
    if (worst_bf > 3e-3) { printf("FAIL brute force\n"); fails++; }  // the brute-force grid's own error is ~1e-3 near u ~ rho

    // Exact least squares: no perturbation of (fs, fb) may beat the returned chi2.
    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0, 1);
    std::vector<DataPoint> data;
    std::vector<double> mag;
    for (int i = 0; i < 500; i++) {
        double t = i * 0.2;
        double A = pspl_mag(std::sqrt(0.05 * 0.05 + ((t - 50) / 12) * ((t - 50) / 12)));
        DataPoint d; d.t = t; d.sigma = 0.01 + 0.01 * (i % 3); d.flux = 0.7 * A + 0.3 + nd(rng) * d.sigma;
        data.push_back(d); mag.push_back(A);
    }
    FluxFit ff = linear_flux_fit(data, mag);
    double worst_gain = 0;
    for (double dfs : { -1e-4, 1e-4, 0.0 })
        for (double dfb : { -1e-4, 1e-4, 0.0 }) {
            double c = 0;
            for (size_t i = 0; i < data.size(); i++) {
                double r = (data[i].flux - ((ff.fs + dfs) * mag[i] + ff.fb + dfb)) / data[i].sigma;
                c += r * r;
            }
            worst_gain = std::max(worst_gain, ff.chi2 - c);
        }
    printf("linear_flux_fit: fs=%.6f fb=%.6f chi2=%.4f degenerate=%d, best chi2 gain from perturbing fs/fb: %.3e\n",
           ff.fs, ff.fb, ff.chi2, ff.degenerate, worst_gain);
    if (worst_gain > 1e-6 || ff.degenerate != 0) { printf("FAIL flux fit not exact least squares\n"); fails++; }

    printf(fails ? "FAILED (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
