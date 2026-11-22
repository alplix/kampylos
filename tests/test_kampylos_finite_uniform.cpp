// Validates kgpu::finite_uniform_try (kampylos_gpu_mag.cuh) against VBMicrolensing: for random
// (s, q, y1, y2, rho) it counts how often the quadrature is accepted, and how far the accepted values are from a
// tight-tolerance VBMicrolensing reference (Tol = 1e-6) and from VBMicrolensing's default (Tol = 1e-2).
// Build (plain g++, no GPU needed):
//   g++ -O2 -std=c++17 -I ../src -I ../vendor/VBMicrolensing/VBMicrolensing/lib test_kampylos_finite_uniform.cpp \
//       ../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp -o test_kampylos_finite_uniform
#include <cstdio>
#include <cmath>
#include <random>
#include <string>
#include "kampylos_gpu_mag.cuh"
#include "VBMicrolensingLibrary.h"

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 20000;
    bool resonant = argc > 2 && std::string(argv[2]) == "resonant";   // stress: lens near the resonant/central-caustic regime, source close to the lens axis
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> U(0, 1);
    VBMicrolensing vtight, vdef;
    vtight.Tol = 1e-6;
    double cq2edge[7] = {1, 3, 10, 30, 100, 1e4, 1e300}; double cqedge[7] = {1, 3, 10, 30, 100, 1e4, 1e300};
    long long bn2[7] = {0}, bn1[7] = {0}; double bm2[7] = {0}, bm1[7] = {0};
    long long n = 0, bad_default = 0, accepted = 0, acc_bad1 = 0, acc_bad3 = 0, d_bad1 = 0, d_bad3 = 0, need_fs = 0;
    double acc_max = 0, d_max = 0;
    for (int it = 0; it < N; it++) {
        double s = pow(10.0, -1.0 + 2.0 * U(rng));
        double q = pow(10.0, -5.0 * U(rng));
        double rho = pow(10.0, -3.5 + 2.5 * U(rng));
        double y1 = -1.5 + 3.0 * U(rng), y2 = -1.5 + 3.0 * U(rng);
        if (resonant) { s = 0.7 + 0.8 * U(rng); q = pow(10.0, -2.0 + 2.0 * U(rng)); rho = pow(10.0, -2.5 + 1.0 * U(rng)); y1 = -0.5 + U(rng); y2 = -0.5 + U(rng); }
        kgpu::BinaryMag0Result r = kgpu::binary_mag0_gpu(s, q, y1, fabs(y2));
        if (!(r.mag > 0)) continue;
        if (kgpu::binary_mag2_fastpath_ok(r, rho)) continue;      // the fast path already handles it: not our case
        n++;
        double Aref = vtight.BinaryMag2(s, q, y1, y2, rho);
        double Ad = vdef.BinaryMag2(s, q, y1, y2, rho);
        double ed = fabs(Ad - Aref) / Aref; if (ed > d_max) d_max = ed; if (ed > 1e-3) d_bad1++; if (ed > 3e-3) d_bad3++;
        double Af;
        if (kgpu::finite_uniform_try(s, q, y1, y2, rho, r.mag, r.corrquad2 * 256.0 * (rho * rho + 1.e-8), &Af)) {
            accepted++;
            { double rho2 = rho * rho; double cq = r.corrquad * 6.0 * (rho2 + 1e-6) / 1e-2, cq2 = r.corrquad2 * 256.0 * (rho2 + 1e-8); double ee = fabs(Af - Aref) / Aref;
              int b2 = 0; while (b2 < 6 && cq2 >= cq2edge[b2]) b2++; int b1 = 0; while (b1 < 6 && cq >= cqedge[b1]) b1++;
              bn2[b2]++; if (ee > bm2[b2]) bm2[b2] = ee; bn1[b1]++; if (ee > bm1[b1]) bm1[b1] = ee; }
            double e = fabs(Af - Aref) / Aref; if (e > acc_max) acc_max = e; if (e > 1e-3) { acc_bad1++; printf("  outlier s=%.5g q=%.4g y1=%.5g y2=%.5g rho=%.4g A0=%.5g Af=%.6g ref=%.6g err=%.2e\n", s, q, y1, y2, rho, r.mag, Af, Aref, e); } if (e > 3e-3) acc_bad3++;
        }
    }
    printf("points the fast path rejects: %lld\n", n);
    printf("quadrature accepted:          %lld (%.1f %%)\n", accepted, 100.0 * accepted / (n ? n : 1));
    printf("quadrature vs tight reference: max %.2e, >1e-3: %lld, >3e-3: %lld\n", acc_max, acc_bad1, acc_bad3);
    printf("VBM default vs tight reference (same points, all %lld): max %.2e, >1e-3: %lld, >3e-3: %lld\n", n, d_max, d_bad1, d_bad3);
    printf("accepted points by hexadecapole indicator cq2 (fast path needs < 1): "); for (int b = 0; b < 7; b++) printf("[<%.0e: n=%lld max=%.1e] ", cq2edge[b], bn2[b], bm2[b]); printf("\n");
    printf("accepted points by quadrupole indicator cq/Tol (fast path needs < 1): "); for (int b = 0; b < 7; b++) printf("[<%.0e: n=%lld max=%.1e] ", cqedge[b], bn1[b], bm1[b]); printf("\n");
    bool ok = acc_bad3 == 0 && acc_max < 1e-2;
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
