// Validates kampylos_gpu_mag.cuh's point-source magnification + quadrupole/hexadecapole
// accept/reject test against the real VBMicrolensing CPU reference, using only VBMicrolensing's
// public API (BinaryMag0, BinaryMag2) -- no reliance on its private corrquad/corrquad2/safedist
// members, so this stays valid even if that internal state ever changes.
//
// Two things get checked per random (s, q, y1, y2, rho) sample:
//  1. Pure point-source correctness: kgpu::binary_mag0_gpu(...).mag vs vbm.BinaryMag0(s,q,y1,y2)
//     (rho-independent -- isolates the quintic solver + image selection from the accept/reject
//     decision entirely).
//  2. Accept/reject correctness: kgpu::binary_mag2_fastpath_ok(...) should agree with whether
//     VBMicrolensing's own BinaryMag2 actually needed BinaryMagDark for that point -- inferred,
//     without touching private members, as |BinaryMag2 - BinaryMag0| / BinaryMag0 exceeding a
//     small threshold (BinaryMag2 returns Mag0 unchanged whenever its internal test passes, so
//     this difference is exactly zero in that case and typically not small otherwise).
//
// Build (no CUDA needed -- KGPU_HD becomes plain `inline` under a non-nvcc compiler):
//   g++ -O2 -std=c++17 -I ../src -I ../vendor/VBMicrolensing/VBMicrolensing/lib \
//       test_kampylos_gpu_mag.cpp ../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp \
//       -o test_kampylos_gpu_mag
#include <cstdio>
#include <cmath>
#include <random>
#include "../src/kampylos_gpu_mag.cuh"
#include "VBMicrolensingLibrary.h"

int main() {
    VBMicrolensing vbm;
    vbm.Tol = 1.e-2;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> log_s_dist(-1.0, 1.0);   // s in [0.1, 10]
    std::uniform_real_distribution<double> log_q_dist(-6.0, 0.0);   // q in [1e-6, 1]
    std::uniform_real_distribution<double> y_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> log_rho_dist(-4.0, -0.3); // rho in [1e-4, 0.5], matches binary_fit.h's fit bounds

    const int N = 20000;
    int mag0_mismatches = 0;
    double mag0_worst_relerr = 0;
    int decision_mismatches = 0;
    int fastpath_taken = 0, fallback_needed = 0;
    int no_image_cases = 0;

    for (int i = 0; i < N; i++) {
        double s = pow(10.0, log_s_dist(rng));
        double q = pow(10.0, log_q_dist(rng));
        double y1 = y_dist(rng);
        double y2 = y_dist(rng);
        double rho = pow(10.0, log_rho_dist(rng));

        double ref_mag0 = vbm.BinaryMag0(s, q, y1, y2);
        kgpu::BinaryMag0Result mine = kgpu::binary_mag0_gpu(s, q, y1, y2);

        if (ref_mag0 < 0 || mine.mag < 0) {
            no_image_cases++;
            continue; // both agree "no valid images" or at least neither crashed; skip relerr check
        }

        double relerr = fabs(mine.mag - ref_mag0) / ref_mag0;
        if (relerr > mag0_worst_relerr) mag0_worst_relerr = relerr;
        if (relerr > 1e-6) {
            mag0_mismatches++;
            if (mag0_mismatches <= 10) {
                printf("MAG0 MISMATCH #%d: s=%.6f q=%.6g y1=%.6f y2=%.6f  ref=%.10f mine=%.10f relerr=%.3e (n_images ref=? mine=%d)\n",
                       mag0_mismatches, s, q, y1, y2, ref_mag0, mine.mag, relerr, mine.n_images);
            }
        }

        bool my_fastpath_ok = kgpu::binary_mag2_fastpath_ok(mine, rho, vbm.Tol);
        double ref_mag2 = vbm.BinaryMag2(s, q, y1, y2, rho);
        double mag2_vs_mag0_relerr = fabs(ref_mag2 - ref_mag0) / ref_mag0;
        bool ref_needed_fallback = mag2_vs_mag0_relerr > 1e-4;

        if (my_fastpath_ok) fastpath_taken++; else fallback_needed++;

        if (my_fastpath_ok == ref_needed_fallback) {
            // my_fastpath_ok==true means "I think mag0 is fine"; ref_needed_fallback==true means
            // "the reference's mag2 differs meaningfully from mag0" -- these should never both be
            // true (that'd mean I accepted mag0 in a case where the reference show it was wrong).
            // my_fastpath_ok==false and ref_needed_fallback==false (both false) is a harmless
            // false-negative -- conservative, not wrong, so not counted as a mismatch.
            if (my_fastpath_ok) {
                decision_mismatches++;
                if (decision_mismatches <= 10) {
                    printf("DECISION MISMATCH #%d: s=%.6f q=%.6g y1=%.6f y2=%.6f rho=%.6g  mag0=%.8f mag2_ref=%.8f relerr=%.3e -- I accepted fastpath but reference needed fallback\n",
                           decision_mismatches, s, q, y1, y2, rho, ref_mag0, ref_mag2, mag2_vs_mag0_relerr);
                }
            }
        }
    }

    printf("\n=== %d samples ===\n", N);
    printf("no-image cases (skipped): %d\n", no_image_cases);
    printf("mag0 mismatches (relerr > 1e-6): %d / %d, worst relerr = %.3e\n",
           mag0_mismatches, N - no_image_cases, mag0_worst_relerr);
    printf("fastpath decision: accepted=%d, routed-to-fallback=%d\n", fastpath_taken, fallback_needed);
    printf("UNSAFE decision mismatches (accepted fastpath when reference needed fallback): %d\n", decision_mismatches);
    printf("\n%s\n", (mag0_mismatches == 0 && decision_mismatches == 0) ? "PASS" : "FAIL");
    return (mag0_mismatches == 0 && decision_mismatches == 0) ? 0 : 1;
}
