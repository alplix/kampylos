// Validates gpu_fit_binary_multistart() against the real CPU fit_binary_multistart(): same
// synthetic light curve (real injected signal via VBMicrolensing), same grid cell, same seeding
// -- does the GPU-batched version converge to the same best-fit solution the proven CPU
// multistart search does? Exact bit-identical isn't the bar here (hundreds of iterations of
// floating-point accumulation in a different order can drift at the last few digits) -- what
// matters is reaching the same minimum, which for a well-posed fit means matching chi2 closely
// and recovering parameters close to the true injected ones.
//
// Build (Windows, MSVC host compiler):
//   nvcc -O2 -std=c++17 -gencode arch=compute_120,code=sm_120 ^
//       -I ../src -I ../vendor/VBMicrolensing/VBMicrolensing/lib ^
//       test_gpu_fit_binary.cu ../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp ^
//       -o test_gpu_fit_binary.exe
#include <cstdio>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vector>
#include <random>
#include <chrono>
#include "../src/gpu_fit_binary.cu"
#include "binary_fit.h"
#include "VBMicrolensingLibrary.h"

int main() {
    const double true_log_s = 0.05, true_log_q = -2.3;
    const double true_t0 = 750.0, true_u0 = 0.08, true_tE = 25.0, true_alpha = 1.2, true_rho = 0.01;
    const double true_fs = 1.0, true_fb = 0.2;

    VBMicrolensing vbm;
    vbm.Tol = 1.e-2;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> cadence(0.2, 1.5);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::vector<DataPoint> data;
    double t = 0.0;
    while (t < 1500.0) {
        t += cadence(rng);
        double pr[7] = { true_log_s, true_log_q, true_u0, true_alpha, log(true_rho), log(true_tE), true_t0 };
        double mag = vbm.BinaryLightCurve(pr, t);
        double sigma = 0.02;
        DataPoint dp; dp.t = t; dp.flux = true_fs * mag + true_fb + noise(rng) * sigma; dp.sigma = sigma;
        data.push_back(dp);
    }
    printf("Synthetic light curve: %zu points\n", data.size());

    // PSPL-anchor-style seed: in the real pipeline this comes from pspl_prefit(), here just use
    // values close to the truth (same role: a rough starting point multistart explores around),
    // since validating pspl_prefit() itself is a separate concern from this file's job.
    double t0_anchor = true_t0 + 2.0, u0_anchor = 0.1, tE_anchor = 22.0;

    printf("\n--- CPU reference (fit_binary_multistart) ---\n");
    auto cpu_start = std::chrono::steady_clock::now();
    BinaryFitResult cpu_r = fit_binary_multistart(vbm, data, true_log_s, true_log_q, t0_anchor, u0_anchor, tE_anchor);
    auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - cpu_start).count();
    printf("chi2=%.4f t0=%.4f u0=%.5f tE=%.4f alpha=%.4f rho=%.6f fs=%.4f fb=%.4f  (%lld ms)\n",
        cpu_r.chi2, cpu_r.t0, cpu_r.u0, cpu_r.tE, cpu_r.alpha, cpu_r.rho, cpu_r.fs, cpu_r.fb, (long long)cpu_ms);

    printf("\n--- GPU batched (gpu_fit_binary_multistart) ---\n");
    auto gpu_start = std::chrono::steady_clock::now();
    BinaryFitResult gpu_r = gpu_fit_binary_multistart(vbm, data, true_log_s, true_log_q, t0_anchor, u0_anchor, tE_anchor);
    auto gpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gpu_start).count();
    printf("chi2=%.4f t0=%.4f u0=%.5f tE=%.4f alpha=%.4f rho=%.6f fs=%.4f fb=%.4f  (%lld ms)\n",
        gpu_r.chi2, gpu_r.t0, gpu_r.u0, gpu_r.tE, gpu_r.alpha, gpu_r.rho, gpu_r.fs, gpu_r.fb, (long long)gpu_ms);

    printf("\n--- comparison ---\n");
    double chi2_relerr = fabs(gpu_r.chi2 - cpu_r.chi2) / fabs(cpu_r.chi2);
    printf("chi2 relative difference: %.3e\n", chi2_relerr);
    printf("true params: t0=%.4f u0=%.5f tE=%.4f alpha=%.4f rho=%.6f fs=%.4f fb=%.4f\n",
        true_t0, true_u0, true_tE, true_alpha, true_rho, true_fs, true_fb);
    printf("speedup: %.2fx\n", cpu_ms > 0 ? (double)cpu_ms / (double)std::max((long long)1, (long long)gpu_ms) : 0.0);

    // Pass bar: both should land close to the true injected chi2 (noise floor ~n_points for a
    // good fit), and the GPU result shouldn't be meaningfully WORSE than the CPU one -- it's
    // allowed to find an equally-good or better minimum (different floating-point paths through
    // 96 parallel searches can legitimately land in slightly different places), not a worse one.
    bool pass = (gpu_r.chi2 < cpu_r.chi2 * 1.05) && (gpu_r.chi2 < data.size() * 2.0);
    printf("\n%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
