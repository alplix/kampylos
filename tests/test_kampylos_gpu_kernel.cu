// Validates kampylos_gpu_kernel.cu's batch candidate-evaluation kernel: runs it on a real GPU
// against a synthetic light curve (a real injected binary-lens signal, generated via the actual
// VBMicrolensing CPU reference, plus noise), and checks two independent things:
//  1. Kernel mechanics: for candidates the kernel itself reports as NOT needing fallback, does
//     its chi2/fs/fb match a CPU computation using the exact same kgpu:: functions (same math,
//     different execution path -- this isolates kernel-specific bugs, like reduction or indexing
//     mistakes, from the per-point magnification math already validated in
//     test_kampylos_gpu_mag.cpp).
//  2. needs_fallback correctness: for a sample of candidates, cross-check the kernel's
//     needs_fallback flag against the real VBMicrolensing reference (same method as
//     test_kampylos_gpu_mag.cpp's decision-mismatch check) -- a false "doesn't need fallback"
//     here would be a silent correctness bug reaching all the way to the kernel level, which is
//     exactly the risk this whole two-tier design exists to avoid.
//
// Build (Windows, MSVC host compiler):
//   nvcc -O2 -std=c++17 -gencode arch=compute_120,code=sm_120 -ccbin "<path to cl.exe>" ^
//       -I ../src -I ../vendor/VBMicrolensing/VBMicrolensing/lib ^
//       test_kampylos_gpu_kernel.cu ../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp ^
//       -o test_kampylos_gpu_kernel.exe
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "../src/kampylos_gpu_kernel.cu"
#include "VBMicrolensingLibrary.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

int main() {
    // --- Build a synthetic light curve: real injected signal (via the real VBMicrolensing
    // reference) + noise, over an irregularly-sampled baseline similar to real survey cadence. ---
    // Baseline spans 1500 days -- tE=25 means the actual event occupies a small fraction of it,
    // matching real survey light curves (mostly flat baseline, with the magnification confined
    // to a window a few tE wide around t0). The first version of this test used a 100-day
    // baseline, comparable to tE itself, which put nearly every sampled epoch near the event
    // peak and made the fast-path/fallback split look far more fallback-heavy than real data --
    // worth keeping in mind as a lesson: a synthetic test's sampling design can quietly misstate
    // a real workload's characteristics even when the underlying math is correct.
    const double true_log_s = 0.05, true_log_q = -2.3; // this "grid cell"
    const double true_t0 = 750.0, true_u0 = 0.08, true_tE = 25.0, true_alpha = 1.2, true_rho = 0.01;
    const double true_fs = 1.0, true_fb = 0.2;

    VBMicrolensing vbm_ref;
    vbm_ref.Tol = 1.e-2;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> cadence(0.2, 1.5);
    std::normal_distribution<double> noise(0.0, 1.0);

    std::vector<LightCurvePoint> h_data;
    double t = 0.0;
    while (t < 1500.0) {
        t += cadence(rng);
        double pr[7] = { true_log_s, true_log_q, true_u0, true_alpha, log(true_rho), log(true_tE), true_t0 };
        double mag = vbm_ref.BinaryLightCurve(pr, t);
        double sigma = 0.02; // flux units
        double flux = true_fs * mag + true_fb + noise(rng) * sigma;
        h_data.push_back({ t, flux, sigma });
    }
    int n_points = (int)h_data.size();
    printf("Synthetic light curve: %d points over 100 days\n", n_points);

    // --- Candidates: the true one, plus random ones spanning a range including near-caustic
    // (small rho denominator effects) and far-from-caustic cases, so some SHOULD need fallback
    // and some SHOULDN'T -- a test that only ever hits one branch wouldn't exercise much. ---
    std::vector<Candidate> h_cand;
    h_cand.push_back({ true_t0, true_u0, log(true_tE), true_alpha, log(true_rho) }); // the true one
    std::uniform_real_distribution<double> t0_dist(40, 60), u0_dist(-0.3, 0.3), alpha_dist(0, 6.28);
    std::uniform_real_distribution<double> logtE_dist(log(10.0), log(50.0)), logrho_dist(-4, -2.3);
    for (int i = 0; i < 2000; i++) {
        h_cand.push_back({ t0_dist(rng), u0_dist(rng), logtE_dist(rng), alpha_dist(rng), logrho_dist(rng) });
    }
    int n_cand = (int)h_cand.size();

    // --- Run on GPU ---
    LightCurvePoint* d_data; Candidate* d_cand; CandidateResult* d_res;
    CUDA_CHECK(cudaMalloc(&d_data, n_points * sizeof(LightCurvePoint)));
    CUDA_CHECK(cudaMalloc(&d_cand, n_cand * sizeof(Candidate)));
    CUDA_CHECK(cudaMalloc(&d_res, n_cand * sizeof(CandidateResult)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n_points * sizeof(LightCurvePoint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cand, h_cand.data(), n_cand * sizeof(Candidate), cudaMemcpyHostToDevice));

    int block_size = 128;
    size_t shmem = 7 * block_size * sizeof(double);
    kampylos_eval_candidates<<<n_cand, block_size, shmem>>>(d_data, n_points, true_log_s, true_log_q, d_cand, n_cand, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<CandidateResult> h_res(n_cand);
    CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_cand * sizeof(CandidateResult), cudaMemcpyDeviceToHost));

    // --- Check 1: kernel mechanics (reduction, indexing, launch config) against a host
    // computation using the exact same kgpu:: functions the kernel itself uses -- checked for
    // EVERY candidate regardless of needs_fallback, since that flag is about whether the
    // RESULT should be trusted for science, not about whether the kernel computed its own
    // (GPU-fast-path) formula correctly. A kernel bug should show up here either way. ---
    int mechanics_checked = 0, mechanics_bad = 0;
    double worst_chi2_relerr = 0;
    double s = exp(true_log_s), q = exp(true_log_q);
    for (int ci = 0; ci < n_cand; ci++) {
        if (h_res[ci].chi2 >= 1e17) continue; // out-of-range candidate, nothing to cross-check
        Candidate c = h_cand[ci];
        double tE = exp(c.log_tE), tE_inv = 1.0 / tE;
        double salpha = sin(c.alpha), calpha = cos(c.alpha);
        double S_AA = 0, S_A1 = 0, S_11 = 0, S_Ay = 0, S_1y = 0;
        bool any_invalid = false;
        for (int i = 0; i < n_points; i++) {
            double tn = (h_data[i].t - c.t0) * tE_inv;
            double y1 = c.u0 * salpha - tn * calpha;
            double y2 = -c.u0 * calpha - tn * salpha;
            kgpu::BinaryMag0Result r = kgpu::binary_mag0_gpu(s, q, y1, y2);
            if (!(r.mag > 0) || !isfinite(r.mag)) { any_invalid = true; break; }
            double w = 1.0 / (h_data[i].sigma * h_data[i].sigma);
            S_AA += w * r.mag * r.mag; S_A1 += w * r.mag; S_11 += w;
            S_Ay += w * r.mag * h_data[i].flux; S_1y += w * h_data[i].flux;
        }
        if (any_invalid) continue;
        double det = S_AA * S_11 - S_A1 * S_A1;
        double fs = (S_11 * S_Ay - S_A1 * S_1y) / det;
        double fb = (S_AA * S_1y - S_A1 * S_Ay) / det;
        double chi2 = 0;
        for (int i = 0; i < n_points; i++) {
            double tn = (h_data[i].t - c.t0) * tE_inv;
            double y1 = c.u0 * salpha - tn * calpha;
            double y2 = -c.u0 * calpha - tn * salpha;
            kgpu::BinaryMag0Result r = kgpu::binary_mag0_gpu(s, q, y1, y2);
            double w = 1.0 / (h_data[i].sigma * h_data[i].sigma);
            double diff = h_data[i].flux - (fs * r.mag + fb);
            chi2 += w * diff * diff;
        }
        mechanics_checked++;
        double relerr = fabs(h_res[ci].chi2 - chi2) / fabs(chi2);
        if (relerr > worst_chi2_relerr) worst_chi2_relerr = relerr;
        if (relerr > 1e-8 || fabs(h_res[ci].fs - fs) > 1e-8 * fabs(fs) || fabs(h_res[ci].fb - fb) > 1e-8 * fmax(1.0, fabs(fb))) {
            mechanics_bad++;
            if (mechanics_bad <= 5) {
                printf("MECHANICS MISMATCH cand#%d: gpu_chi2=%.10f cpu_chi2=%.10f relerr=%.3e gpu_fs=%.6f cpu_fs=%.6f gpu_fb=%.6f cpu_fb=%.6f\n",
                    ci, h_res[ci].chi2, chi2, relerr, h_res[ci].fs, fs, h_res[ci].fb, fb);
            }
        }
    }

    // --- Check 2: needs_fallback correctness against the real VBMicrolensing reference, same
    // method as test_kampylos_gpu_mag.cpp (compare BinaryMag2 against BinaryMag0 per point). ---
    int fallback_checked = 0, fallback_false_negatives = 0;
    for (int ci = 0; ci < n_cand; ci += 20) { // sample every 20th candidate -- this check is O(n_points) per candidate on the CPU, no need to run it on all 2001
        if (h_res[ci].chi2 >= 1e17) continue;
        Candidate c = h_cand[ci];
        double tE = exp(c.log_tE), tE_inv = 1.0 / tE, rho = exp(c.log_rho);
        double salpha = sin(c.alpha), calpha = cos(c.alpha);
        bool ref_any_needs_fallback = false;
        for (int i = 0; i < n_points; i++) {
            double tn = (h_data[i].t - c.t0) * tE_inv;
            double y1 = c.u0 * salpha - tn * calpha;
            double y2 = -c.u0 * calpha - tn * salpha;
            double mag0 = vbm_ref.BinaryMag0(s, q, y1, y2);
            if (mag0 <= 0) continue;
            double mag2 = vbm_ref.BinaryMag2(s, q, y1, y2, rho);
            if (fabs(mag2 - mag0) / mag0 > 1e-4) { ref_any_needs_fallback = true; break; }
        }
        fallback_checked++;
        if (ref_any_needs_fallback && !h_res[ci].needs_fallback) {
            fallback_false_negatives++;
            if (fallback_false_negatives <= 5) {
                printf("FALLBACK FALSE NEGATIVE cand#%d: reference says this light curve needed BinaryMagDark somewhere, kernel didn't flag it\n", ci);
            }
        }
    }

    printf("\n=== kernel test: %d points, %d candidates ===\n", n_points, n_cand);
    printf("mechanics check: %d candidates cross-checked, %d mismatches, worst chi2 relerr %.3e\n",
        mechanics_checked, mechanics_bad, worst_chi2_relerr);
    printf("fallback check: %d candidates sampled, %d FALSE NEGATIVES (unsafe)\n", fallback_checked, fallback_false_negatives);
    int n_flagged = 0;
    for (int ci = 0; ci < n_cand; ci++) if (h_res[ci].needs_fallback) n_flagged++;
    printf("kernel flagged needs_fallback on %d / %d candidates\n", n_flagged, n_cand);
    printf("true-parameter candidate: chi2=%.4f fs=%.4f fb=%.4f needs_fallback=%d (injected fs=%.2f fb=%.2f)\n",
        h_res[0].chi2, h_res[0].fs, h_res[0].fb, h_res[0].needs_fallback, true_fs, true_fb);

    bool pass = (mechanics_bad == 0) && (fallback_false_negatives == 0);
    printf("\n%s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_data); cudaFree(d_cand); cudaFree(d_res);
    return pass ? 0 : 1;
}
