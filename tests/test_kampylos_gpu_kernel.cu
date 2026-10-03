// Validates the revision-2 hybrid pipeline (kampylos_gpu_kernel.cu + kampylos_gpu_complete.h) end
// to end: runs the kernel on a real GPU against a synthetic light curve (a real injected
// binary-lens signal, generated via the actual VBMicrolensing CPU reference, plus noise), then
// completes each candidate's fit via kampylos_complete_candidate(), and checks three things:
//  1. Correctness: does the hybrid (GPU fast-path + CPU patch for the few bad points) chi2/fs/fb
//     match a FULL reference computation that uses the real VBMicrolensing BinaryMag2 for EVERY
//     point (not just the ones the kernel flagged bad)? This is the real end-to-end correctness
//     claim -- the whole point of this design is that the hybrid result should be
//     indistinguishable from "just use the real library for everything", not an approximation.
//  2. n_bad stays small: confirms the actual motivation for revision 2 -- even when most
//     candidates have at least one bad point (as revision 1's test found), the CPU only ever
//     touches a handful of points per candidate, not the whole light curve.
//  3. No overflow/poisoned surprises on well-posed candidates.
//
// Build (Windows, MSVC host compiler):
//   nvcc -O2 -std=c++17 -gencode arch=compute_120,code=sm_120 ^
//       -I ../src -I ../vendor/VBMicrolensing/VBMicrolensing/lib ^
//       test_kampylos_gpu_kernel.cu ../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp ^
//       -o test_kampylos_gpu_kernel.exe
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "../src/kampylos_gpu_kernel.cu"
#include "../src/kampylos_gpu_complete.h"
#include "VBMicrolensingLibrary.h"

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

int main() {
    // Baseline spans 1500 days -- tE=25 means the actual event occupies a small fraction of it,
    // matching real survey light curves (mostly flat baseline, with the magnification confined
    // to a window a few tE wide around t0).
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
        double sigma = 0.02;
        double flux = true_fs * mag + true_fb + noise(rng) * sigma;
        h_data.push_back({ t, flux, sigma });
    }
    int n_points = (int)h_data.size();
    printf("Synthetic light curve: %d points over 1500 days\n", n_points);

    std::vector<Candidate> h_cand;
    h_cand.push_back({ true_t0, true_u0, log(true_tE), true_alpha, log(true_rho) }); // the true one
    std::uniform_real_distribution<double> t0_dist(740, 760), u0_dist(-0.3, 0.3), alpha_dist(0, 6.28);
    std::uniform_real_distribution<double> logtE_dist(log(10.0), log(50.0)), logrho_dist(-4, -2.3);
    for (int i = 0; i < 2000; i++) {
        h_cand.push_back({ t0_dist(rng), u0_dist(rng), logtE_dist(rng), alpha_dist(rng), logrho_dist(rng) });
    }
    int n_cand = (int)h_cand.size();

    LightCurvePoint* d_data; Candidate* d_cand; CandidateResult* d_res;
    CUDA_CHECK(cudaMalloc(&d_data, n_points * sizeof(LightCurvePoint)));
    CUDA_CHECK(cudaMalloc(&d_cand, n_cand * sizeof(Candidate)));
    CUDA_CHECK(cudaMalloc(&d_res, n_cand * sizeof(CandidateResult)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n_points * sizeof(LightCurvePoint), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cand, h_cand.data(), n_cand * sizeof(Candidate), cudaMemcpyHostToDevice));

    int block_size = 128;
    size_t shmem = KAMPYLOS_NREDUCE * block_size * sizeof(double);
    kampylos_eval_candidates<<<n_cand, block_size, shmem>>>(d_data, n_points, true_log_s, true_log_q, d_cand, n_cand, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<CandidateResult> h_res(n_cand);
    CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_cand * sizeof(CandidateResult), cudaMemcpyDeviceToHost));

    double s = exp(true_log_s), q = exp(true_log_q);
    int n_dead = 0, n_overflow = 0, n_with_bad = 0;
    long long total_bad = 0;
    int max_bad = 0;
    int mismatches = 0;
    double worst_relerr = 0;

    for (int ci = 0; ci < n_cand; ci++) {
        CompletedFit cf = kampylos_complete_candidate(vbm_ref, h_res[ci], h_cand[ci], h_data.data(), s, q);
        if (h_res[ci].overflow) n_overflow++;
        if (cf.dead) { n_dead++; continue; }
        if (h_res[ci].n_bad > 0) n_with_bad++;
        total_bad += h_res[ci].n_bad;
        if (h_res[ci].n_bad > max_bad) max_bad = h_res[ci].n_bad;

        // Full reference: real VBMicrolensing for EVERY point, not just the kernel's bad list --
        // this is the actual correctness bar (does the hybrid match "do it the slow correct way
        // for the whole light curve"), independent of which points the kernel happened to flag.
        Candidate c = h_cand[ci];
        double tE = exp(c.log_tE), tE_inv = 1.0 / tE, rho = exp(c.log_rho);
        double salpha = sin(c.alpha), calpha = cos(c.alpha);
        double S_AA = 0, S_A1 = 0, S_11 = 0, S_Ay = 0, S_1y = 0, S_yy = 0;
        for (int i = 0; i < n_points; i++) {
            double tn = (h_data[i].t - c.t0) * tE_inv;
            double y1 = c.u0 * salpha - tn * calpha;
            double y2 = -c.u0 * calpha - tn * salpha;
            double A = vbm_ref.BinaryMag2(s, q, y1, y2, rho);
            double w = 1.0 / (h_data[i].sigma * h_data[i].sigma);
            double flux = h_data[i].flux;
            S_AA += w * A * A; S_A1 += w * A; S_11 += w; S_Ay += w * A * flux; S_1y += w * flux; S_yy += w * flux * flux;
        }
        double det = S_AA * S_11 - S_A1 * S_A1;
        double ref_fs = (S_11 * S_Ay - S_A1 * S_1y) / det;
        double ref_fb = (S_AA * S_1y - S_A1 * S_Ay) / det;
        double ref_chi2 = S_yy - ref_fs * S_Ay - ref_fb * S_1y;

        double relerr = fabs(cf.chi2 - ref_chi2) / fabs(ref_chi2);
        if (relerr > worst_relerr) worst_relerr = relerr;
        if (relerr > 1e-6) {
            mismatches++;
            if (mismatches <= 8) {
                printf("MISMATCH cand#%d (n_bad=%d): hybrid_chi2=%.8f ref_chi2=%.8f relerr=%.3e hybrid_fs=%.4f ref_fs=%.4f hybrid_fb=%.4f ref_fb=%.4f\n",
                    ci, h_res[ci].n_bad, cf.chi2, ref_chi2, relerr, cf.fs, ref_fs, cf.fb, ref_fb);
            }
        }
    }

    printf("\n=== hybrid pipeline test: %d points, %d candidates ===\n", n_points, n_cand);
    printf("dead (poisoned/overflow): %d, overflow specifically: %d\n", n_dead, n_overflow);
    printf("candidates with >=1 bad point: %d / %d, total bad-point evaluations: %lld, max n_bad for one candidate: %d\n",
        n_with_bad, n_cand - n_dead, total_bad, max_bad);
    printf("chi2 mismatches (relerr > 1e-6): %d, worst relerr %.3e\n", mismatches, worst_relerr);

    bool pass = (mismatches == 0) && (n_overflow == 0);
    printf("\n%s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_data); cudaFree(d_cand); cudaFree(d_res);
    return pass ? 0 : 1;
}
