// GPU-accelerated replacement for binary_fit.h's fit_binary_seeds(): instead of running every
// seed search one at a time on the CPU, drives all of them with
// NelderMeadStepper, batching every active stepper's pending evaluation points into a single
// kampylos_eval_candidates kernel launch per round, completing each candidate's fit via
// kampylos_gpu_complete.h (which patches in the few points needing real VBMicrolensing-exact
// treatment). Same seeding pattern, same per-search max_iter/ftol/xtol and n_restarts as
// fit_binary_local/fit_binary_multistart -- this is meant to reach the same answer those do,
// just by evaluating many candidates per step instead of one.
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include "kampylos_gpu_mag.cuh"
#include "kampylos_gpu_types.h"
#include "kampylos_gpu_kernel.cu"
#include "kampylos_gpu_complete.h"
#include "kampylos_cpu_patch_pool.h"
#include "nelder_mead_stepper.h"
#include "kampylos_batched_fit.h"
#include "gpu_fit_binary.h" // GpuFitContext, BinaryFitResult, function declarations

#define CUDA_CHECK_RT(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "gpu_fit_binary CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

// One batched evaluation round: uploads `points` as candidates (all at the fixed linear s, q of
// this cell), launches the kernel, completes each result on the CPU (patching in any bad points),
// returns one chi2 per input point, same order.
static std::vector<double> gpu_eval_batch(
    GpuFitContext& ctx, double s, double q,
    const std::vector<Candidate>& points
) {
    int n = (int)points.size();
    std::vector<double> chi2(n);
    if (n == 0) return chi2;

    ctx.ensure_cand_capacity(n);
    CUDA_CHECK_RT(cudaMemcpy(ctx.d_cand, points.data(), n * sizeof(Candidate), cudaMemcpyHostToDevice));

    int block_size = 128;
    size_t shmem = KAMPYLOS_NREDUCE * block_size * sizeof(double);
    kampylos_eval_candidates<<<n, block_size, shmem>>>(ctx.d_data, ctx.n_points, s, q, ctx.d_cand, n, ctx.d_res);
    CUDA_CHECK_RT(cudaGetLastError());

    std::vector<CandidateResult> h_res(n);
    CUDA_CHECK_RT(cudaMemcpy(h_res.data(), ctx.d_res, n * sizeof(CandidateResult), cudaMemcpyDeviceToHost));

    // Was a single-threaded loop calling kampylos_complete_candidate() once per candidate here,
    // every round of the batched Nelder-Mead (up to ~1200 rounds/cell) -- fine when each
    // candidate needs 0-1 points patched, but for a pathological anchor (tE far beyond the
    // data's own span) a candidate can need many points patched, and this loop dominated
    // wall-clock: one real-hardware test cell took 91+ minutes, almost entirely here, defeating
    // the GPU kernel's own speed advantage.
    //
    // First attempt at this (2026-10-04, earlier the same night) gave each worker thread its own
    // VBMicrolensing instance and crashed immediately on real hardware (heap corruption) --
    // VBMicrolensing's magnification functions turned out to be full of function-local `static`
    // scratch variables (not reentrant even across separate instances). Root-caused and fixed AT
    // THE SOURCE instead of working around it here: every such static in vendor/kampylos/vendor/
    // VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.{cpp,h} is now `thread_local
    // static` (243 of them). Verified real-hardware-safe (see kampylos_cpu_patch_pool.h's own
    // history note) before this file was changed back to use it.
    // Created once, never destroyed: see kampylos_cpu_patch_pool.h (persistent workers).
    static KampylosCpuPatchPool& patch_pool = *new KampylosCpuPatchPool(4);
    patch_pool.complete_batch(h_res, points, ctx.h_data.data(), s, q, chi2, ctx.fs_bound, ctx.flux_offset);
    return chi2;
}

// Converts the DataPoint vector (Kampylos's own lightcurve.h type) to this file's
// LightCurvePoint (kampylos_gpu_types.h), flux-centred (see kampylos_flux_offset()), and uploads
// it -- call once per WU (the light curve is the same across every grid cell a WU processes),
// not once per cell, since it can be thousands of points.
void gpu_upload_light_curve(GpuFitContext& ctx, const std::vector<DataPoint>& data) {
    ctx.flux_offset = kampylos_flux_offset(data);
    ctx.fs_bound = flux_fit_bound(data);
    ctx.h_data.resize(data.size());
    for (size_t i = 0; i < data.size(); i++) ctx.h_data[i] = { data[i].t, data[i].flux - ctx.flux_offset, data[i].sigma };
    ctx.upload_light_curve(ctx.h_data);
}

BinaryFitResult gpu_fit_binary_seeds(
    GpuFitContext& ctx,
    const std::vector<DataPoint>& data,
    double ln_s, double ln_q,
    const BinarySeedSet& seeds,
    int n_restarts
) {
    (void)data;
    double s = exp(ln_s), q = exp(ln_q);
    auto eval = [&](const std::vector<Candidate>& batch) { return gpu_eval_batch(ctx, s, q, batch); };
    return kampylos_batched_fit_seeds(eval, ln_s, ln_q, seeds, n_restarts);
}

// Legacy entry point kept for tests/test_gpu_fit_binary.cu: same 96-seed grid as
// binary_fit.h's fit_binary_multistart(), batched; fs/fb/chi2 recomputed exactly on the host.
BinaryFitResult gpu_fit_binary_multistart(
    GpuFitContext& ctx,
    VBMicrolensing& vbm,
    const std::vector<DataPoint>& data,
    double log_s, double log_q,
    double t0_anchor, double u0_anchor, double tE_anchor,
    double rho_seed,
    int n_alpha_seeds,
    int n_restarts
) {
    if (ctx.n_points != (int)data.size()) gpu_upload_light_curve(ctx, data);
    BinarySeedSet seeds = kampylos_legacy_multistart_seeds(t0_anchor, u0_anchor, tE_anchor, rho_seed, n_alpha_seeds);
    BinaryFitResult out = gpu_fit_binary_seeds(ctx, data, log_s, log_q, seeds, n_restarts);
    std::vector<double> mag(data.size());
    double pr[7] = { log_s, log_q, out.u0, out.alpha, log(out.rho), log(out.tE), out.t0 };
    for (size_t i = 0; i < data.size(); i++) mag[i] = vbm.BinaryLightCurve(pr, data[i].t);
    FluxFit ff = linear_flux_fit(data, mag);
    out.fs = ff.fs; out.fb = ff.fb;
    return out;
}
