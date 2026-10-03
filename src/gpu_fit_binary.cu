// GPU-accelerated replacement for binary_fit.h's fit_binary_multistart(): instead of running 96
// independent seed searches one at a time on the CPU, drives all of them with
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
#include "nelder_mead_stepper.h"
#include "gpu_fit_binary.h" // GpuFitContext, BinaryFitResult, function declarations

#define CUDA_CHECK_RT(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "gpu_fit_binary CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

// One batched evaluation round: uploads `points` as candidates (all at the fixed log_s/log_q for
// this cell), launches the kernel, completes each result on the CPU (patching in any bad points),
// returns one chi2 per input point, same order.
static std::vector<double> gpu_eval_batch(
    GpuFitContext& ctx, VBMicrolensing& vbm,
    double log_s, double log_q, double s, double q,
    const std::vector<LightCurvePoint>& h_data,
    const std::vector<Candidate>& points
) {
    int n = (int)points.size();
    std::vector<double> chi2(n);
    if (n == 0) return chi2;

    ctx.ensure_cand_capacity(n);
    CUDA_CHECK_RT(cudaMemcpy(ctx.d_cand, points.data(), n * sizeof(Candidate), cudaMemcpyHostToDevice));

    int block_size = 128;
    size_t shmem = KAMPYLOS_NREDUCE * block_size * sizeof(double);
    kampylos_eval_candidates<<<n, block_size, shmem>>>(ctx.d_data, ctx.n_points, log_s, log_q, ctx.d_cand, n, ctx.d_res);
    CUDA_CHECK_RT(cudaGetLastError());

    std::vector<CandidateResult> h_res(n);
    CUDA_CHECK_RT(cudaMemcpy(h_res.data(), ctx.d_res, n * sizeof(CandidateResult), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        CompletedFit cf = kampylos_complete_candidate(vbm, h_res[i], points[i], h_data.data(), s, q);
        chi2[i] = cf.chi2;
    }
    return chi2;
}

// Converts a stepper's pending 5-vector (t0, u0, log_tE, alpha, log_rho) into a Candidate, and
// runs the same tE/rho range guard fit_binary_local's objective() applies -- the kernel itself
// also guards this (see kampylos_eval_candidates), so this isn't load-bearing for correctness,
// just avoids uploading obviously-out-of-range candidates in the first place.
static Candidate to_candidate(const std::vector<double>& x) {
    Candidate c;
    c.t0 = x[0]; c.u0 = x[1]; c.log_tE = x[2]; c.alpha = x[3]; c.log_rho = x[4];
    return c;
}

// Drives N independent NelderMeadStepper instances to completion, batching every round's pending
// points across all of them into one gpu_eval_batch() call. Returns each stepper's final result,
// same order as the input x0/step vectors.
static std::vector<SimplexResultLite> gpu_batched_nelder_mead(
    GpuFitContext& ctx, VBMicrolensing& vbm,
    double log_s, double log_q, double s, double q,
    const std::vector<LightCurvePoint>& h_data,
    const std::vector<std::vector<double>>& x0s,
    const std::vector<std::vector<double>>& steps,
    int max_iter, double ftol, double xtol
) {
    int n_seeds = (int)x0s.size();
    std::vector<NelderMeadStepper> steppers;
    steppers.reserve(n_seeds);
    for (int i = 0; i < n_seeds; i++) steppers.emplace_back(x0s[i], steps[i], max_iter, ftol, xtol);

    for (;;) {
        // Gather every active stepper's pending points into one flat batch, remembering which
        // (stepper, local-index) each slot came from so results can be routed back.
        std::vector<Candidate> batch;
        std::vector<std::pair<int,int>> owner; // (stepper index, count-within-that-stepper's-pending)
        std::vector<int> pending_count(n_seeds, 0);
        bool any_active = false;
        for (int s_i = 0; s_i < n_seeds; s_i++) {
            if (steppers[s_i].done()) continue;
            any_active = true;
            const auto& pts = steppers[s_i].pending_points();
            pending_count[s_i] = (int)pts.size();
            for (const auto& p : pts) {
                batch.push_back(to_candidate(p));
                owner.push_back({ s_i, 0 });
            }
        }
        if (!any_active) break;

        std::vector<double> chi2 = gpu_eval_batch(ctx, vbm, log_s, log_q, s, q, h_data, batch);

        // Route results back to each stepper in the same order its pending_points() listed them.
        int offset = 0;
        for (int s_i = 0; s_i < n_seeds; s_i++) {
            if (pending_count[s_i] == 0) continue;
            std::vector<double> sub(chi2.begin() + offset, chi2.begin() + offset + pending_count[s_i]);
            offset += pending_count[s_i];
            steppers[s_i].submit_results(sub);
        }
    }

    std::vector<SimplexResultLite> out;
    out.reserve(n_seeds);
    for (auto& st : steppers) out.push_back(st.result());
    return out;
}

// Converts the DataPoint vector (Kampylos's own lightcurve.h type) to this file's
// LightCurvePoint (kampylos_gpu_types.h) and uploads it -- call once per WU (the light curve is
// the same across every grid cell a WU processes), not once per cell, since it can be thousands
// of points and re-uploading it n_q times per WU for no reason is pure waste.
void gpu_upload_light_curve(GpuFitContext& ctx, const std::vector<DataPoint>& data) {
    std::vector<LightCurvePoint> h_data(data.size());
    for (size_t i = 0; i < data.size(); i++) h_data[i] = { data[i].t, data[i].flux, data[i].sigma };
    ctx.upload_light_curve(h_data);
}

// GPU-accelerated equivalent of binary_fit.h's fit_binary_multistart(): same seed grid (8 alpha x
// 2 u0 magnitudes x 2 signs x 3 rho seeds = 96 searches), same restart-from-converged refinement
// pass (n_restarts default 2) -- every one of those 96 (and then their restarts) runs as one
// batched GPU pass instead of 96 sequential CPU calls.
//
// Takes an already-initialized GpuFitContext (device already selected via cudaSetDevice, light
// curve already uploaded via gpu_upload_light_curve) rather than owning one itself -- a WU calls
// this once per grid cell (n_q times), and the context's device buffers/light curve upload should
// persist across those calls, not be torn down and rebuilt every cell.
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
    std::vector<LightCurvePoint> h_data(data.size());
    for (size_t i = 0; i < data.size(); i++) h_data[i] = { data[i].t, data[i].flux, data[i].sigma };
    double s = exp(log_s), q = exp(log_q);

    const double rho_seeds[] = { rho_seed, 1e-4, 1e-2 };
    const double u0_mag_seeds[] = { fabs(u0_anchor), 0.02 };

    std::vector<std::vector<double>> x0s, steps;
    for (int i = 0; i < n_alpha_seeds; i++) {
        double alpha_seed = 2.0 * M_PI * i / n_alpha_seeds;
        for (double u0_mag : u0_mag_seeds) {
            for (double u0_sign : { 1.0, -1.0 }) {
                for (double rs : rho_seeds) {
                    x0s.push_back({ t0_anchor, u0_sign * u0_mag, log(tE_anchor), alpha_seed, log(rs) });
                    steps.push_back({ std::max(0.5, tE_anchor * 0.05), 0.05, 0.2, 0.3, 0.5 });
                }
            }
        }
    }

    auto pass1 = gpu_batched_nelder_mead(ctx, vbm, log_s, log_q, s, q, h_data, x0s, steps, 400, 1e-9, 1e-8);

    // Restart passes: same "restart once converged, with smaller steps" logic as
    // fit_binary_local, run on every seed's pass1 result in parallel (not just the overall best --
    // matches the CPU version restarting EVERY seed, since a seed that only looked good after
    // pass1 might still improve, and one that looked bad might not be worth restarting, but the
    // CPU code doesn't discriminate either).
    std::vector<SimplexResultLite> current = pass1;
    for (int restart = 0; restart < n_restarts; restart++) {
        std::vector<std::vector<double>> rx0, rsteps;
        for (auto& r : current) {
            rx0.push_back(r.x);
            rsteps.push_back({ 0.3, 0.02, 0.08, 0.15, 0.2 });
        }
        auto next = gpu_batched_nelder_mead(ctx, vbm, log_s, log_q, s, q, h_data, rx0, rsteps, 400, 1e-10, 1e-9);
        for (size_t i = 0; i < current.size(); i++) {
            if (next[i].fval < current[i].fval) current[i] = next[i];
        }
    }

    int best = 0;
    for (size_t i = 1; i < current.size(); i++) if (current[i].fval < current[best].fval) best = (int)i;

    BinaryFitResult out;
    out.log_s = log_s; out.log_q = log_q;
    out.t0 = current[best].x[0]; out.u0 = current[best].x[1]; out.tE = exp(current[best].x[2]);
    out.alpha = current[best].x[3]; out.rho = exp(current[best].x[4]);
    out.chi2 = current[best].fval;

    // fs/fb for the winning point, via the real CPU library over every point -- same as
    // fit_binary_local's own final fs/fb computation, cheap next to everything above (one pass,
    // once, not per-seed).
    std::vector<double> mag(data.size());
    double pr[7] = { log_s, log_q, out.u0, out.alpha, log(out.rho), log(out.tE), out.t0 };
    for (size_t i = 0; i < data.size(); i++) mag[i] = vbm.BinaryLightCurve(pr, data[i].t);
    FluxFit ff = linear_flux_fit(data, mag);
    out.fs = ff.fs; out.fb = ff.fb;
    return out;
}
