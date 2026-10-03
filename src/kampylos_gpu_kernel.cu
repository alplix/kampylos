// Batch candidate-evaluation CUDA kernel: given one light curve (fixed s, q for this grid cell)
// and a batch of candidate (t0, u0, tE, alpha, rho) parameter vectors, computes each candidate's
// chi2 the same way binary_fit.h's fit_binary_local() objective() does on the CPU -- one thread
// BLOCK per candidate, threads within a block splitting the light curve's data points among
// themselves (this is where the real parallelism is: a light curve can have anywhere from
// hundreds to tens of thousands of points, see kampylos's three survey sources).
//
// Correctness strategy (the part that matters most here -- see kampylos_gpu_mag.cuh's own top
// comment for why): this kernel ONLY ever uses the point-source + quadrupole/hexadecapole fast
// path (binary_mag0_gpu / binary_mag2_fastpath_ok). It never attempts finite-source contour
// integration (BinaryMagDark) itself -- that stays on the CPU, in the real, already-correct
// VBMicrolensing library, for just the handful of points that actually need it.
//
// Revision 2 (2026-10-03) -- per-point fallback, not per-candidate: the first version flagged a
// whole candidate "needs_fallback" the moment ANY one of its points failed the fast-path test,
// discarding that candidate's entire chi2 for a full O(n_points) CPU recompute. Tested against a
// realistic light curve (long baseline, tens of near-peak points), that turned out to hit nearly
// every candidate near the true parameters -- the "rare exception" the original design assumed
// wasn't rare at all, which would have defeated most of the point of running this on a GPU.
//
// The fix follows from one identity: a weighted linear fit's chi2 at its own best-fit (fs, fb)
// reduces to chi2 = S_yy - fs*S_Ay - fb*S_1y (S_yy = sum w*flux^2; substitute the normal
// equations into the full quadratic expansion of sum w*(flux - fs*A - fb)^2 to see why) -- i.e.
// chi2 and (fs, fb) are BOTH fully determined by six running sums (S_AA, S_A1, S_11, S_Ay, S_1y,
// S_yy), with no need for lightcurve.h's reference implementation's second pass over the data.
// Those six sums are also trivially SEPARABLE across points: a point's contribution doesn't
// depend on any other point's. So this kernel accumulates them from the fast-path-OK points
// only, and separately reports which point INDICES failed the test (not just a yes/no) -- the
// caller adds each bad point's real (BinaryMagDark-computed) contribution to the six sums itself,
// then gets the correct fs/fb/chi2 for the whole light curve from the same closed form. The GPU
// still does the expensive O(n_points) pass over every candidate; the CPU only ever touches the
// (usually small) per-candidate list of bad points, not the full light curve -- which holds even
// when most candidates have at least one bad point, unlike revision 1's design.
#include "kampylos_gpu_mag.cuh"
#include "kampylos_gpu_types.h"
#include <cfloat>

// Tree reduction of running sums across a block, via shared memory -- simple and correct over
// clever (warp-shuffle combining many independent reductions isn't meaningfully faster here; the
// magnification math per point dominates cost, not this). NSUMS = 6 statistics + 1 "poisoned" OR
// flag = 7 lanes.
#define KAMPYLOS_NREDUCE 7

__device__ void block_reduce(double* sh) {
    int tid = threadIdx.x;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int k = 0; k < KAMPYLOS_NREDUCE; k++) {
                sh[k * blockDim.x + tid] += sh[k * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void kampylos_eval_candidates(
    const LightCurvePoint* data, int n_points,
    double log_s, double log_q,
    const Candidate* candidates, int n_candidates,
    CandidateResult* results
) {
    int cand_idx = blockIdx.x;
    if (cand_idx >= n_candidates) return;

    extern __shared__ double sh[]; // KAMPYLOS_NREDUCE * blockDim.x doubles
    // Per-block bad-point collection: a single shared counter, written via atomicAdd so threads
    // across the block can append concurrently without clobbering each other's slot.
    __shared__ int bad_count;
    __shared__ int overflow_flag;
    if (threadIdx.x == 0) { bad_count = 0; overflow_flag = 0; }
    __syncthreads();

    Candidate c = candidates[cand_idx];
    double tE = exp(c.log_tE);
    double rho = exp(c.log_rho);

    // Same guard as binary_fit.h's objective() -- out-of-range tE/rho short-circuits without
    // evaluating any magnification. Uniform across the whole block (same candidate), so no warp
    // divergence from this branch.
    if (tE < 0.01 || tE > 10000.0 || rho < 1e-6 || rho > 1.0) {
        if (threadIdx.x == 0) {
            CandidateResult* r = &results[cand_idx];
            r->S_AA = r->S_A1 = r->S_11 = r->S_Ay = r->S_1y = r->S_yy = 0;
            r->n_bad = 0; r->overflow = 0; r->poisoned = 1; // reuse "poisoned" to signal "dead candidate, chi2=1e18" uniformly to the caller
        }
        return;
    }

    double s = exp(log_s), q = exp(log_q);
    double salpha = sin(c.alpha), calpha = cos(c.alpha);
    double tE_inv = 1.0 / tE;

    double S_AA = 0, S_A1 = 0, S_11 = 0, S_Ay = 0, S_1y = 0, S_yy = 0;
    double any_invalid = 0;

    for (int i = threadIdx.x; i < n_points; i += blockDim.x) {
        double t = data[i].t;
        double tn = (t - c.t0) * tE_inv;
        double y1 = c.u0 * salpha - tn * calpha;
        double y2 = -c.u0 * calpha - tn * salpha;

        kgpu::BinaryMag0Result r = kgpu::binary_mag0_gpu(s, q, y1, y2);
        double A = r.mag;
        if (!(A > 0) || !isfinite(A)) {
            any_invalid = 1.0;
            continue; // whole candidate is dead regardless of n_bad/overflow -- see "poisoned" above
        }
        if (!kgpu::binary_mag2_fastpath_ok(r, rho)) {
            int slot = atomicAdd(&bad_count, 1);
            if (slot < KAMPYLOS_MAX_BAD_POINTS) {
                results[cand_idx].bad_idx[slot] = i;
            } else {
                overflow_flag = 1;
            }
            continue; // excluded from this block's sums -- caller adds its real contribution
        }

        double sigma = data[i].sigma;
        double w = 1.0 / (sigma * sigma);
        double flux = data[i].flux;
        S_AA += w * A * A;
        S_A1 += w * A;
        S_11 += w;
        S_Ay += w * A * flux;
        S_1y += w * flux;
        S_yy += w * flux * flux;
    }

    int tid = threadIdx.x;
    sh[0 * blockDim.x + tid] = S_AA;
    sh[1 * blockDim.x + tid] = S_A1;
    sh[2 * blockDim.x + tid] = S_11;
    sh[3 * blockDim.x + tid] = S_Ay;
    sh[4 * blockDim.x + tid] = S_1y;
    sh[5 * blockDim.x + tid] = S_yy;
    sh[6 * blockDim.x + tid] = any_invalid;
    __syncthreads();
    block_reduce(sh);

    if (tid == 0) {
        CandidateResult* r = &results[cand_idx];
        if (sh[6 * blockDim.x] > 0) {
            r->S_AA = r->S_A1 = r->S_11 = r->S_Ay = r->S_1y = r->S_yy = 0;
            r->n_bad = 0; r->overflow = 0; r->poisoned = 1;
        } else {
            r->S_AA = sh[0 * blockDim.x]; r->S_A1 = sh[1 * blockDim.x]; r->S_11 = sh[2 * blockDim.x];
            r->S_Ay = sh[3 * blockDim.x]; r->S_1y = sh[4 * blockDim.x]; r->S_yy = sh[5 * blockDim.x];
            r->n_bad = min(bad_count, KAMPYLOS_MAX_BAD_POINTS);
            r->overflow = overflow_flag;
            r->poisoned = 0;
        }
    }
}
