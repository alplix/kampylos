// Batch candidate-evaluation CUDA kernel: given one light curve (fixed s, q for this grid cell)
// and a batch of candidate (t0, u0, tE, alpha, rho) parameter vectors, computes each candidate's
// chi2 the same way binary_fit.h's fit_binary_local() objective() does on the CPU -- one thread
// BLOCK per candidate, threads within a block splitting the light curve's data points among
// themselves (this is where the real parallelism is: a light curve can have anywhere from
// hundreds to tens of thousands of points, see kampylos's three survey sources).
//
// Correctness strategy (the part that matters most here -- see kampylos_gpu_mag.cuh's own
// top comment for why): this kernel ONLY ever uses the point-source + quadrupole/hexadecapole
// fast path (binary_mag0_gpu / binary_mag2_fastpath_ok). It does NOT attempt finite-source
// contour integration (BinaryMagDark) on the GPU -- that stays on the CPU, in the real,
// already-correct VBMicrolensing library. Instead, this kernel also reports, per candidate,
// whether ANY of its data points failed the fast-path accept/reject test (output.needs_fallback).
// The caller's job (not this file's) is: for any candidate with needs_fallback set, discard this
// kernel's chi2 for that one candidate and recompute it on the CPU via the real
// fit_binary_local()-style objective function instead -- the GPU's result for that candidate was
// never meant to be trusted, only its SIGNAL that CPU-exact evaluation is needed here. This
// keeps every number that's actually used either genuinely GPU-fast-and-correct, or genuinely
// CPU-exact -- never a silent approximation presented as a real result, which is what would make
// this scientifically unusable.
//
// KNOWN LIMITATION (found via tests/test_kampylos_gpu_kernel.cu against a realistic synthetic
// light curve, 2026-10-03): needs_fallback is a single bit per candidate -- ANY failing point
// discards the WHOLE candidate's chi2 for CPU recomputation over ALL of its points. Tested
// against a light curve with a real event (tens of near-peak points out of ~1700 total), nearly
// every candidate near the true parameters had at least one near-peak point fail the fast-path
// test, so nearly every candidate fell back to a full CPU recompute -- which defeats most of the
// GPU speedup this kernel exists for, since the expensive part (an O(n_points) CPU loop) ends up
// running for most candidates anyway, not as the rare exception the original design assumed.
// Correct, but not yet the performant version. The real fix: have this kernel emit a compact
// per-candidate LIST of which point indices failed (not just a bool), so the caller's CPU
// fallback only recomputes those specific few points via BinaryMagDark and adds their
// contribution to the chi2 this kernel already computed from the rest -- not every point. Not
// yet implemented; next step on this file.
#include "kampylos_gpu_mag.cuh"
#include <cfloat>

struct LightCurvePoint {
    double t;
    double flux;
    double sigma;
};

// Matches fit_binary_local's x[] order (t0, u0, log_tE, alpha, log_rho) -- s and q are fixed for
// the whole grid cell (passed separately, not per-candidate) since every candidate in one kernel
// launch belongs to the same (log_s, log_q) cell.
struct Candidate {
    double t0;
    double u0;
    double log_tE;
    double alpha;
    double log_rho;
};

struct CandidateResult {
    double chi2;
    double fs;
    double fb;
    int needs_fallback;   // 1 if any data point needed BinaryMagDark -- caller must redo this
                            // candidate's chi2 on the CPU rather than trust this struct's chi2/fs/fb
};

// Tree reduction of 7 running sums (S_AA, S_A1, S_11, S_Ay, S_1y, any_invalid, needs_fallback)
// across a block, via shared memory. One shared double array of size 7*blockDim.x -- simple and
// correct over clever (warp-shuffle combining across 7 independent reductions isn't meaningfully
// faster here; the magnification math per point dominates cost, not this).
__device__ void block_reduce7(double* sh, double* out7) {
    int tid = threadIdx.x;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int k = 0; k < 7; k++) {
                sh[k * blockDim.x + tid] += sh[k * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int k = 0; k < 7; k++) out7[k] = sh[k * blockDim.x];
    }
    __syncthreads();
}

extern "C" __global__ void kampylos_eval_candidates(
    const LightCurvePoint* data, int n_points,
    double log_s, double log_q,
    const Candidate* candidates, int n_candidates,
    CandidateResult* results
) {
    int cand_idx = blockIdx.x;
    if (cand_idx >= n_candidates) return;

    extern __shared__ double sh[]; // 7 * blockDim.x doubles
    double out7[7];

    Candidate c = candidates[cand_idx];
    double tE = exp(c.log_tE);
    double rho = exp(c.log_rho);

    // Same guard as binary_fit.h's objective() -- out-of-range tE/rho short-circuits to a fixed
    // bad chi2 without evaluating any magnification at all. Uniform across the whole block (same
    // candidate), so no warp divergence from this branch.
    if (tE < 0.01 || tE > 10000.0 || rho < 1e-6 || rho > 1.0) {
        if (threadIdx.x == 0) {
            results[cand_idx].chi2 = 1e18;
            results[cand_idx].fs = 0;
            results[cand_idx].fb = 0;
            results[cand_idx].needs_fallback = 0;
        }
        return;
    }

    double s = exp(log_s), q = exp(log_q);
    double salpha = sin(c.alpha), calpha = cos(c.alpha);
    double tE_inv = 1.0 / tE;

    double S_AA = 0, S_A1 = 0, S_11 = 0, S_Ay = 0, S_1y = 0;
    double any_invalid = 0, needs_fallback = 0;

    for (int i = threadIdx.x; i < n_points; i += blockDim.x) {
        double t = data[i].t;
        double tn = (t - c.t0) * tE_inv;
        double y1 = c.u0 * salpha - tn * calpha;
        double y2 = -c.u0 * calpha - tn * salpha;

        kgpu::BinaryMag0Result r = kgpu::binary_mag0_gpu(s, q, y1, y2);
        double A = r.mag;
        if (!(A > 0) || !isfinite(A)) {
            any_invalid = 1.0;
            continue; // this candidate is dead (chi2=1e18 below) -- no point summing further
        }
        if (!kgpu::binary_mag2_fastpath_ok(r, rho)) {
            needs_fallback = 1.0;
        }

        double sigma = data[i].sigma;
        double w = 1.0 / (sigma * sigma);
        double flux = data[i].flux;
        S_AA += w * A * A;
        S_A1 += w * A;
        S_11 += w;
        S_Ay += w * A * flux;
        S_1y += w * flux;
    }

    int tid = threadIdx.x;
    sh[0 * blockDim.x + tid] = S_AA;
    sh[1 * blockDim.x + tid] = S_A1;
    sh[2 * blockDim.x + tid] = S_11;
    sh[3 * blockDim.x + tid] = S_Ay;
    sh[4 * blockDim.x + tid] = S_1y;
    sh[5 * blockDim.x + tid] = any_invalid;
    sh[6 * blockDim.x + tid] = needs_fallback;
    __syncthreads();
    block_reduce7(sh, out7);

    if (out7[5] > 0) {
        // Same short-circuit as objective()'s "if (!isfinite(a) || a <= 0) return 1e18;" --
        // matches the CPU reference's behavior of treating one bad point as a dead candidate.
        if (tid == 0) {
            results[cand_idx].chi2 = 1e18;
            results[cand_idx].fs = 0;
            results[cand_idx].fb = 0;
            results[cand_idx].needs_fallback = 0; // irrelevant -- chi2 is already the CPU-matching sentinel, no recompute needed
        }
        return;
    }

    // fs, fb via the same closed-form weighted linear fit as lightcurve.h's linear_flux_fit().
    __shared__ double fs_sh, fb_sh;
    if (tid == 0) {
        double det = out7[0] * out7[2] - out7[1] * out7[1];
        if (fabs(det) < 1e-300) {
            fs_sh = 0;
            fb_sh = out7[2] > 0 ? out7[4] / out7[2] : 0;
        } else {
            fs_sh = (out7[2] * out7[3] - out7[1] * out7[4]) / det;
            fb_sh = (out7[0] * out7[4] - out7[1] * out7[3]) / det;
        }
    }
    __syncthreads();
    double fs = fs_sh, fb = fb_sh;

    // Second pass for chi2 -- linear_flux_fit() recomputes chi2 directly from (data, mag) rather
    // than using the sum-of-squares shortcut algebraically available from the 5 S_* sums above,
    // so this does the same (magnification recomputed per point rather than cached, trading a
    // second quintic solve per point for not needing O(n_points) of shared/global scratch space
    // per candidate -- the quintic solve is cheap next to everything else this kernel already
    // does per point).
    double chi2_partial = 0;
    for (int i = threadIdx.x; i < n_points; i += blockDim.x) {
        double t = data[i].t;
        double tn = (t - c.t0) * tE_inv;
        double y1 = c.u0 * salpha - tn * calpha;
        double y2 = -c.u0 * calpha - tn * salpha;
        kgpu::BinaryMag0Result r = kgpu::binary_mag0_gpu(s, q, y1, y2);
        double A = r.mag;
        double sigma = data[i].sigma;
        double w = 1.0 / (sigma * sigma);
        double model = fs * A + fb;
        double diff = data[i].flux - model;
        chi2_partial += w * diff * diff;
    }
    sh[tid] = chi2_partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        results[cand_idx].chi2 = sh[0];
        results[cand_idx].fs = fs;
        results[cand_idx].fb = fb;
        results[cand_idx].needs_fallback = (out7[6] > 0) ? 1 : 0;
    }
}
