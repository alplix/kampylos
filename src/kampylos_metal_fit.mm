// Host-side orchestration for Kampylos Metal (Apple GPU) -- same batched Nelder-Mead driving
// design as gpu_fit_binary.cu (CUDA) / kampylos_opencl_fit.cpp (OpenCL): NelderMeadStepper
// instances advanced in lockstep rounds, every active stepper's pending points batched into one
// Metal compute dispatch per round, kampylos_gpu_complete.h patching in the few per-candidate
// points that need the real VBMicrolensing finite-source treatment. Same overall shape as
// keraunos_metal_search.mm/metal_search.mm (persistent MTLBuffers, one MTLDevice = one real GPU
// on any Apple Silicon or Intel+discrete-GPU Mac).
//
// PRECISION, read before touching this file: Metal Shading Language has NO double type at all --
// a hardware limitation (Apple GPUs have no FP64 ALUs), not a syntax restriction, confirmed by an
// actual compile attempt ("'double' is not supported in Metal"). Measured directly (see
// tests/test_kampylos_mag_f32.cpp) that plain float32 is NOT an acceptable substitute here: it
// fails the quintic solver by up to 460x on realistic inputs, not just losing a few digits. The
// fix validated in tests/test_ds_primitives.cpp + tests/test_kampylos_mag_ds.cpp and used
// throughout this kernel's magnification math is double-single (float-float compensated)
// arithmetic -- NOT the same abandoned path the opencode/"big-pickle" attempt took (that
// transcript's own analysis found no evidence its primitives were ever validated in isolation;
// these were, against true double arithmetic, before anything was built on top of them). Measured
// ceiling: ~1.8e-7 worst-case relative error on the full magnification+fast-path test across 70k
// samples including the hardest (resonant topology, planetary mass ratio) regime -- more than
// adequate for Kampylos's own Tol=1e-2 accept/reject threshold.
//
// All buffer-facing structs here (LightCurvePointMSL/CandidateMSL/CandidateResultMSL) use float,
// not kampylos_gpu_types.h's double-based ones (which Metal can't represent in a buffer at all)
// -- float<->double conversion happens at upload (metal_fit_upload_light_curve) and at the
// completion step (metal_eval_batch, reusing kampylos_gpu_complete.h's existing double-based
// CandidateResult by converting the float kernel output into it).
//
// COMPILE OPTIONS, also before touching this file: the kernel must be compiled with fast math
// OFF (metal_fit_select_device sets it and the kampylos_ds_selftest kernel verifies it).
// Metal's default fast math is free to re-associate the error-free transformations the ds
// layer rests on, which silently turns it back into float32 -- the 460x failure above.
//
// ERRORS: every command buffer's status is checked and a failed one is re-submitted (see
// metal_eval_batch); a persistent failure throws std::runtime_error, which the callers turn
// into a clean exit. Never read buf_res after an unchecked buffer: it holds the previous round.
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "kampylos_gpu_types.h"
#include "kampylos_gpu_complete.h"
#include "kampylos_cpu_patch_pool.h"
#include "kampylos_metal_fit.h"
#include "nelder_mead_stepper.h"
#include "binary_fit.h"
#include "kampylos_batched_fit.h"
#include <algorithm>

static const char *kKampylosMSLSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// ---- Double-single (float-float compensated) scalar arithmetic -----------------------------
// Validated against true double precision in tests/test_ds_primitives.cpp (worst relative error
// 2.1e-10 across 500k cases including an adversarial catastrophic-cancellation case) -- this is
// the SAME algorithm (Knuth two-sum, FMA-based two-product, Newton-refined sqrt/div), re-typed
// into MSL. fma() is a real hardware instruction on every Metal-capable GPU, correctly rounded,
// which is what makes two_prod exact.
struct ds { float hi, lo; };
inline ds dsmake(float hi, float lo = 0.0) { ds r; r.hi = hi; r.lo = lo; return r; }
inline void two_sum(float a, float b, thread float& hi, thread float& lo) {
    hi = a + b; float bb = hi - a; lo = (a - (hi - bb)) + (b - bb);
}
inline void two_prod(float a, float b, thread float& hi, thread float& lo) {
    hi = a * b; lo = fma(a, b, -hi);
}
inline ds ds_add(ds a, ds b) {
    float s_hi, s_lo; two_sum(a.hi, b.hi, s_hi, s_lo); s_lo += a.lo + b.lo;
    float r_hi, r_lo; two_sum(s_hi, s_lo, r_hi, r_lo); return dsmake(r_hi, r_lo);
}
inline ds ds_neg(ds a) { return dsmake(-a.hi, -a.lo); }
inline ds ds_sub(ds a, ds b) { return ds_add(a, ds_neg(b)); }
inline ds ds_mul(ds a, ds b) {
    float p_hi, p_lo; two_prod(a.hi, b.hi, p_hi, p_lo); p_lo += a.hi * b.lo + a.lo * b.hi;
    float r_hi, r_lo; two_sum(p_hi, p_lo, r_hi, r_lo); return dsmake(r_hi, r_lo);
}
inline ds ds_mulf(ds a, float b) { return ds_mul(a, dsmake(b)); }
inline ds ds_div(ds a, ds b) {
    float q1 = a.hi / b.hi; ds r = ds_sub(a, ds_mul(b, dsmake(q1))); float q2 = r.hi / b.hi;
    float r_hi, r_lo; two_sum(q1, q2, r_hi, r_lo); return dsmake(r_hi, r_lo);
}
inline ds ds_sqrt(ds a) {
    if (a.hi <= 0) return dsmake(0, 0);
    float x = sqrt(a.hi); ds xd = dsmake(x);
    return ds_mulf(ds_add(xd, ds_div(a, xd)), 0.5);
}
inline bool ds_eq(ds a, ds b) { return a.hi == b.hi && a.lo == b.lo; }
inline float ds_val(ds a) { return a.hi + a.lo; } // single-float collapse -- fine for branch
                                                    // conditions/final outputs, not used mid-chain

// ---- Double-single complex arithmetic -------------------------------------------------------
struct dscx { ds re, im; };
inline dscx dcm(ds re, ds im) { dscx z; z.re = re; z.im = im; return z; }
inline dscx dcf(float re, float im = 0.0) { return dcm(dsmake(re), dsmake(im)); }
inline dscx dc_add(dscx a, dscx b) { return dcm(ds_add(a.re, b.re), ds_add(a.im, b.im)); }
inline dscx dc_sub(dscx a, dscx b) { return dcm(ds_sub(a.re, b.re), ds_sub(a.im, b.im)); }
inline dscx dc_mul(dscx a, dscx b) {
    return dcm(ds_sub(ds_mul(a.re, b.re), ds_mul(a.im, b.im)), ds_add(ds_mul(a.re, b.im), ds_mul(a.im, b.re)));
}
inline dscx dc_div(dscx a, dscx b) {
    ds md = ds_add(ds_mul(b.re, b.re), ds_mul(b.im, b.im));
    ds re = ds_div(ds_add(ds_mul(a.re, b.re), ds_mul(a.im, b.im)), md);
    ds im = ds_div(ds_sub(ds_mul(a.im, b.re), ds_mul(a.re, b.im)), md);
    return dcm(re, im);
}
inline dscx dc_scale(dscx z, float a) { return dcm(ds_mulf(z.re, a), ds_mulf(z.im, a)); }
inline dscx dc_neg(dscx z) { return dcm(ds_neg(z.re), ds_neg(z.im)); }
inline bool dc_eq(dscx a, dscx b) { return ds_eq(a.re, b.re) && ds_eq(a.im, b.im); }
inline float dc_abs(dscx z) { float re = ds_val(z.re), im = ds_val(z.im); return sqrt(re * re + im * im); }
inline dscx dc_conj(dscx z) { return dcm(z.re, ds_neg(z.im)); }
inline dscx dc_sqrt(dscx z) {
    float zre = ds_val(z.re), zim = ds_val(z.im);
    float md = sqrt(zre * zre + zim * zim);
    if (md <= 0) return dcf(0, 0);
    float rre = sqrt((md + zre) / 2), rim = sqrt((md - zre) / 2) * ((zim > 0) ? 1.0 : -1.0);
    return dcf(rre, rim);
}
inline dscx dc_exp(dscx z) {
    float zre = ds_val(z.re), zim = ds_val(z.im);
    float r = exp(zre);
    return dcf(r * cos(zim), r * sin(zim));
}

#define KAMPYLOS_MSL_MAXIT 2000

inline float abs2poly_ds(thread const dscx* poly, int degree, dscx z) {
    dscx pv = poly[degree];
    for (int k = degree - 1; k >= 0; k--) pv = dc_add(poly[k], dc_mul(z, pv));
    return ds_val(dc_mul(dc_conj(pv), pv).re);
}
inline void laguerre_ds(thread const dscx* poly, int degree, thread dscx* root) {
    const float FRAC_JUMPS[10] = { 0.64109297, 0.91577881, 0.25921289, 0.50487203, 0.08177045, 0.13653241, 0.306162, 0.37794326, 0.04618805, 0.75132137 };
    const float FRAC_ERR = 1.0e-6; // double-single's own measured ceiling, not double's 2e-15
    const float two_pi = 6.283185307;
    dscx c_one = dcf(1, 0), zero = dcf(0, 0);
    float one_nth = 1.0 / degree, n_1_nth = (degree - 1.0) * one_nth, two_n_div_n_1 = 2.0 / n_1_nth;
    dscx c_one_nth = dcf(one_nth, 0.0);
    for (int i = 1; i <= KAMPYLOS_MSL_MAXIT; i++) {
        float ek = dc_abs(poly[degree]);
        float absroot = dc_abs(*root);
        dscx p = poly[degree], dp = zero, d2p_half = zero;
        for (int k = degree - 1; k >= 0; k--) {
            d2p_half = dc_add(dp, dc_mul(d2p_half, *root));
            dp = dc_add(p, dc_mul(dp, *root));
            p = dc_add(poly[k], dc_mul(p, *root));
            ek = absroot * ek + dc_abs(p);
        }
        float abs2p = ds_val(dc_mul(dc_conj(p), p).re);
        if (abs2p == 0) return;
        float stopping_crit2 = (FRAC_ERR * ek) * (FRAC_ERR * ek);
        bool good_to_go = false;
        if (abs2p < stopping_crit2) { if (abs2p < 0.01 * stopping_crit2) return; good_to_go = true; }
        dscx denom = zero, dx;
        if (!dc_eq(dp, zero)) {
            dscx fac_newton = dc_div(p, dp);
            dscx fac_extra = dc_div(d2p_half, dp);
            dscx F_half = dc_mul(fac_newton, fac_extra);
            dscx denom_sqrt = dc_sqrt(dc_sub(c_one, dc_scale(F_half, two_n_div_n_1)));
            denom = dc_add(c_one_nth, dc_scale(denom_sqrt, n_1_nth));
            if (!dc_eq(denom, zero)) dx = dc_div(fac_newton, denom);
        }
        if (dc_eq(denom, zero)) dx = dc_scale(dc_exp(dcf(0.0, FRAC_JUMPS[i % 10] * two_pi)), absroot + 1.0);
        dscx newroot = dc_sub(*root, dx);
        if (dc_eq(newroot, *root)) return;
        if (good_to_go) { if (abs2poly_ds(poly, degree, newroot) < abs2p) *root = newroot; return; }
        if (i % 10 == 0) { float faq = FRAC_JUMPS[(i / 10 - 1) % 10]; newroot = dc_sub(*root, dc_scale(dx, faq)); }
        *root = newroot;
    }
}
inline void newton_spec_ds(thread const dscx* poly, int degree, thread dscx* root) {
    const float FRAC_JUMPS[10] = { 0.64109297, 0.91577881, 0.25921289, 0.50487203, 0.08177045, 0.13653241, 0.306162, 0.37794326, 0.04618805, 0.75132137 };
    const float FRAC_ERR = 1.0e-6;
    const float two_pi = 6.283185307;
    dscx zero = dcf(0, 0);
    float stopping_crit2 = 0.0;
    for (int i = 1; i <= KAMPYLOS_MSL_MAXIT; i++) {
        dscx p = poly[degree], dp = zero;
        if (i % 10 == 1) {
            float ek = dc_abs(poly[degree]); float absroot = dc_abs(*root);
            for (int k = degree - 1; k >= 0; k--) { dp = dc_add(p, dc_mul(dp, *root)); p = dc_add(poly[k], dc_mul(p, *root)); ek = absroot * ek + dc_abs(p); }
            stopping_crit2 = (FRAC_ERR * ek) * (FRAC_ERR * ek);
        } else {
            for (int k = degree - 1; k >= 0; k--) { dp = dc_add(p, dc_mul(dp, *root)); p = dc_add(poly[k], dc_mul(p, *root)); }
        }
        float abs2p = ds_val(dc_mul(dc_conj(p), p).re);
        if (abs2p == 0.0) return;
        bool good_to_go = false;
        if (abs2p < stopping_crit2) { if (dc_eq(dp, zero)) return; if (abs2p < 0.01 * stopping_crit2) return; good_to_go = true; }
        dscx dx;
        if (dc_eq(dp, zero)) dx = dc_scale(dc_exp(dcf(0.0, FRAC_JUMPS[i % 10] * two_pi)), dc_abs(*root) + 1.0);
        else dx = dc_div(p, dp);
        dscx newroot = dc_sub(*root, dx);
        if (dc_eq(newroot, *root)) return;
        if (good_to_go) { if (abs2poly_ds(poly, degree, newroot) < abs2p) *root = newroot; return; }
        if (i % 10 == 0) { float faq = FRAC_JUMPS[(i / 10 - 1) % 10]; newroot = dc_sub(*root, dc_scale(dx, faq)); }
        *root = newroot;
    }
}
inline void solve_quad_ds(thread dscx& x0, thread dscx& x1, thread const dscx* poly) {
    dscx a = poly[2], b = poly[1], c = poly[0];
    dscx b2 = dc_mul(b, b);
    dscx delta = dc_sqrt(dc_sub(b2, dc_scale(dc_mul(a, c), 4.0)));
    if (ds_val(dc_mul(dc_conj(b), delta).re) >= 0) x0 = dc_neg(dc_scale(dc_add(b, delta), 0.5));
    else x0 = dc_neg(dc_scale(dc_sub(b, delta), 0.5));
    if (dc_eq(x0, dcf(0, 0))) { x1 = dcf(0, 0); } else { x1 = dc_div(c, x0); x0 = dc_div(x0, a); }
}
#define KAMPYLOS_MSL_MAXDEG 6
inline void roots_gen_ds(thread dscx* roots, thread const dscx* poly, int degree) {
    dscx poly2[KAMPYLOS_MSL_MAXDEG];
    for (int j = 0; j <= degree; j++) poly2[j] = poly[j];
    for (int j = 0; j < degree; j++) roots[j] = dcf(0, 0);
    for (int n = degree; n >= 3; n--) {
        laguerre_ds(poly2, n, &roots[n - 1]);
        dscx coef = poly2[n];
        for (int i = n - 1; i >= 0; i--) { dscx prev = poly2[i]; poly2[i] = coef; coef = dc_add(prev, dc_mul(roots[n - 1], coef)); }
    }
    solve_quad_ds(roots[1], roots[0], poly2);
    for (int n = 0; n < degree; n++) newton_spec_ds(poly, degree, &roots[n]);
}

struct BinaryMag0Result { float mag, corrquad, corrquad2, safedist; int n_images; };

inline BinaryMag0Result binary_mag0_gpu(float s, float q, float y1v, float y2v) {
    BinaryMag0Result out; out.mag = -1.0; out.corrquad = 0.0; out.corrquad2 = 0.0; out.safedist = 10.0; out.n_images = 0;
    dscx a, qc;
    if (q < 1.0) { a = dcf(-s, 0); qc = dcf(q, 0); } else { a = dcf(s, 0); qc = dcf(1.0 / q, 0); }
    dscx m1 = dc_div(dcf(1, 0), dc_add(dcf(1, 0), qc));
    dscx m2 = dc_mul(qc, m1);
    dscx c20 = a, c21 = m1, c22 = m2;
    dscx c6 = dc_mul(a, a), c7 = dc_mul(c6, a), c8 = dc_mul(m2, m2), c9 = dc_mul(c6, c8), c10 = dc_mul(a, m2), c11 = dc_mul(a, m1);
    dscx y = dcf(y1v, y2v);
    dscx yfull = dc_add(y, c11);
    dscx yc = dc_conj(yfull);
    dscx c12 = dc_sub(c20, yc), c13 = dc_add(c20, yfull), c14 = dc_add(c13, yfull), c15 = dc_conj(c14);
    dscx c16 = dc_mul(c20, yfull), c17 = dc_conj(c16), c18 = dc_conj(c12);
    dscx poly[6];
    poly[0] = dc_mul(c9, yfull);
    poly[1] = dc_add(dc_neg(c9), dc_mul(c10, dc_add(c20, dc_mul(dc_sub(dc_scale(c17, 2.0), dc_add(dcf(2, 0), c6)), yfull))));
    poly[2] = dc_sub(dc_mul(c10, dc_sub(dc_add(dcf(1, 0), c16), dc_scale(dc_mul(yc, c13), 2.0))), dc_mul(dc_sub(c17, dcf(1, 0)), dc_sub(dc_mul(c16, c12), c18)));
    poly[3] = dc_sub(dc_add(dc_mul(c10, c15), dc_mul(dc_sub(dc_add(c7, dc_scale(dc_mul(dc_add(dcf(1,0),c6), yfull), 2.0)), dc_mul(c17, c14)), yc)), dc_mul(c20, c13));
    poly[4] = dc_sub(dc_neg(c10), dc_mul(c12, dc_sub(dc_mul(yc, dc_add(c13, c20)), dcf(1, 0))));
    poly[5] = dc_mul(yc, c12);
    thread dscx zr[5];
    roots_gen_ds(zr, poly, 5);
    float good[5];
    for (int i = 0; i < 5; i++) {
        dscx z = zr[i]; dscx zc = dc_conj(z);
        dscx ll = dc_add(dc_sub(yfull, z), dc_add(dc_div(c21, dc_sub(zc, c20)), dc_div(c22, zc)));
        good[i] = dc_abs(ll);
    }
    int order[5] = { 0, 1, 2, 3, 4 };
    for (int i = 0; i < 5; i++) for (int j = i + 1; j < 5; j++) if (good[order[j]] > good[order[i]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
    int worst1 = order[0], worst2 = order[1], worst3 = order[2];
    const float dlmin = 1.0e-4;
    float mag = 0.0; int n_images = 0;
    bool five_roots = !(good[worst2] * dlmin > good[worst3] + 1.e-12);
    float corrquad = 0.0;
    for (int i = 0; i < 5; i++) {
        if (!five_roots && (i == worst1 || i == worst2)) continue;
        dscx z = zr[i]; dscx dza = dc_sub(z, c20); dscx za2 = dc_mul(dza, dza); dscx zb2 = dc_mul(z, z);
        dscx J1 = dc_add(dc_div(c21, za2), dc_div(c22, zb2)); dscx J1c = dc_conj(J1);
        dscx dJ = dc_sub(dcf(1, 0), dc_mul(J1, J1c));
        dscx J2 = dc_neg(dc_scale(dc_add(dc_div(c21, dc_mul(za2, dza)), dc_div(c22, dc_mul(zb2, z))), 2.0));
        float dJre = ds_val(dJ.re);
        if (fabs(dJre) < 1.e-9) continue;
        mag += fabs(1.0 / dJre); n_images++;
        dscx J3 = dc_scale(dc_add(dc_div(c21, dc_mul(za2, za2)), dc_div(c22, dc_mul(zb2, zb2))), 6.0);
        float dJ2 = dJre * dJre;
        dscx za2c = dc_mul(J1c, J1c);
        dscx J3s = dc_mul(J3, za2c);
        float J2re = ds_val(J2.re), J2im = ds_val(J2.im);
        float ob2 = (J2re * J2re + J2im * J2im) * (6.0 - 6.0 * dJre + dJ2);
        dscx J2s = dc_mul(dc_mul(J2, J2), dc_mul(za2c, J1c));
        float cq = 0.5 * (fabs(ob2 - 6.0 * ds_val(J2s.re) - 2.0 * ds_val(J3s.re) * dJre) + 3.0 * fabs(ds_val(J2s.im))) / fabs(dJre * dJ2 * dJ2);
        corrquad += cq;
    }
    float corrquad2;
    if (!five_roots) {
        int idxs[2] = { worst1, worst2 }; float cq2 = -2.0;
        for (int k = 0; k < 2; k++) {
            int i = idxs[k];
            dscx z = zr[i]; dscx dza = dc_sub(z, c20); dscx za2 = dc_mul(dza, dza); dscx zb2 = dc_mul(z, z);
            dscx J1 = dc_add(dc_div(c21, za2), dc_div(c22, zb2)); dscx J1c = dc_conj(J1);
            dscx dJ = dc_sub(dcf(1, 0), dc_mul(J1, J1c));
            dscx J2 = dc_neg(dc_scale(dc_add(dc_div(c21, dc_mul(za2, dza)), dc_div(c22, dc_mul(zb2, z))), 2.0));
            dscx zaltc = dc_add(yc, dc_add(dc_div(c21, dza), dc_div(c22, z)));
            dscx za2b = dc_sub(zaltc, c20);
            dscx Jaltc = dc_add(dc_div(c21, dc_mul(za2b, za2b)), dc_div(c22, dc_mul(zaltc, zaltc)));
            dscx Jalt = dc_conj(Jaltc);
            dscx JJalt2 = dc_sub(dcf(1, 0), dc_mul(J1c, Jalt));
            dscx J3 = dc_mul(dc_mul(J2, J1c), JJalt2);
            J3 = dc_div(dc_sub(J3, dc_mul(dc_conj(J3), Jalt)), dc_scale(dc_mul(JJalt2, JJalt2), ds_val(dJ.re)));
            float cq = (ds_val(dJ.re) < -10) ? -1.0 : (ds_val(J3.re) * ds_val(J3.re) + ds_val(J3.im) * ds_val(J3.im));
            if (k == 0) cq2 = cq; else { if (cq2 < 0 || cq < 0) cq2 = 0; else if (cq > cq2) cq2 = cq; }
        }
        corrquad2 = cq2;
    } else corrquad2 = 0.0;
    float safedist;
    float a_re = ds_val(a.re), qc_re = ds_val(qc.re), c21_re = ds_val(c21.re);
    if (qc_re < 0.01) {
        float sd = y1v + (a_re - 1.0 / a_re) * c21_re;
        safedist = sd * sd + y2v * y2v - 4.0 * sqrt(qc_re) / (a_re * a_re);
    } else {
        float sabs = fabs(a_re); float Rinf = sabs + 1.0 / sabs + 2.0;
        float d2 = y1v * y1v + y2v * y2v; safedist = 10.0;
        if (d2 > Rinf * Rinf) { float d = sqrt(d2); safedist = (d - Rinf) * (d - Rinf); }
    }
    out.mag = (n_images > 0) ? mag : -1.0;
    out.corrquad = corrquad; out.corrquad2 = corrquad2; out.safedist = safedist; out.n_images = n_images;
    return out;
}
inline bool binary_mag2_fastpath_ok(thread const BinaryMag0Result& r, float rho, float Tol) {
    float rho2 = rho * rho;
    float cq = r.corrquad * 6.0 * (rho2 + 1.e-4 * Tol);
    float cq2 = r.corrquad2 * 256.0 * (rho2 + 1.e-8);
    return (cq < Tol) && (cq2 < 1.0) && (r.safedist > 4.0 * rho2);
}

// ---- Kernel entry point: the six per-candidate running sums are formed AND accumulated in
// double-single (2026-10-10). They used to be float products (w*A*A ~ 1e6 with a 6e-8 relative
// rounding error each) under Kahan summation; Kahan protects the sum, not the products, and the
// products' rounding alone left 0.1-0.4 chi2 of candidate-to-candidate noise in the objective
// (measured on the synthetic planet cell against the CPU chi2) -- enough to steer Nelder-Mead
// (ftol 1e-9) into a wrong basin: that cell ended at chi2 689 instead of the CPU's 411. With ds
// products and sums the disagreement is ~0.002 and the cell reproduces the CPU result exactly.
// Each sum is returned as a hi/lo float pair and promoted to double on the host.
struct LightCurvePointMSL { float t, flux, sigma; };
struct CandidateMSL { float t0, u0, log_tE, alpha, log_rho; };

#define KAMPYLOS_MSL_MAX_BAD_POINTS 512
struct CandidateResultMSL {
    float S_AA, S_A1, S_11, S_Ay, S_1y, S_yy;
    float S_AA_lo, S_A1_lo, S_11_lo, S_Ay_lo, S_1y_lo, S_yy_lo;
    int n_bad;
    int overflow;
    int bad_idx[KAMPYLOS_MSL_MAX_BAD_POINTS];
    int poisoned;
};
#define KAMPYLOS_MSL_NREDUCE 13   // 6 sums x (hi, lo) + the "any point invalid" flag

kernel void kampylos_eval_candidates(
    device const LightCurvePointMSL* data [[buffer(0)]],
    constant int& n_points [[buffer(1)]],
    constant float& s [[buffer(2)]],   // LINEAR separation / mass ratio (2026-10-08)
    constant float& q [[buffer(3)]],
    device const CandidateMSL* candidates [[buffer(4)]],
    constant int& n_candidates [[buffer(5)]],
    device CandidateResultMSL* results [[buffer(6)]],
    threadgroup float* sh [[threadgroup(0)]],
    threadgroup atomic_int* bad_count [[threadgroup(1)]],
    threadgroup atomic_int* overflow_flag [[threadgroup(2)]],
    uint cand_idx [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    if ((int)cand_idx >= n_candidates) return;

    if (lid == 0) {
        atomic_store_explicit(bad_count, 0, memory_order_relaxed);
        atomic_store_explicit(overflow_flag, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    CandidateMSL c = candidates[cand_idx];
    float tE = exp(c.log_tE);
    float rho = exp(c.log_rho);

    if (tE < 0.01 || tE > 10000.0 || rho < 1e-6 || rho > 1.0) {
        if (lid == 0) {
            device CandidateResultMSL& r = results[cand_idx];
            r.S_AA = r.S_A1 = r.S_11 = r.S_Ay = r.S_1y = r.S_yy = 0;
            r.n_bad = 0; r.overflow = 0; r.poisoned = 1;
        }
        return;
    }

    float salpha = sin(c.alpha), calpha = cos(c.alpha);
    float tE_inv = 1.0 / tE;

    ds S_AA = dsmake(0), S_A1 = dsmake(0), S_11 = dsmake(0), S_Ay = dsmake(0), S_1y = dsmake(0), S_yy = dsmake(0);
    float any_invalid = 0;

    for (int i = (int)lid; i < n_points; i += (int)lsize) {
        float t = data[i].t;
        float tn = (t - c.t0) * tE_inv;
        float y1 = c.u0 * salpha - tn * calpha;
        float y2 = -c.u0 * calpha - tn * salpha;

        BinaryMag0Result r = binary_mag0_gpu(s, q, y1, y2);
        float A = r.mag;
        if (!(A > 0) || !isfinite(A)) {
            any_invalid = 1.0;
            continue;
        }
        if (!binary_mag2_fastpath_ok(r, rho, 1.e-2)) {
            int slot = atomic_fetch_add_explicit(bad_count, 1, memory_order_relaxed);
            if (slot < KAMPYLOS_MSL_MAX_BAD_POINTS) {
                results[cand_idx].bad_idx[slot] = i;
            } else {
                atomic_store_explicit(overflow_flag, 1, memory_order_relaxed);
            }
            continue;
        }

        ds dsig = dsmake(data[i].sigma);
        ds dw = ds_div(dsmake(1.0), ds_mul(dsig, dsig));
        ds dA = dsmake(A);
        ds df = dsmake(data[i].flux);
        S_AA = ds_add(S_AA, ds_mul(dw, ds_mul(dA, dA)));
        S_A1 = ds_add(S_A1, ds_mul(dw, dA));
        S_11 = ds_add(S_11, dw);
        S_Ay = ds_add(S_Ay, ds_mul(dw, ds_mul(dA, df)));
        S_1y = ds_add(S_1y, ds_mul(dw, df));
        S_yy = ds_add(S_yy, ds_mul(dw, ds_mul(df, df)));
    }

    // Lanes 0-5 hold the hi parts, 6-11 the lo parts, 12 the invalid flag.
    int tid = (int)lid, ls = (int)lsize;
    sh[0 * ls + tid] = S_AA.hi; sh[6 * ls + tid] = S_AA.lo;
    sh[1 * ls + tid] = S_A1.hi; sh[7 * ls + tid] = S_A1.lo;
    sh[2 * ls + tid] = S_11.hi; sh[8 * ls + tid] = S_11.lo;
    sh[3 * ls + tid] = S_Ay.hi; sh[9 * ls + tid] = S_Ay.lo;
    sh[4 * ls + tid] = S_1y.hi; sh[10 * ls + tid] = S_1y.lo;
    sh[5 * ls + tid] = S_yy.hi; sh[11 * ls + tid] = S_yy.lo;
    sh[12 * ls + tid] = any_invalid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (int k = 0; k < 6; k++) {
                ds a = dsmake(sh[k * ls + tid], sh[(k + 6) * ls + tid]);
                ds b = dsmake(sh[k * ls + tid + (int)stride], sh[(k + 6) * ls + tid + (int)stride]);
                ds r = ds_add(a, b);
                sh[k * ls + tid] = r.hi; sh[(k + 6) * ls + tid] = r.lo;
            }
            sh[12 * ls + tid] += sh[12 * ls + tid + (int)stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        device CandidateResultMSL& r = results[cand_idx];
        if (sh[12 * ls] > 0) {
            r.S_AA = r.S_A1 = r.S_11 = r.S_Ay = r.S_1y = r.S_yy = 0;
            r.S_AA_lo = r.S_A1_lo = r.S_11_lo = r.S_Ay_lo = r.S_1y_lo = r.S_yy_lo = 0;
            r.n_bad = 0; r.overflow = 0; r.poisoned = 1;
        } else {
            r.S_AA = sh[0 * ls]; r.S_A1 = sh[1 * ls]; r.S_11 = sh[2 * ls];
            r.S_Ay = sh[3 * ls]; r.S_1y = sh[4 * ls]; r.S_yy = sh[5 * ls];
            r.S_AA_lo = sh[6 * ls]; r.S_A1_lo = sh[7 * ls]; r.S_11_lo = sh[8 * ls];
            r.S_Ay_lo = sh[9 * ls]; r.S_1y_lo = sh[10 * ls]; r.S_yy_lo = sh[11 * ls];
            r.n_bad = min(atomic_load_explicit(bad_count, memory_order_relaxed), (int)KAMPYLOS_MSL_MAX_BAD_POINTS);
            r.overflow = atomic_load_explicit(overflow_flag, memory_order_relaxed);
            r.poisoned = 0;
        }
    }
}

// Guard against the compiler re-associating the error-free transformations everything above
// rests on (Metal's DEFAULT is fast math, which is allowed to do exactly that and silently turns
// the whole double-single layer back into plain float32 -- see the host's compile options).
// Both lo parts are exact powers of two; with fast math they come out as 0.
kernel void kampylos_ds_selftest(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float a = 1.0f + 1.0f / 4096.0f;                  // (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24
    float p_hi, p_lo; two_prod(a, a, p_hi, p_lo);
    out[0] = p_lo;                                    // exact: 2^-24
    float s_hi, s_lo; two_sum(1.0f, 1.0f / 1073741824.0f, s_hi, s_lo);
    out[1] = s_lo;                                    // exact: 2^-30
}
)MSL";

// Host-side (Objective-C++) mirrors of the MSL kernel's own LightCurvePointMSL/CandidateMSL/
// CandidateResultMSL structs above -- the MSL string's struct definitions only exist inside that
// string, compiled separately by Metal at runtime; they're invisible to this surrounding host
// code, which needs its OWN matching definitions (same field order/types -> same memory layout)
// to build/read the actual MTLBuffers.
struct LightCurvePointMSL { float t, flux, sigma; };
struct CandidateMSL { float t0, u0, log_tE, alpha, log_rho; };
#define KAMPYLOS_MSL_MAX_BAD_POINTS_HOST 512
struct CandidateResultMSL {
    float S_AA, S_A1, S_11, S_Ay, S_1y, S_yy;
    float S_AA_lo, S_A1_lo, S_11_lo, S_Ay_lo, S_1y_lo, S_yy_lo;
    int n_bad;
    int overflow;
    int bad_idx[KAMPYLOS_MSL_MAX_BAD_POINTS_HOST];
    int poisoned;
};
#define KAMPYLOS_MSL_NREDUCE_HOST 13

// Work per command buffer, in candidate x data-point evaluations. Measured on a base M1: 1440
// candidates x 4010 points (5.8e6) = 0.69 s, so this is ~0.5 s per buffer. Short buffers are not
// a watchdog workaround (a 3.4 s buffer completes fine on an idle GPU) but what makes the retry
// below cheap: when another process is hammering the same GPU the driver sporadically aborts
// in-flight command buffers of EVERY process with "Internal Error (0000000e)", and a buffer
// that is re-submitted after that normally completes.
#define KAMPYLOS_METAL_BUFFER_BUDGET 4.0e6
#define KAMPYLOS_METAL_MAX_ATTEMPTS 6

// ---------------------------------------------------------------------------------------------
// Host-side orchestration -- same shape as gpu_fit_binary.cu/kampylos_opencl_fit.cpp, differing
// only in using float-typed buffers/structs (LightCurvePointMSL/CandidateMSL/CandidateResultMSL,
// matching the kernel's own types above) instead of the double-typed ones those backends use
// directly from kampylos_gpu_types.h -- Metal can't represent double in a buffer at all. The
// float<->double conversion happens right at this boundary: upload truncates double photometry
// to float (losing precision only in the INPUT data's own representation, same as any float-based
// GPU path would for the raw light curve values -- not the quintic solver's internal math, which
// stays double-single throughout), and kampylos_complete_candidate()'s own CandidateResult
// (double-based, shared with every other backend) gets its fields promoted back from this
// kernel's float output before the CPU-side completion step runs.
struct MetalFitImpl {
    // 2026-10-08: times are uploaded RELATIVE to t_ref (and candidate t0 likewise). They used to
    // be raw HJD (~2.45e6) cast to float, whose resolution there is 0.25 day -- every
    // (t - t0) the kernel formed was quantised to 6 hours, far coarser than a planetary anomaly.
    // Relative to the light curve's median time the resolution is ~1e-4 day over a decade-long
    // baseline. Fluxes are centred (kampylos_flux_offset) to keep the float chi2 identity sane.
    std::vector<LightCurvePoint> h_data;  // double, original t, centred flux
    double t_ref = 0.0, flux_offset = 0.0, fs_bound = 1e300;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLBuffer> buf_data = nil;
    int n_points = 0;
    id<MTLBuffer> buf_cand = nil;
    id<MTLBuffer> buf_res = nil;
    int cand_capacity = 0;
    int local_size = 128;          // threads per threadgroup (power of two, <= the pipeline's maximum)
    long buffers_retried = 0;      // command buffers that failed at least once and were re-run
    bool setup_transient = false;  // the last metal_fit_select_device() failure was "no usable GPU right
                                   // now" (worth retrying later), not a compiler/pipeline problem
};

MetalFitContext::MetalFitContext() { impl = new MetalFitImpl(); }
MetalFitContext::~MetalFitContext() {
    MetalFitImpl* m = (MetalFitImpl*)impl;
    delete m;
}

bool metal_fit_select_device(MetalFitContext& ctx, std::string* err_out) {
    MetalFitImpl* m = (MetalFitImpl*)ctx.impl;
    m->setup_transient = false;
    m->device = MTLCreateSystemDefaultDevice();
    if (!m->device) {
        // 2026-10-12: MTLCreateSystemDefaultDevice() resolves the DISPLAY's GPU (Apple documents
        // that it needs CoreGraphics linked), and from a process without a window-server session
        // it can come back nil although the GPU is right there -- and a BOINC task is exactly
        // that: started by the client daemon as the boinc_project user, possibly with nobody
        // logged in. MTLCopyAllDevices() enumerates the GPUs without that dependency. Apple
        // Silicon has one GPU, so the choice is unambiguous; an Intel Mac with two gets the
        // non-low-power one, as the system default would pick.
        NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
        for (id<MTLDevice> d in all)
            if (!m->device || (m->device.isLowPower && !d.isLowPower)) m->device = d;
        if (m->device)
            fprintf(stderr, "metal_fit_select_device: MTLCreateSystemDefaultDevice() returned nil; using \"%s\" from MTLCopyAllDevices() (%lu device(s))\n",
                    [[m->device name] UTF8String], (unsigned long)all.count);
    }
    if (!m->device) {
        m->setup_transient = true;
        if (err_out) *err_out = "no Metal device found (MTLCreateSystemDefaultDevice and MTLCopyAllDevices both returned nothing)";
        return false;
    }
    fprintf(stderr, "metal_fit_select_device: using Metal device \"%s\"\n", [[m->device name] UTF8String]);

    m->queue = [m->device newCommandQueue];
    if (!m->queue) { if (err_out) *err_out = "newCommandQueue failed"; return false; }

    NSError* error = nil;
    NSString* src = [NSString stringWithUTF8String:kKampylosMSLSource];
    // 2026-10-10: Metal compiles with fast math unless told otherwise, and fast math may
    // re-associate `lo = (a - (hi - bb)) + (b - bb)` and `lo = fma(a, b, -hi)` to 0 -- the whole
    // double-single layer silently degrades to float32, the quintic solver then fails on
    // ordinary planetary-regime points and the kernel poisons those candidates (measured: the
    // TRUE parameters of the synthetic planet evaluated to chi2 = 1e18 on an M1, and the cell
    // converged to 692 instead of 411). fastMathEnabled is the macOS 10.x-14 spelling, mathMode
    // the macOS 15 one; set both, and verify with the self-test kernel below.
    MTLCompileOptions* copts = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    copts.fastMathEnabled = NO;
#pragma clang diagnostic pop
#if defined(__MAC_15_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) copts.mathMode = MTLMathModeSafe;
#endif
    id<MTLLibrary> library = [m->device newLibraryWithSource:src options:copts error:&error];
    if (!library) {
        if (err_out) *err_out = std::string("MSL compile failed: ") + (error ? [[error localizedDescription] UTF8String] : "unknown");
        return false;
    }
    id<MTLFunction> fn = [library newFunctionWithName:@"kampylos_eval_candidates"];
    if (!fn) { if (err_out) *err_out = "kernel function not found"; return false; }
    m->pipeline = [m->device newComputePipelineStateWithFunction:fn error:&error];
    if (!m->pipeline) {
        if (err_out) *err_out = std::string("newComputePipelineStateWithFunction failed: ") + (error ? [[error localizedDescription] UTF8String] : "unknown");
        return false;
    }
    // The reduction needs a power of two; 128 fits every Apple GPU but ask rather than assume.
    int max_threads = (int)m->pipeline.maxTotalThreadsPerThreadgroup;
    m->local_size = 128;
    while (m->local_size > max_threads && m->local_size > 1) m->local_size /= 2;

    // Self-test: both lo parts are exact powers of two unless the compiler broke the error-free
    // transformations (see above). Refuse to run rather than return garbage.
    id<MTLFunction> self_fn = [library newFunctionWithName:@"kampylos_ds_selftest"];
    id<MTLComputePipelineState> self_pipe = self_fn ? [m->device newComputePipelineStateWithFunction:self_fn error:&error] : nil;
    if (!self_pipe) { if (err_out) *err_out = "double-single self-test kernel not found"; return false; }
    id<MTLBuffer> self_buf = [m->device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    memset([self_buf contents], 0, 16);
    id<MTLCommandBuffer> cb = [m->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:self_pipe];
    [enc setBuffer:self_buf offset:0 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    const float* lo = (const float*)[self_buf contents];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        // The device would not even run a one-thread kernel: not a compiler problem but a GPU
        // that is busy or lost right now (the same aborts metal_eval_batch retries), so the
        // caller may try again later rather than give the work unit up.
        m->setup_transient = true;
        if (err_out) *err_out = std::string("self-test command buffer failed (status ") + std::to_string((long)cb.status) + "): " +
                                (cb.error ? [[cb.error localizedDescription] UTF8String] : "no error object");
        return false;
    }
    if (lo[0] != 1.0f / 16777216.0f || lo[1] != 1.0f / 1073741824.0f) {
        char buf[200];
        snprintf(buf, sizeof(buf), "double-single self-test failed (two_prod lo = %g, two_sum lo = %g): the Metal compiler applied fast math, results would be wrong", lo[0], lo[1]);
        if (err_out) *err_out = buf;
        return false;
    }
    return true;
}

void metal_fit_upload_light_curve(MetalFitContext& ctx, const std::vector<DataPoint>& data) {
    MetalFitImpl* m = (MetalFitImpl*)ctx.impl;
    std::vector<double> ts;
    ts.reserve(data.size());
    for (const auto& d : data) ts.push_back(d.t);
    std::sort(ts.begin(), ts.end());
    m->t_ref = ts.empty() ? 0.0 : ts[ts.size() / 2];
    m->flux_offset = kampylos_flux_offset(data);
    m->fs_bound = flux_fit_bound(data);
    m->h_data.resize(data.size());
    std::vector<LightCurvePointMSL> h_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        m->h_data[i] = { data[i].t, data[i].flux - m->flux_offset, data[i].sigma };
        h_data[i].t = (float)(data[i].t - m->t_ref);
        h_data[i].flux = (float)(data[i].flux - m->flux_offset);
        h_data[i].sigma = (float)data[i].sigma;
    }
    m->n_points = (int)h_data.size();
    m->buf_data = [m->device newBufferWithBytes:h_data.data()
                                          length:h_data.size() * sizeof(LightCurvePointMSL)
                                         options:MTLResourceStorageModeShared];
}

static void ensure_cand_capacity(MetalFitImpl* m, int n) {
    if (n <= m->cand_capacity) return;
    m->buf_cand = [m->device newBufferWithLength:n * sizeof(CandidateMSL) options:MTLResourceStorageModeShared];
    m->buf_res = [m->device newBufferWithLength:n * sizeof(CandidateResultMSL) options:MTLResourceStorageModeShared];
    m->cand_capacity = n;
}

static std::vector<double> metal_eval_batch(
    MetalFitContext& ctx, double s, double q,
    const std::vector<Candidate>& points
) {
    MetalFitImpl* m = (MetalFitImpl*)ctx.impl;
    int n = (int)points.size();
    std::vector<double> chi2(n);
    if (n == 0) return chi2;

    ensure_cand_capacity(m, n);
    std::vector<CandidateMSL> h_cand(n);
    for (int i = 0; i < n; i++) {
        h_cand[i].t0 = (float)(points[i].t0 - m->t_ref); h_cand[i].u0 = (float)points[i].u0;
        h_cand[i].log_tE = (float)points[i].log_tE; h_cand[i].alpha = (float)points[i].alpha;
        h_cand[i].log_rho = (float)points[i].log_rho;
    }
    memcpy([m->buf_cand contents], h_cand.data(), n * sizeof(CandidateMSL));

    float s_f = (float)s, q_f = (float)q;

    // One command buffer per chunk of candidates (see KAMPYLOS_METAL_BUFFER_BUDGET), each checked
    // and re-submitted on failure. Before 2026-10-10 the whole batch went into one buffer whose
    // status was never read: an aborted buffer left the previous round's numbers in buf_res and
    // the fit went on with them.
    int chunk = (int)(KAMPYLOS_METAL_BUFFER_BUDGET / std::max(1, m->n_points));
    chunk = std::max(16, std::min(chunk, 4096));
    for (int c0 = 0; c0 < n; c0 += chunk) {
        int nc = std::min(chunk, n - c0);
        std::string last_error;
        bool done = false;
        for (int attempt = 1; attempt <= KAMPYLOS_METAL_MAX_ATTEMPTS && !done; attempt++) {
            id<MTLCommandBuffer> cmdbuf = [m->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            [enc setComputePipelineState:m->pipeline];
            [enc setBuffer:m->buf_data offset:0 atIndex:0];
            [enc setBytes:&m->n_points length:sizeof(int) atIndex:1];
            [enc setBytes:&s_f length:sizeof(float) atIndex:2];
            [enc setBytes:&q_f length:sizeof(float) atIndex:3];
            [enc setBuffer:m->buf_cand offset:c0 * sizeof(CandidateMSL) atIndex:4];
            [enc setBytes:&nc length:sizeof(int) atIndex:5];
            [enc setBuffer:m->buf_res offset:c0 * sizeof(CandidateResultMSL) atIndex:6];
            // threadgroup memory lengths must be multiples of 16 bytes
            [enc setThreadgroupMemoryLength:KAMPYLOS_MSL_NREDUCE_HOST * m->local_size * sizeof(float) atIndex:0];
            [enc setThreadgroupMemoryLength:16 atIndex:1];
            [enc setThreadgroupMemoryLength:16 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(nc, 1, 1) threadsPerThreadgroup:MTLSizeMake(m->local_size, 1, 1)];
            [enc endEncoding];
            [cmdbuf commit];
            [cmdbuf waitUntilCompleted];
            if (cmdbuf.status == MTLCommandBufferStatusCompleted && cmdbuf.error == nil) { done = true; break; }
            last_error = cmdbuf.error ? [[cmdbuf.error localizedDescription] UTF8String] : "no error object";
            fprintf(stderr, "metal_eval_batch: command buffer (%d candidates) failed, status %ld: %s -- attempt %d of %d\n",
                    nc, (long)cmdbuf.status, last_error.c_str(), attempt, KAMPYLOS_METAL_MAX_ATTEMPTS);
            if (attempt == 1) m->buffers_retried++;
            usleep(200000 * attempt);
        }
        if (!done) throw std::runtime_error("Metal command buffer failed " + std::to_string(KAMPYLOS_METAL_MAX_ATTEMPTS) +
                                            " times in a row: " + last_error);
    }

    std::vector<CandidateResultMSL> h_res_f(n);
    memcpy(h_res_f.data(), [m->buf_res contents], n * sizeof(CandidateResultMSL));

    // Promote the kernel's float output into the shared (double-based) CandidateResult type so
    // kampylos_complete_candidate() -- unchanged, same as every other backend -- can patch in the
    // bad points via the real VBMicrolensing library and derive the final chi2 in double. Each
    // sum is a double-single pair; hi + lo in double is exact. This conversion pass is cheap
    // (plain struct field copies, no VBMicrolensing calls) and stays sequential; only the actual
    // patch-completion step below is worth parallelizing.
    std::vector<CandidateResult> results(n);
    for (int i = 0; i < n; i++) {
        CandidateResult& r = results[i];
        const CandidateResultMSL& f = h_res_f[i];
        r.S_AA = (double)f.S_AA + (double)f.S_AA_lo; r.S_A1 = (double)f.S_A1 + (double)f.S_A1_lo;
        r.S_11 = (double)f.S_11 + (double)f.S_11_lo; r.S_Ay = (double)f.S_Ay + (double)f.S_Ay_lo;
        r.S_1y = (double)f.S_1y + (double)f.S_1y_lo; r.S_yy = (double)f.S_yy + (double)f.S_yy_lo;
        r.n_bad = f.n_bad; r.overflow = f.overflow; r.poisoned = f.poisoned;
        for (int k = 0; k < f.n_bad && k < KAMPYLOS_MAX_BAD_POINTS; k++) r.bad_idx[k] = f.bad_idx[k];
    }
    // Same fix as gpu_fit_binary.cu's own gpu_eval_batch() (see that file and
    // kampylos_cpu_patch_pool.h for the full story) -- this was the same single-threaded
    // bottleneck for a pathological anchor, now spread across a small worker-thread pool instead.
    // Created once, never destroyed: see kampylos_cpu_patch_pool.h (persistent workers).
    static KampylosCpuPatchPool& patch_pool = *new KampylosCpuPatchPool();   // kampylos_patch_threads() workers
    patch_pool.complete_batch(results, points, m->h_data.data(), m->n_points, s, q, chi2, m->fs_bound, m->flux_offset);
    return chi2;
}

std::vector<double> metal_fit_eval_chi2(MetalFitContext& ctx, double s, double q, const std::vector<Candidate>& points) {
    return metal_eval_batch(ctx, s, q, points);
}

long metal_fit_buffers_retried(const MetalFitContext& ctx) {
    return ((const MetalFitImpl*)ctx.impl)->buffers_retried;
}

bool metal_fit_setup_failure_is_transient(const MetalFitContext& ctx) {
    return ((const MetalFitImpl*)ctx.impl)->setup_transient;
}

BinaryFitResult metal_fit_binary_seeds(
    MetalFitContext& ctx,
    const std::vector<DataPoint>& data,
    double ln_s, double ln_q,
    const BinarySeedSet& seeds,
    int n_restarts,
    const KampylosBatchedHooks* hooks,
    int cell
) {
    (void)data;
    double s = exp(ln_s), q = exp(ln_q);
    auto eval = [&](const std::vector<Candidate>& batch) { return metal_eval_batch(ctx, s, q, batch); };
    return kampylos_batched_fit_seeds(eval, ln_s, ln_q, seeds, n_restarts, hooks, cell, stderr);
}

BinaryFitResult metal_fit_binary_multistart(
    MetalFitContext& ctx,
    VBMicrolensing& vbm,
    const std::vector<DataPoint>& data,
    double log_s, double log_q,
    double t0_anchor, double u0_anchor, double tE_anchor,
    double rho_seed,
    int n_alpha_seeds,
    int n_restarts
) {
    MetalFitImpl* m = (MetalFitImpl*)ctx.impl;
    if (m->n_points != (int)data.size()) metal_fit_upload_light_curve(ctx, data);
    BinarySeedSet seeds = kampylos_legacy_multistart_seeds(t0_anchor, u0_anchor, tE_anchor, rho_seed, n_alpha_seeds);
    BinaryFitResult out = metal_fit_binary_seeds(ctx, data, log_s, log_q, seeds, n_restarts);
    std::vector<double> mag(data.size());
    double pr[7] = { log_s, log_q, out.u0, out.alpha, log(out.rho), log(out.tE), out.t0 };
    for (size_t i = 0; i < data.size(); i++) mag[i] = vbm.BinaryLightCurve(pr, data[i].t);
    FluxFit ff = linear_flux_fit(data, mag);
    out.fs = ff.fs; out.fb = ff.fb;
    return out;
}
