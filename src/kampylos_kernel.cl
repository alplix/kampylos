// Kampylos's batch candidate-evaluation kernel, OpenCL C port of kampylos_gpu_mag.cuh +
// kampylos_gpu_kernel.cu -- same algorithm (point-source + quadrupole/hexadecapole fast path,
// per-point fallback reporting, the chi2=S_yy-fs*S_Ay-fb*S_1y closed form), same validated
// correctness, just OpenCL C instead of CUDA C++ since OpenCL C (based on C99) has no classes,
// operator overloading, references or templates -- every `kgpu::cx` operator becomes a named
// function (cx_add, cx_mul, ...), every struct method becomes a free function taking the struct
// by value or a pointer to it.
//
// Requires cl_khr_fp64 (double precision) -- this math was deliberately kept in native double
// throughout the CUDA version (see that file's own top comment: an earlier, abandoned GPU-port
// attempt burned a full session chasing FP32 + emulated-double precision bugs in exactly this
// quintic solver, for no real throughput benefit). Every current discrete desktop GPU this
// project targets (AMD/Intel/NVIDIA) supports this extension; a device that doesn't gets
// filtered out by the host-side init code the same way a device with too little GPU RAM already
// is, not silently given wrong answers.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef struct { double re, im; } cx;

inline cx cx_make(double re, double im) { cx z; z.re = re; z.im = im; return z; }
inline cx cx_add(cx a, cx b) { return cx_make(a.re + b.re, a.im + b.im); }
inline cx cx_sub(cx a, cx b) { return cx_make(a.re - b.re, a.im - b.im); }
inline cx cx_mul(cx a, cx b) { return cx_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
inline cx cx_div(cx a, cx b) {
    double md = b.re * b.re + b.im * b.im;
    return cx_make((a.re * b.re + a.im * b.im) / md, (a.im * b.re - a.re * b.im) / md);
}
inline cx cx_scale(cx z, double a) { return cx_make(z.re * a, z.im * a); }
inline cx cx_neg(cx z) { return cx_make(-z.re, -z.im); }
inline int cx_eq(cx a, cx b) { return a.re == b.re && a.im == b.im; }
inline double cx_abs2(cx z) { return z.re * z.re + z.im * z.im; }
inline double cx_abs(cx z) { return sqrt(z.re * z.re + z.im * z.im); }
inline cx cx_conj(cx z) { return cx_make(z.re, -z.im); }
inline cx cx_sqrt(cx z) {
    double md = sqrt(z.re * z.re + z.im * z.im);
    if (md <= 0) return cx_make(0, 0);
    return cx_make(sqrt((md + z.re) / 2), sqrt((md - z.re) / 2) * ((z.im > 0) ? 1.0 : -1.0));
}
inline cx cx_exp(cx z) {
    double r = exp(z.re);
    return cx_make(r * cos(z.im), r * sin(z.im));
}

#define KAMPYLOS_CL_MAXIT 2000

inline double kgpu_abs2poly(const cx* poly, int degree, cx z) {
    cx pv = poly[degree];
    for (int k = degree - 1; k >= 0; k--) pv = cx_add(poly[k], cx_mul(z, pv));
    return cx_mul(cx_conj(pv), pv).re;
}

// Plain Laguerre's method for one root -- same simplification as the CUDA version (every root
// via plain Laguerre with deflation, not the full Laguerre->SG->Newton hybrid
// cmplx_roots_gen() uses): fewer code paths to get wrong on a port, same guaranteed-ish
// convergence, see kampylos_gpu_mag.cuh's own comment for the full rationale.
inline void kgpu_cmplx_laguerre(const cx* poly, int degree, cx* root) {
    const double FRAC_JUMPS[10] = { 0.64109297, 0.91577881, 0.25921289, 0.50487203,
        0.08177045, 0.13653241, 0.306162, 0.37794326, 0.04618805, 0.75132137 };
    const double FRAC_ERR = 2.0e-15;
    const double two_pi = 6.283185307179586;
    cx c_one = cx_make(1, 0), zero = cx_make(0, 0);

    double one_nth = 1.0 / degree;
    double n_1_nth = (degree - 1.0) * one_nth;
    double two_n_div_n_1 = 2.0 / n_1_nth;
    cx c_one_nth = cx_make(one_nth, 0.0);

    for (int i = 1; i <= KAMPYLOS_CL_MAXIT; i++) {
        double ek = cx_abs(poly[degree]);
        double absroot = cx_abs(*root);
        cx p = poly[degree], dp = zero, d2p_half = zero;
        for (int k = degree - 1; k >= 0; k--) {
            d2p_half = cx_add(dp, cx_mul(d2p_half, *root));
            dp = cx_add(p, cx_mul(dp, *root));
            p = cx_add(poly[k], cx_mul(p, *root));
            ek = absroot * ek + cx_abs(p);
        }

        double abs2p = cx_mul(cx_conj(p), p).re;
        if (abs2p == 0) return;
        double stopping_crit2 = (FRAC_ERR * ek) * (FRAC_ERR * ek);
        int good_to_go = 0;
        if (abs2p < stopping_crit2) {
            if (abs2p < 0.01 * stopping_crit2) return;
            good_to_go = 1;
        }

        cx denom = zero, dx;
        if (!cx_eq(dp, zero)) {
            cx fac_newton = cx_div(p, dp);
            cx fac_extra = cx_div(d2p_half, dp);
            cx F_half = cx_mul(fac_newton, fac_extra);
            cx denom_sqrt = cx_sqrt(cx_sub(c_one, cx_scale(F_half, two_n_div_n_1)));
            denom = cx_add(c_one_nth, cx_scale(denom_sqrt, n_1_nth));
            if (!cx_eq(denom, zero)) dx = cx_div(fac_newton, denom);
        }
        if (cx_eq(denom, zero)) {
            dx = cx_scale(cx_exp(cx_make(0.0, FRAC_JUMPS[i % 10] * two_pi)), absroot + 1.0);
        }

        cx newroot = cx_sub(*root, dx);
        if (cx_eq(newroot, *root)) return;
        if (good_to_go) {
            if (kgpu_abs2poly(poly, degree, newroot) < abs2p) *root = newroot;
            return;
        }
        if (i % 10 == 0) {
            double faq = FRAC_JUMPS[(i / 10 - 1) % 10];
            newroot = cx_sub(*root, cx_scale(dx, faq));
        }
        *root = newroot;
    }
}

inline void kgpu_cmplx_newton_spec(const cx* poly, int degree, cx* root) {
    const double FRAC_JUMPS[10] = { 0.64109297, 0.91577881, 0.25921289, 0.50487203,
        0.08177045, 0.13653241, 0.306162, 0.37794326, 0.04618805, 0.75132137 };
    const double FRAC_ERR = 2.0e-15;
    const double two_pi = 6.283185307179586;
    cx zero = cx_make(0, 0);
    double stopping_crit2 = 0.0;
    for (int i = 1; i <= KAMPYLOS_CL_MAXIT; i++) {
        cx p = poly[degree], dp = zero;
        if (i % 10 == 1) {
            double ek = cx_abs(poly[degree]);
            double absroot = cx_abs(*root);
            for (int k = degree - 1; k >= 0; k--) {
                dp = cx_add(p, cx_mul(dp, *root));
                p = cx_add(poly[k], cx_mul(p, *root));
                ek = absroot * ek + cx_abs(p);
            }
            stopping_crit2 = (FRAC_ERR * ek) * (FRAC_ERR * ek);
        } else {
            for (int k = degree - 1; k >= 0; k--) {
                dp = cx_add(p, cx_mul(dp, *root));
                p = cx_add(poly[k], cx_mul(p, *root));
            }
        }
        double abs2p = cx_mul(cx_conj(p), p).re;
        if (abs2p == 0.0) return;
        int good_to_go = 0;
        if (abs2p < stopping_crit2) {
            if (cx_eq(dp, zero)) return;
            if (abs2p < 0.01 * stopping_crit2) return;
            good_to_go = 1;
        }
        cx dx;
        if (cx_eq(dp, zero)) {
            dx = cx_scale(cx_exp(cx_make(0.0, FRAC_JUMPS[i % 10] * two_pi)), cx_abs(*root) + 1.0);
        } else {
            dx = cx_div(p, dp);
        }
        cx newroot = cx_sub(*root, dx);
        if (cx_eq(newroot, *root)) return;
        if (good_to_go) {
            if (kgpu_abs2poly(poly, degree, newroot) < abs2p) *root = newroot;
            return;
        }
        if (i % 10 == 0) {
            double faq = FRAC_JUMPS[(i / 10 - 1) % 10];
            newroot = cx_sub(*root, cx_scale(dx, faq));
        }
        *root = newroot;
    }
}

inline void kgpu_solve_quadratic_eq(cx* x0, cx* x1, const cx* poly) {
    cx a = poly[2], b = poly[1], c = poly[0];
    cx b2 = cx_mul(b, b);
    cx delta = cx_sqrt(cx_sub(b2, cx_scale(cx_mul(a, c), 4.0)));
    if (cx_mul(cx_conj(b), delta).re >= 0) *x0 = cx_neg(cx_scale(cx_add(b, delta), 0.5));
    else *x0 = cx_neg(cx_scale(cx_sub(b, delta), 0.5));
    if (cx_eq(*x0, cx_make(0, 0))) {
        *x1 = cx_make(0, 0);
    } else {
        *x1 = cx_div(c, *x0);
        *x0 = cx_div(*x0, a);
    }
}

#define KAMPYLOS_CL_MAXDEG 6

inline void kgpu_roots_gen(cx* roots, const cx* poly, int degree) {
    cx poly2[KAMPYLOS_CL_MAXDEG];
    for (int j = 0; j <= degree; j++) poly2[j] = poly[j];
    for (int j = 0; j < degree; j++) roots[j] = cx_make(0, 0);

    for (int n = degree; n >= 3; n--) {
        kgpu_cmplx_laguerre(poly2, n, &roots[n - 1]);
        cx coef = poly2[n];
        for (int i = n - 1; i >= 0; i--) {
            cx prev = poly2[i];
            poly2[i] = coef;
            coef = cx_add(prev, cx_mul(roots[n - 1], coef));
        }
    }
    kgpu_solve_quadratic_eq(&roots[1], &roots[0], poly2);

    for (int n = 0; n < degree; n++) {
        kgpu_cmplx_newton_spec(poly, degree, &roots[n]);
    }
}

typedef struct {
    double mag;
    double corrquad;
    double corrquad2;
    double safedist;
    int n_images;
} BinaryMag0Result;

inline BinaryMag0Result binary_mag0_gpu(double s, double q, double y1v, double y2v) {
    BinaryMag0Result out;
    out.mag = -1.0; out.corrquad = 0.0; out.corrquad2 = 0.0; out.safedist = 10.0; out.n_images = 0;

    cx a, qc;
    if (q < 1.0) { a = cx_make(-s, 0); qc = cx_make(q, 0); }
    else { a = cx_make(s, 0); qc = cx_make(1.0 / q, 0); }
    cx m1 = cx_div(cx_make(1, 0), cx_add(cx_make(1, 0), qc));
    cx m2 = cx_mul(qc, m1);

    cx c20 = a, c21 = m1, c22 = m2;
    cx c6 = cx_mul(a, a), c7 = cx_mul(c6, a), c8 = cx_mul(m2, m2), c9 = cx_mul(c6, c8);
    cx c10 = cx_mul(a, m2), c11 = cx_mul(a, m1);

    cx y = cx_make(y1v, y2v);
    cx yfull = cx_add(y, c11);
    cx yc = cx_conj(yfull);

    cx c12 = cx_sub(c20, yc);
    cx c13 = cx_add(c20, yfull);
    cx c14 = cx_add(c13, yfull);
    cx c15 = cx_conj(c14);
    cx c16 = cx_mul(c20, yfull);
    cx c17 = cx_conj(c16);
    cx c18 = cx_conj(c12);

    cx poly[6];
    poly[0] = cx_mul(c9, yfull);
    poly[1] = cx_add(cx_neg(c9), cx_mul(c10, cx_add(c20, cx_mul(cx_sub(cx_scale(c17, 2.0), cx_add(cx_make(2,0), c6)), yfull))));
    poly[2] = cx_sub(cx_mul(c10, cx_sub(cx_add(cx_make(1,0), c16), cx_scale(cx_mul(yc, c13), 2.0))), cx_mul(cx_sub(c17, cx_make(1,0)), cx_sub(cx_mul(c16, c12), c18)));
    poly[3] = cx_sub(cx_add(cx_mul(c10, c15), cx_mul(cx_sub(cx_add(c7, cx_scale(cx_mul(cx_add(cx_make(1,0),c6), yfull), 2.0)), cx_mul(c17, c14)), yc)), cx_mul(c20, c13));
    poly[4] = cx_sub(cx_neg(c10), cx_mul(c12, cx_sub(cx_mul(yc, cx_add(c13, c20)), cx_make(1,0))));
    poly[5] = cx_mul(yc, c12);

    cx zr[5];
    kgpu_roots_gen(zr, poly, 5);

    double good[5];
    for (int i = 0; i < 5; i++) {
        cx z = zr[i];
        cx zc = cx_conj(z);
        cx ll = cx_add(cx_sub(yfull, z), cx_add(cx_div(c21, cx_sub(zc, c20)), cx_div(c22, zc)));
        good[i] = cx_abs(ll);
    }
    int order[5] = { 0, 1, 2, 3, 4 };
    for (int i = 0; i < 5; i++)
        for (int j = i + 1; j < 5; j++)
            if (good[order[j]] > good[order[i]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
    int worst1 = order[0], worst2 = order[1], worst3 = order[2];
    const double dlmin = 1.0e-4;

    double mag = 0.0;
    int n_images = 0;
    int five_roots = !(good[worst2] * dlmin > good[worst3] + 1.e-12);
    double corrquad = 0.0;

    for (int i = 0; i < 5; i++) {
        if (!five_roots && (i == worst1 || i == worst2)) continue;
        cx z = zr[i];
        cx dza = cx_sub(z, c20);
        cx za2 = cx_mul(dza, dza);
        cx zb2 = cx_mul(z, z);
        cx J1 = cx_add(cx_div(c21, za2), cx_div(c22, zb2));
        cx J1c = cx_conj(J1);
        cx dJ = cx_sub(cx_make(1, 0), cx_mul(J1, J1c));
        cx J2 = cx_neg(cx_scale(cx_add(cx_div(c21, cx_mul(za2, dza)), cx_div(c22, cx_mul(zb2, z))), 2.0));

        double dJre = dJ.re;
        if (fabs(dJre) < 1.e-9) continue;

        mag += fabs(1.0 / dJre);
        n_images++;

        cx J3 = cx_scale(cx_add(cx_div(c21, cx_mul(za2, za2)), cx_div(c22, cx_mul(zb2, zb2))), 6.0);
        double dJ2 = dJre * dJre;
        cx za2c = cx_mul(J1c, J1c);
        cx J3s = cx_mul(J3, za2c);
        double ob2 = (J2.re * J2.re + J2.im * J2.im) * (6.0 - 6.0 * dJre + dJ2);
        cx J2s = cx_mul(cx_mul(cx_mul(J2, J2), za2c), J1c);
        double cq = 0.5 * (fabs(ob2 - 6.0 * J2s.re - 2.0 * J3s.re * dJre) + 3.0 * fabs(J2s.im)) / fabs(dJre * dJ2 * dJ2);
        corrquad += cq;
    }

    double corrquad2;
    if (!five_roots) {
        int idxs[2] = { worst1, worst2 };
        double cq2 = -2.0;
        for (int k = 0; k < 2; k++) {
            int i = idxs[k];
            cx z = zr[i];
            cx dza = cx_sub(z, c20);
            cx za2 = cx_mul(dza, dza);
            cx zb2 = cx_mul(z, z);
            cx J1 = cx_add(cx_div(c21, za2), cx_div(c22, zb2));
            cx J1c = cx_conj(J1);
            cx dJ = cx_sub(cx_make(1, 0), cx_mul(J1, J1c));
            cx J2 = cx_neg(cx_scale(cx_add(cx_div(c21, cx_mul(za2, dza)), cx_div(c22, cx_mul(zb2, z))), 2.0));

            cx zaltc = cx_add(yc, cx_add(cx_div(c21, dza), cx_div(c22, z)));
            cx za2b = cx_sub(zaltc, c20);
            cx Jaltc = cx_add(cx_div(c21, cx_mul(za2b, za2b)), cx_div(c22, cx_mul(zaltc, zaltc)));
            cx Jalt = cx_conj(Jaltc);
            cx JJalt2 = cx_sub(cx_make(1, 0), cx_mul(J1c, Jalt));
            cx J3 = cx_mul(cx_mul(J2, J1c), JJalt2);
            J3 = cx_div(cx_sub(J3, cx_mul(cx_conj(J3), Jalt)), cx_scale(cx_mul(JJalt2, JJalt2), dJ.re));
            double cq = (dJ.re < -10) ? -1.0 : (J3.re * J3.re + J3.im * J3.im);
            if (k == 0) {
                cq2 = cq;
            } else {
                if (cq2 < 0 || cq < 0) cq2 = 0;
                else if (cq > cq2) cq2 = cq;
            }
        }
        corrquad2 = cq2;
    } else {
        corrquad2 = 0.0;
    }

    double safedist;
    if (qc.re < 0.01) {
        double sd = y1v + (a.re - 1.0 / a.re) * c21.re;
        safedist = sd * sd + y2v * y2v - 4.0 * sqrt(qc.re) / (a.re * a.re);
    } else {
        double sabs = fabs(a.re);
        double Rinf = sabs + 1.0 / sabs + 2.0;
        double d2 = y1v * y1v + y2v * y2v;
        safedist = 10.0;
        if (d2 > Rinf * Rinf) {
            double d = sqrt(d2);
            safedist = (d - Rinf) * (d - Rinf);
        }
    }

    out.mag = (n_images > 0) ? mag : -1.0;
    out.corrquad = corrquad;
    out.corrquad2 = corrquad2;
    out.safedist = safedist;
    out.n_images = n_images;
    return out;
}

inline int binary_mag2_fastpath_ok(const BinaryMag0Result* r, double rho, double Tol) {
    double rho2 = rho * rho;
    double cq = r->corrquad * 6.0 * (rho2 + 1.e-4 * Tol);
    double cq2 = r->corrquad2 * 256.0 * (rho2 + 1.e-8);
    return (cq < Tol) && (cq2 < 1.0) && (r->safedist > 4.0 * rho2);
}

// ---------------------------------------------------------------------------------------------
// The actual kernel. Mirrors kampylos_gpu_kernel.cu's kampylos_eval_candidates exactly -- one
// OpenCL work-GROUP per candidate (matching one CUDA thread-BLOCK per candidate), work-items
// within the group splitting the light curve's points, local-memory reduction, per-point bad-list
// instead of a whole-candidate discard.
//
// LightCurvePoint/Candidate/CandidateResult come from kampylos_gpu_types.h -- the SAME header the
// CUDA side and the host-side completion code (kampylos_gpu_complete.h) use, not a separately
// hand-copied redefinition, so the two backends' struct layouts can never silently drift apart.
// That header has no C++-only syntax (plain structs, one #define), so it's valid OpenCL C (a C99
// dialect) as-is; the host code concatenates its text before this file's own when building the
// OpenCL program (see kampylos_opencl_search.c), the same way it would be #included in C/C++.
#define KAMPYLOS_CL_MAX_BAD_POINTS KAMPYLOS_MAX_BAD_POINTS
// kampylos_gpu_types.h declares "struct LightCurvePoint { ... };" (plain C99/C++-compatible
// syntax) -- C++ treats that tag as a usable type name on its own, but strict C99 (what OpenCL C
// is based on) doesn't, so these typedefs are added locally here rather than changing the shared
// header (which needs to stay exactly as C++-usable as it already is for the CUDA/host side).
typedef struct LightCurvePoint LightCurvePoint;
typedef struct Candidate Candidate;
typedef struct CandidateResult CandidateResult;

__kernel void kampylos_eval_candidates(
    __global const LightCurvePoint* data, int n_points,
    double log_s, double log_q,
    __global const Candidate* candidates, int n_candidates,
    __global CandidateResult* results,
    __local double* sh,             // 7 * local_size doubles
    __local int* bad_count,         // 1 int
    __local int* overflow_flag      // 1 int
) {
    int cand_idx = get_group_id(0);
    if (cand_idx >= n_candidates) return;

    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    if (lid == 0) { *bad_count = 0; *overflow_flag = 0; }
    barrier(CLK_LOCAL_MEM_FENCE);

    Candidate c = candidates[cand_idx];
    double tE = exp(c.log_tE);
    double rho = exp(c.log_rho);

    if (tE < 0.01 || tE > 10000.0 || rho < 1e-6 || rho > 1.0) {
        if (lid == 0) {
            __global CandidateResult* r = &results[cand_idx];
            r->S_AA = r->S_A1 = r->S_11 = r->S_Ay = r->S_1y = r->S_yy = 0;
            r->n_bad = 0; r->overflow = 0; r->poisoned = 1;
        }
        return;
    }

    double s = exp(log_s), q = exp(log_q);
    double salpha = sin(c.alpha), calpha = cos(c.alpha);
    double tE_inv = 1.0 / tE;

    double S_AA = 0, S_A1 = 0, S_11 = 0, S_Ay = 0, S_1y = 0, S_yy = 0;
    double any_invalid = 0;

    for (int i = lid; i < n_points; i += lsize) {
        double t = data[i].t;
        double tn = (t - c.t0) * tE_inv;
        double y1 = c.u0 * salpha - tn * calpha;
        double y2 = -c.u0 * calpha - tn * salpha;

        BinaryMag0Result r = binary_mag0_gpu(s, q, y1, y2);
        double A = r.mag;
        if (!(A > 0) || !isfinite(A)) {
            any_invalid = 1.0;
            continue;
        }
        if (!binary_mag2_fastpath_ok(&r, rho, 1.e-2)) {
            int slot = atomic_inc(bad_count);
            if (slot < KAMPYLOS_CL_MAX_BAD_POINTS) {
                results[cand_idx].bad_idx[slot] = i;
            } else {
                *overflow_flag = 1;
            }
            continue;
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

    sh[0 * lsize + lid] = S_AA;
    sh[1 * lsize + lid] = S_A1;
    sh[2 * lsize + lid] = S_11;
    sh[3 * lsize + lid] = S_Ay;
    sh[4 * lsize + lid] = S_1y;
    sh[5 * lsize + lid] = S_yy;
    sh[6 * lsize + lid] = any_invalid;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            for (int k = 0; k < 7; k++) sh[k * lsize + lid] += sh[k * lsize + lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        __global CandidateResult* r = &results[cand_idx];
        if (sh[6 * lsize] > 0) {
            r->S_AA = r->S_A1 = r->S_11 = r->S_Ay = r->S_1y = r->S_yy = 0;
            r->n_bad = 0; r->overflow = 0; r->poisoned = 1;
        } else {
            r->S_AA = sh[0 * lsize]; r->S_A1 = sh[1 * lsize]; r->S_11 = sh[2 * lsize];
            r->S_Ay = sh[3 * lsize]; r->S_1y = sh[4 * lsize]; r->S_yy = sh[5 * lsize];
            r->n_bad = min(*bad_count, (int)KAMPYLOS_CL_MAX_BAD_POINTS);
            r->overflow = *overflow_flag;
            r->poisoned = 0;
        }
    }
}
