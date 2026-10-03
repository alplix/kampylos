// Measures the FULL point-source magnification + quadrupole/hexadecapole fast-path test built on
// VALIDATED double-single (float-float) arithmetic primitives (see test_ds_primitives.cpp, which
// must be run and pass first -- this file assumes those primitives are sound and only tests the
// algorithm built on top of them). Same validation methodology as test_kampylos_mag_f32.cpp
// (plain float, which failed by up to 460x) and the original double-precision CUDA/OpenCL tests:
// general random sampling + a stress pass on the hardest regime, checked against the real
// VBMicrolensing double-precision reference.
//
// Build: g++ -O2 -std=c++17 -I ../src -I ../vendor/VBMicrolensing/VBMicrolensing/lib
//   test_kampylos_mag_ds.cpp ../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp
//   -o test_kampylos_mag_ds
#include <cstdio>
#include <cmath>
#include <random>
#include "VBMicrolensingLibrary.h"

// ---------------------------------------------------------------------------------------------
// Double-single (float-float) scalar arithmetic -- validated separately in test_ds_primitives.cpp
// (worst relative error 2.1e-10 across 500k general cases, including an adversarial catastrophic-
// cancellation case). Copied here rather than shared via a header since this is still an
// exploratory measurement, not yet the production Metal port -- once this validates the full
// algorithm, these primitives move into the real kampylos_metal_fit.mm MSL source as the next step.
// ---------------------------------------------------------------------------------------------
struct ds { float hi, lo; };
inline ds dsf(float hi, float lo = 0) { return { hi, lo }; }
inline ds ds_from_double(double d) { float hi = (float)d; float lo = (float)(d - (double)hi); return { hi, lo }; }
inline double ds_to_double(ds a) { return (double)a.hi + (double)a.lo; }
inline void two_sum(float a, float b, float& hi, float& lo) { hi = a + b; float bb = hi - a; lo = (a - (hi - bb)) + (b - bb); }
inline void two_prod(float a, float b, float& hi, float& lo) { hi = a * b; lo = fmaf(a, b, -hi); }
inline ds ds_add(ds a, ds b) { float s_hi, s_lo; two_sum(a.hi, b.hi, s_hi, s_lo); s_lo += a.lo + b.lo; float r_hi, r_lo; two_sum(s_hi, s_lo, r_hi, r_lo); return { r_hi, r_lo }; }
inline ds ds_neg(ds a) { return { -a.hi, -a.lo }; }
inline ds ds_sub(ds a, ds b) { return ds_add(a, ds_neg(b)); }
inline ds ds_mul(ds a, ds b) { float p_hi, p_lo; two_prod(a.hi, b.hi, p_hi, p_lo); p_lo += a.hi * b.lo + a.lo * b.hi; float r_hi, r_lo; two_sum(p_hi, p_lo, r_hi, r_lo); return { r_hi, r_lo }; }
inline ds ds_mulf(ds a, float b) { return ds_mul(a, dsf(b)); }
inline ds ds_div(ds a, ds b) { float q1 = a.hi / b.hi; ds r = ds_sub(a, ds_mul(b, dsf(q1))); float q2 = r.hi / b.hi; float r_hi, r_lo; two_sum(q1, q2, r_hi, r_lo); return { r_hi, r_lo }; }
inline ds ds_sqrt(ds a) { if (a.hi <= 0) return dsf(0, 0); float x = sqrtf(a.hi); ds xd = dsf(x); return ds_mulf(ds_add(xd, ds_div(a, xd)), 0.5f); }
inline bool ds_eq(ds a, ds b) { return a.hi == b.hi && a.lo == b.lo; }
inline bool ds_lt(ds a, ds b) { double da = ds_to_double(a), db = ds_to_double(b); return da < db; } // comparisons: double-cast is fine, only used for branching, not precision-critical
inline double ds_abs_d(ds a) { return fabs(ds_to_double(a)); }

// ---------------------------------------------------------------------------------------------
// Complex double-single arithmetic.
// ---------------------------------------------------------------------------------------------
struct dscx { ds re, im; };
inline dscx dc(ds re, ds im) { return { re, im }; }
inline dscx dc(double re, double im = 0) { return { ds_from_double(re), ds_from_double(im) }; }
inline dscx dc_add(dscx a, dscx b) { return dc(ds_add(a.re, b.re), ds_add(a.im, b.im)); }
inline dscx dc_sub(dscx a, dscx b) { return dc(ds_sub(a.re, b.re), ds_sub(a.im, b.im)); }
inline dscx dc_mul(dscx a, dscx b) {
    return dc(ds_sub(ds_mul(a.re, b.re), ds_mul(a.im, b.im)), ds_add(ds_mul(a.re, b.im), ds_mul(a.im, b.re)));
}
inline dscx dc_div(dscx a, dscx b) {
    ds md = ds_add(ds_mul(b.re, b.re), ds_mul(b.im, b.im));
    ds re = ds_div(ds_add(ds_mul(a.re, b.re), ds_mul(a.im, b.im)), md);
    ds im = ds_div(ds_sub(ds_mul(a.im, b.re), ds_mul(a.re, b.im)), md);
    return dc(re, im);
}
inline dscx dc_scale(dscx z, double a) { return dc(ds_mul(z.re, ds_from_double(a)), ds_mul(z.im, ds_from_double(a))); }
inline dscx dc_neg(dscx z) { return dc(ds_neg(z.re), ds_neg(z.im)); }
inline bool dc_eq(dscx a, dscx b) { return ds_eq(a.re, b.re) && ds_eq(a.im, b.im); }
inline double dc_abs(dscx z) { return sqrt(ds_to_double(z.re) * ds_to_double(z.re) + ds_to_double(z.im) * ds_to_double(z.im)); }
inline dscx dc_conj(dscx z) { return dc(z.re, ds_neg(z.im)); }
inline dscx dc_sqrt(dscx z) {
    double zre = ds_to_double(z.re), zim = ds_to_double(z.im);
    double md = sqrt(zre * zre + zim * zim);
    if (md <= 0) return dc(0, 0);
    // Real/imaginary parts of the result computed in plain double (this is just the SPLIT
    // direction, not accumulated error) then promoted to ds -- matches the precision structure
    // of the original double-precision csqrt, adequate here since sqrt's own result doesn't
    // need double-single accumulation the way the root-polishing loop's iterated sums do.
    double rre = sqrt((md + zre) / 2), rim = sqrt((md - zre) / 2) * ((zim > 0) ? 1.0 : -1.0);
    return dc(rre, rim);
}
inline dscx dc_exp(dscx z) {
    double zre = ds_to_double(z.re), zim = ds_to_double(z.im);
    double r = exp(zre);
    return dc(r * cos(zim), r * sin(zim));
}

#define MAXIT_DS 2000
inline double abs2poly_ds(const dscx* poly, int degree, dscx z) {
    dscx pv = poly[degree];
    for (int k = degree - 1; k >= 0; k--) pv = dc_add(poly[k], dc_mul(z, pv));
    return ds_to_double(dc_mul(dc_conj(pv), pv).re);
}
inline void laguerre_ds(const dscx* poly, int degree, dscx* root) {
    const double FRAC_JUMPS[10] = { 0.64109297, 0.91577881, 0.25921289, 0.50487203, 0.08177045, 0.13653241, 0.306162, 0.37794326, 0.04618805, 0.75132137 };
    const double FRAC_ERR = 1.0e-9; // double-single's own measured ceiling, not double's 2e-15
    const double two_pi = 6.283185307179586;
    dscx c_one = dc(1, 0), zero = dc(0, 0);
    double one_nth = 1.0 / degree, n_1_nth = (degree - 1.0) * one_nth, two_n_div_n_1 = 2.0 / n_1_nth;
    dscx c_one_nth = dc(one_nth, 0.0);
    for (int i = 1; i <= MAXIT_DS; i++) {
        double ek = dc_abs(poly[degree]);
        double absroot = dc_abs(*root);
        dscx p = poly[degree], dp = zero, d2p_half = zero;
        for (int k = degree - 1; k >= 0; k--) {
            d2p_half = dc_add(dp, dc_mul(d2p_half, *root));
            dp = dc_add(p, dc_mul(dp, *root));
            p = dc_add(poly[k], dc_mul(p, *root));
            ek = absroot * ek + dc_abs(p);
        }
        double abs2p = ds_to_double(dc_mul(dc_conj(p), p).re);
        if (abs2p == 0) return;
        double stopping_crit2 = (FRAC_ERR * ek) * (FRAC_ERR * ek);
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
        if (dc_eq(denom, zero)) dx = dc_scale(dc_exp(dc(0.0, FRAC_JUMPS[i % 10] * two_pi)), absroot + 1.0);
        dscx newroot = dc_sub(*root, dx);
        if (dc_eq(newroot, *root)) return;
        if (good_to_go) { if (abs2poly_ds(poly, degree, newroot) < abs2p) *root = newroot; return; }
        if (i % 10 == 0) { double faq = FRAC_JUMPS[(i / 10 - 1) % 10]; newroot = dc_sub(*root, dc_scale(dx, faq)); }
        *root = newroot;
    }
}
inline void newton_spec_ds(const dscx* poly, int degree, dscx* root) {
    const double FRAC_JUMPS[10] = { 0.64109297, 0.91577881, 0.25921289, 0.50487203, 0.08177045, 0.13653241, 0.306162, 0.37794326, 0.04618805, 0.75132137 };
    const double FRAC_ERR = 1.0e-9;
    const double two_pi = 6.283185307179586;
    dscx zero = dc(0, 0);
    double stopping_crit2 = 0.0;
    for (int i = 1; i <= MAXIT_DS; i++) {
        dscx p = poly[degree], dp = zero;
        if (i % 10 == 1) {
            double ek = dc_abs(poly[degree]); double absroot = dc_abs(*root);
            for (int k = degree - 1; k >= 0; k--) { dp = dc_add(p, dc_mul(dp, *root)); p = dc_add(poly[k], dc_mul(p, *root)); ek = absroot * ek + dc_abs(p); }
            stopping_crit2 = (FRAC_ERR * ek) * (FRAC_ERR * ek);
        } else {
            for (int k = degree - 1; k >= 0; k--) { dp = dc_add(p, dc_mul(dp, *root)); p = dc_add(poly[k], dc_mul(p, *root)); }
        }
        double abs2p = ds_to_double(dc_mul(dc_conj(p), p).re);
        if (abs2p == 0.0) return;
        bool good_to_go = false;
        if (abs2p < stopping_crit2) { if (dc_eq(dp, zero)) return; if (abs2p < 0.01 * stopping_crit2) return; good_to_go = true; }
        dscx dx;
        if (dc_eq(dp, zero)) dx = dc_scale(dc_exp(dc(0.0, FRAC_JUMPS[i % 10] * two_pi)), dc_abs(*root) + 1.0);
        else dx = dc_div(p, dp);
        dscx newroot = dc_sub(*root, dx);
        if (dc_eq(newroot, *root)) return;
        if (good_to_go) { if (abs2poly_ds(poly, degree, newroot) < abs2p) *root = newroot; return; }
        if (i % 10 == 0) { double faq = FRAC_JUMPS[(i / 10 - 1) % 10]; newroot = dc_sub(*root, dc_scale(dx, faq)); }
        *root = newroot;
    }
}
inline void solve_quad_ds(dscx& x0, dscx& x1, const dscx* poly) {
    dscx a = poly[2], b = poly[1], c = poly[0];
    dscx b2 = dc_mul(b, b);
    dscx delta = dc_sqrt(dc_sub(b2, dc_scale(dc_mul(a, c), 4.0)));
    if (ds_to_double(dc_mul(dc_conj(b), delta).re) >= 0) x0 = dc_neg(dc_scale(dc_add(b, delta), 0.5));
    else x0 = dc_neg(dc_scale(dc_sub(b, delta), 0.5));
    if (dc_eq(x0, dc(0, 0))) { x1 = dc(0, 0); } else { x1 = dc_div(c, x0); x0 = dc_div(x0, a); }
}
inline void roots_gen_ds(dscx* roots, const dscx* poly, int degree) {
    dscx poly2[6];
    for (int j = 0; j <= degree; j++) poly2[j] = poly[j];
    for (int j = 0; j < degree; j++) roots[j] = dc(0, 0);
    for (int n = degree; n >= 3; n--) {
        laguerre_ds(poly2, n, &roots[n - 1]);
        dscx coef = poly2[n];
        for (int i = n - 1; i >= 0; i--) { dscx prev = poly2[i]; poly2[i] = coef; coef = dc_add(prev, dc_mul(roots[n - 1], coef)); }
    }
    solve_quad_ds(roots[1], roots[0], poly2);
    for (int n = 0; n < degree; n++) newton_spec_ds(poly, degree, &roots[n]);
}

struct Mag0ResultDS { double mag, corrquad, corrquad2, safedist; int n_images; };

inline Mag0ResultDS binary_mag0_ds(double s, double q, double y1v, double y2v) {
    Mag0ResultDS out; out.mag = -1.0; out.corrquad = 0; out.corrquad2 = 0; out.safedist = 10.0; out.n_images = 0;
    dscx a, qc;
    if (q < 1.0) { a = dc(-s, 0); qc = dc(q, 0); } else { a = dc(s, 0); qc = dc(1.0 / q, 0); }
    dscx m1 = dc_div(dc(1, 0), dc_add(dc(1, 0), qc));
    dscx m2 = dc_mul(qc, m1);
    dscx c20 = a, c21 = m1, c22 = m2;
    dscx c6 = dc_mul(a, a), c7 = dc_mul(c6, a), c8 = dc_mul(m2, m2), c9 = dc_mul(c6, c8), c10 = dc_mul(a, m2), c11 = dc_mul(a, m1);
    dscx y = dc(y1v, y2v);
    dscx yfull = dc_add(y, c11);
    dscx yc = dc_conj(yfull);
    dscx c12 = dc_sub(c20, yc), c13 = dc_add(c20, yfull), c14 = dc_add(c13, yfull), c15 = dc_conj(c14);
    dscx c16 = dc_mul(c20, yfull), c17 = dc_conj(c16), c18 = dc_conj(c12);
    dscx poly[6];
    poly[0] = dc_mul(c9, yfull);
    poly[1] = dc_add(dc_neg(c9), dc_mul(c10, dc_add(c20, dc_mul(dc_sub(dc_scale(c17, 2.0), dc_add(dc(2, 0), c6)), yfull))));
    poly[2] = dc_sub(dc_mul(c10, dc_sub(dc_add(dc(1, 0), c16), dc_scale(dc_mul(yc, c13), 2.0))), dc_mul(dc_sub(c17, dc(1, 0)), dc_sub(dc_mul(c16, c12), c18)));
    poly[3] = dc_sub(dc_add(dc_mul(c10, c15), dc_mul(dc_sub(dc_add(c7, dc_scale(dc_mul(dc_add(dc(1,0),c6), yfull), 2.0)), dc_mul(c17, c14)), yc)), dc_mul(c20, c13));
    poly[4] = dc_sub(dc_neg(c10), dc_mul(c12, dc_sub(dc_mul(yc, dc_add(c13, c20)), dc(1, 0))));
    poly[5] = dc_mul(yc, c12);
    dscx zr[5];
    roots_gen_ds(zr, poly, 5);
    double good[5];
    for (int i = 0; i < 5; i++) {
        dscx z = zr[i]; dscx zc = dc_conj(z);
        dscx ll = dc_add(dc_sub(yfull, z), dc_add(dc_div(c21, dc_sub(zc, c20)), dc_div(c22, zc)));
        good[i] = dc_abs(ll);
    }
    int order[5] = { 0, 1, 2, 3, 4 };
    for (int i = 0; i < 5; i++) for (int j = i + 1; j < 5; j++) if (good[order[j]] > good[order[i]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
    int worst1 = order[0], worst2 = order[1], worst3 = order[2];
    const double dlmin = 1.0e-4;
    double mag = 0.0; int n_images = 0;
    bool five_roots = !(good[worst2] * dlmin > good[worst3] + 1.e-12);
    double corrquad = 0.0;
    for (int i = 0; i < 5; i++) {
        if (!five_roots && (i == worst1 || i == worst2)) continue;
        dscx z = zr[i]; dscx dza = dc_sub(z, c20); dscx za2 = dc_mul(dza, dza); dscx zb2 = dc_mul(z, z);
        dscx J1 = dc_add(dc_div(c21, za2), dc_div(c22, zb2)); dscx J1c = dc_conj(J1);
        dscx dJ = dc_sub(dc(1, 0), dc_mul(J1, J1c));
        dscx J2 = dc_neg(dc_scale(dc_add(dc_div(c21, dc_mul(za2, dza)), dc_div(c22, dc_mul(zb2, z))), 2.0));
        double dJre = ds_to_double(dJ.re);
        if (fabs(dJre) < 1.e-9) continue;
        mag += fabs(1.0 / dJre); n_images++;
        dscx J3 = dc_scale(dc_add(dc_div(c21, dc_mul(za2, za2)), dc_div(c22, dc_mul(zb2, zb2))), 6.0);
        double dJ2 = dJre * dJre;
        dscx za2c = dc_mul(J1c, J1c);
        dscx J3s = dc_mul(J3, za2c);
        double J2re = ds_to_double(J2.re), J2im = ds_to_double(J2.im);
        double ob2 = (J2re * J2re + J2im * J2im) * (6.0 - 6.0 * dJre + dJ2);
        dscx J2s = dc_mul(dc_mul(J2, J2), dc_mul(za2c, J1c));
        double cq = 0.5 * (fabs(ob2 - 6.0 * ds_to_double(J2s.re) - 2.0 * ds_to_double(J3s.re) * dJre) + 3.0 * fabs(ds_to_double(J2s.im))) / fabs(dJre * dJ2 * dJ2);
        corrquad += cq;
    }
    double corrquad2;
    if (!five_roots) {
        int idxs[2] = { worst1, worst2 }; double cq2 = -2.0;
        for (int k = 0; k < 2; k++) {
            int i = idxs[k];
            dscx z = zr[i]; dscx dza = dc_sub(z, c20); dscx za2 = dc_mul(dza, dza); dscx zb2 = dc_mul(z, z);
            dscx J1 = dc_add(dc_div(c21, za2), dc_div(c22, zb2)); dscx J1c = dc_conj(J1);
            dscx dJ = dc_sub(dc(1, 0), dc_mul(J1, J1c));
            dscx J2 = dc_neg(dc_scale(dc_add(dc_div(c21, dc_mul(za2, dza)), dc_div(c22, dc_mul(zb2, z))), 2.0));
            dscx zaltc = dc_add(yc, dc_add(dc_div(c21, dza), dc_div(c22, z)));
            dscx za2b = dc_sub(zaltc, c20);
            dscx Jaltc = dc_add(dc_div(c21, dc_mul(za2b, za2b)), dc_div(c22, dc_mul(zaltc, zaltc)));
            dscx Jalt = dc_conj(Jaltc);
            dscx JJalt2 = dc_sub(dc(1, 0), dc_mul(J1c, Jalt));
            dscx J3 = dc_mul(dc_mul(J2, J1c), JJalt2);
            J3 = dc_div(dc_sub(J3, dc_mul(dc_conj(J3), Jalt)), dc_scale(dc_mul(JJalt2, JJalt2), ds_to_double(dJ.re)));
            double cq = (ds_to_double(dJ.re) < -10) ? -1.0 : (ds_to_double(J3.re) * ds_to_double(J3.re) + ds_to_double(J3.im) * ds_to_double(J3.im));
            if (k == 0) cq2 = cq; else { if (cq2 < 0 || cq < 0) cq2 = 0; else if (cq > cq2) cq2 = cq; }
        }
        corrquad2 = cq2;
    } else corrquad2 = 0.0;
    double safedist;
    double a_re = ds_to_double(a.re), qc_re = ds_to_double(qc.re), c21_re = ds_to_double(c21.re);
    if (qc_re < 0.01) {
        double sd = y1v + (a_re - 1.0 / a_re) * c21_re;
        safedist = sd * sd + y2v * y2v - 4.0 * sqrt(qc_re) / (a_re * a_re);
    } else {
        double sabs = fabs(a_re); double Rinf = sabs + 1.0 / sabs + 2.0;
        double d2 = y1v * y1v + y2v * y2v; safedist = 10.0;
        if (d2 > Rinf * Rinf) { double d = sqrt(d2); safedist = (d - Rinf) * (d - Rinf); }
    }
    out.mag = (n_images > 0) ? mag : -1.0;
    out.corrquad = corrquad; out.corrquad2 = corrquad2; out.safedist = safedist; out.n_images = n_images;
    return out;
}
inline bool fastpath_ds(const Mag0ResultDS& r, double rho, double Tol = 1.e-2) {
    double rho2 = rho * rho;
    double cq = r.corrquad * 6.0 * (rho2 + 1.e-4 * Tol);
    double cq2 = r.corrquad2 * 256.0 * (rho2 + 1.e-8);
    return (cq < Tol) && (cq2 < 1.0) && (r.safedist > 4.0 * rho2);
}

// ---------------------------------------------------------------------------------------------
int n_checked = 0, n_mismatch = 0, n_decision_unsafe = 0, n_noimage = 0;
double worst_relerr = 0;

void run_pass(const char* label, VBMicrolensing& vbm, std::mt19937& rng,
              double log_s_lo, double log_s_hi, double log_q_lo, double log_q_hi,
              double y_lo, double y_hi, double log_rho_lo, double log_rho_hi, int N,
              double mag_tol) {
    std::uniform_real_distribution<double> log_s_dist(log_s_lo, log_s_hi);
    std::uniform_real_distribution<double> log_q_dist(log_q_lo, log_q_hi);
    std::uniform_real_distribution<double> y_dist(y_lo, y_hi);
    std::uniform_real_distribution<double> log_rho_dist(log_rho_lo, log_rho_hi);
    int local_mismatch = 0, local_unsafe = 0, local_noimage = 0;
    double local_worst = 0;
    for (int i = 0; i < N; i++) {
        double s = pow(10.0, log_s_dist(rng));
        double q = pow(10.0, log_q_dist(rng));
        double y1 = y_dist(rng), y2 = y_dist(rng);
        double rho = pow(10.0, log_rho_dist(rng));

        double ref_mag0 = vbm.BinaryMag0(s, q, y1, y2);
        Mag0ResultDS mine = binary_mag0_ds(s, q, y1, y2);

        if (ref_mag0 < 0 || mine.mag < 0) { local_noimage++; continue; }

        double relerr = fabs(mine.mag - ref_mag0) / ref_mag0;
        if (relerr > local_worst) local_worst = relerr;
        if (relerr > mag_tol) {
            local_mismatch++;
            if (local_mismatch <= 5) printf("  [%s] MAG MISMATCH: s=%.4f q=%.4g y1=%.4f y2=%.4f ref=%.10f ds=%.10f relerr=%.3e\n",
                label, s, q, y1, y2, ref_mag0, mine.mag, relerr);
        }

        bool my_ok = fastpath_ds(mine, rho);
        double ref_mag2 = vbm.BinaryMag2(s, q, y1, y2, rho);
        bool ref_needed_fallback = fabs(ref_mag2 - ref_mag0) / ref_mag0 > 1e-4;
        if (my_ok && ref_needed_fallback) {
            local_unsafe++;
            if (local_unsafe <= 5) printf("  [%s] UNSAFE DECISION: s=%.4f q=%.4g y1=%.4f y2=%.4f rho=%.4g mag0=%.6f mag2=%.6f\n",
                label, s, q, y1, y2, rho, ref_mag0, ref_mag2);
        }
    }
    printf("=== %s: %d samples, noimage=%d, mag mismatches(>%.0e)=%d, worst relerr=%.3e, UNSAFE=%d ===\n",
        label, N, local_noimage, mag_tol, local_mismatch, local_worst, local_unsafe);
    n_checked += N; n_mismatch += local_mismatch; n_decision_unsafe += local_unsafe; n_noimage += local_noimage;
    if (local_worst > worst_relerr) worst_relerr = local_worst;
}

int main() {
    VBMicrolensing vbm;
    vbm.Tol = 1.e-2;
    std::mt19937 rng(12345);

    run_pass("general", vbm, rng, -1.0, 1.0, -6.0, 0.0, -2.0, 2.0, -4.0, -0.3, 20000, 1e-6);
    run_pass("resonant/planetary", vbm, rng, -0.05, 0.05, -5.0, -1.0, -0.6, 0.6, -4.0, -1.5, 50000, 1e-6);

    printf("\n=== TOTAL: %d checked, %d no-image, %d mag mismatches (relerr>1e-6), worst relerr %.3e, %d UNSAFE decisions ===\n",
        n_checked, n_noimage, n_mismatch, worst_relerr, n_decision_unsafe);
    bool pass = (n_decision_unsafe == 0) && (worst_relerr < 1e-5);
    printf("\n%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
