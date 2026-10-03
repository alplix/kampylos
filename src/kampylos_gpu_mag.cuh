#pragma once
// GPU-portable point-source binary-lens magnification, faithfully mirroring the point-source
// path of VBMicrolensing's BinaryMag0()/NewImages()/BinaryMag2() (see
// vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp) -- same polynomial
// construction, same Skowron & Gould (arXiv:1203.1034) root solver, same quadrupole/hexadecapole
// error-estimate formulas (the "_Jacobians3"/"_Jacobians4" macros there), same accept/reject
// test as BinaryMag2. Native double throughout: a quintic solve is cheap next to the thousands
// of magnification evaluations one fit needs, so there's no reason to chase FP32 + double-single
// emulation the way an earlier (abandoned) GPU attempt did -- that cost a whole session chasing
// precision bugs in the *easy* half of this problem (see opencode transcript analysis,
// 2026-10-03) without ever reaching the actually-hard half (finite-source/contour integration,
// below).
//
// What this file does NOT do: BinaryMagDark's adaptive annulus integration (the recursive,
// branch-divergent finite-source path VBMicrolensing falls back to when the point-source
// approximation isn't accurate enough -- large rho, or a source close to a caustic). That stays
// on the CPU host side, called only for the points this file's own accept/reject test flags as
// needing it -- same two-tier strategy VBMicrolensing's own BinaryMag2 already uses internally,
// just split across a GPU fast path and a CPU host fallback instead of being all on one CPU.
//
// __host__ __device__ throughout so this same translation unit can be unit-tested on the CPU
// (against the real VBMicrolensing reference, via vbm.BinaryMag0/BinaryMag2) before ever touching
// a GPU -- see tests/test_kampylos_gpu_mag.cpp. Plain nvcc/g++ build, no CUDA-specific intrinsics
// used here (those belong in the kernel that calls this per-point, once grid/thread mapping is
// designed -- this file is the single-point building block, not the kernel).
#if defined(__CUDACC__)
#define KGPU_HD __host__ __device__
#else
#define KGPU_HD
#endif

#include <cmath>

namespace kgpu {

// ---------------------------------------------------------------------------------------------
// complex: same layout and semantics as VBMicrolensing's own `complex` (lib/VBMicrolensingLibrary.h),
// transcribed rather than reused because that header pulls in VBMicrolensing's full class
// definition (std::vector, <random>, host-only members) which doesn't compile under nvcc for
// device code.
// ---------------------------------------------------------------------------------------------
struct cx {
    double re, im;
    KGPU_HD cx() : re(0), im(0) {}
    KGPU_HD cx(double a) : re(a), im(0) {}
    KGPU_HD cx(double a, double b) : re(a), im(b) {}
};

KGPU_HD inline cx operator+(cx a, cx b) { return cx(a.re + b.re, a.im + b.im); }
KGPU_HD inline cx operator-(cx a, cx b) { return cx(a.re - b.re, a.im - b.im); }
KGPU_HD inline cx operator*(cx a, cx b) { return cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
KGPU_HD inline cx operator/(cx a, cx b) {
    double md = b.re * b.re + b.im * b.im;
    return cx((a.re * b.re + a.im * b.im) / md, (a.im * b.re - a.re * b.im) / md);
}
KGPU_HD inline cx operator+(cx z, double a) { return cx(z.re + a, z.im); }
KGPU_HD inline cx operator-(cx z, double a) { return cx(z.re - a, z.im); }
KGPU_HD inline cx operator*(cx z, double a) { return cx(z.re * a, z.im * a); }
KGPU_HD inline cx operator/(cx z, double a) { return cx(z.re / a, z.im / a); }
KGPU_HD inline cx operator+(double a, cx z) { return cx(z.re + a, z.im); }
KGPU_HD inline cx operator-(double a, cx z) { return cx(a - z.re, -z.im); }
KGPU_HD inline cx operator*(double a, cx z) { return cx(a * z.re, a * z.im); }
KGPU_HD inline cx operator-(cx z) { return cx(-z.re, -z.im); }
KGPU_HD inline bool operator==(cx a, cx b) { return a.re == b.re && a.im == b.im; }
KGPU_HD inline bool operator!=(cx a, cx b) { return !(a == b); }

KGPU_HD inline double re(cx z) { return z.re; }
KGPU_HD inline double im(cx z) { return z.im; }
KGPU_HD inline double abs2(cx z) { return z.re * z.re + z.im * z.im; }
KGPU_HD inline double cabs(cx z) { return sqrt(z.re * z.re + z.im * z.im); }
KGPU_HD inline cx conjg(cx z) { return cx(z.re, -z.im); }
KGPU_HD inline cx csqrt(cx z) {
    double md = sqrt(z.re * z.re + z.im * z.im);
    if (md <= 0) return cx(0, 0);
    return cx(sqrt((md + z.re) / 2), sqrt((md - z.re) / 2) * ((z.im > 0) ? 1.0 : -1.0));
}
KGPU_HD inline cx cexp(cx z) {
    double r = exp(z.re);
    return cx(r * cos(z.im), r * sin(z.im));
}

// ---------------------------------------------------------------------------------------------
// Degree-5 complex polynomial root solver (Skowron & Gould 2012). This is a faithful-in-spirit,
// deliberately SIMPLER port of VBMicrolensing's cmplx_roots_gen(): that function tries the fast
// 3-mode hybrid (cmplx_laguerre2newton: Laguerre -> SG -> Newton, switching on a convergence
// indicator) and falls back to plain Laguerre only on failure. Here, every root goes straight
// through plain Laguerre (cmplx_laguerre below, transcribed verbatim from VBMicrolensingLibrary.cpp)
// -- same method, same guaranteed-ish convergence properties, just without the "try the faster
// path first" optimization. That's a deliberate correctness-first simplification for this first
// GPU version: fewer code paths to get subtly wrong on a first port. Switching to the full
// 3-mode hybrid for the throughput win is a tracked follow-up once this is validated, not a
// capability gap -- the actual roots returned are the same roots, found a bit less efficiently.
//
// Newton polish (cmplx_newton_spec) IS ported, since cmplx_roots_gen always runs it as a final
// pass and it's short/simple.
// ---------------------------------------------------------------------------------------------

#define KGPU_MAXIT 2000
#define KGPU_MAXDEG 6

KGPU_HD inline double kgpu_abs2poly(const cx* poly, int degree, cx z) {
    cx pv = poly[degree];
    for (int k = degree - 1; k >= 0; k--) pv = poly[k] + z * pv;
    return re(conjg(pv) * pv);
}

// Plain Laguerre's method for one root, starting from *root (0,0 if no better guess).
// Transcribed from VBMicrolensing::cmplx_laguerre.
KGPU_HD inline void kgpu_cmplx_laguerre(const cx* poly, int degree, cx* root, int& iter, bool& success) {
    const double FRAC_JUMPS[10] = { 0.64109297, 0.91577881, 0.25921289, 0.50487203,
        0.08177045, 0.13653241, 0.306162, 0.37794326, 0.04618805, 0.75132137 };
    const double FRAC_ERR = 2.0e-15;
    const double two_pi = 6.283185307179586;
    cx c_one(1, 0), zero(0, 0);

    double one_nth = 1.0 / degree;
    double n_1_nth = (degree - 1.0) * one_nth;
    double two_n_div_n_1 = 2.0 / n_1_nth;
    cx c_one_nth(one_nth, 0.0);

    success = true;
    for (int i = 1; i <= KGPU_MAXIT; i++) {
        double ek = cabs(poly[degree]);
        double absroot = cabs(*root);
        cx p = poly[degree], dp = zero, d2p_half = zero;
        for (int k = degree - 1; k >= 0; k--) {
            d2p_half = dp + d2p_half * (*root);
            dp = p + dp * (*root);
            p = poly[k] + p * (*root);
            ek = absroot * ek + cabs(p);
        }
        iter += 1;

        double abs2p = re(conjg(p) * p);
        if (abs2p == 0) return;
        double stopping_crit2 = (FRAC_ERR * ek) * (FRAC_ERR * ek);
        bool good_to_go;
        if (abs2p < stopping_crit2) {
            if (abs2p < 0.01 * stopping_crit2) return;
            good_to_go = true;
        } else {
            good_to_go = false;
        }

        double faq = 1.0;
        cx denom = zero, dx;
        if (!(dp == zero)) {
            cx fac_newton = p / dp;
            cx fac_extra = d2p_half / dp;
            cx F_half = fac_newton * fac_extra;
            cx denom_sqrt = csqrt(c_one - two_n_div_n_1 * F_half);
            denom = c_one_nth + n_1_nth * denom_sqrt;
            if (!(denom == zero)) dx = fac_newton / denom;
        }
        if (denom == zero) {
            dx = (absroot + 1.0) * cexp(cx(0.0, FRAC_JUMPS[i % 10] * two_pi));
        }

        cx newroot = *root - dx;
        if (newroot == *root) return;
        if (good_to_go) {
            if (kgpu_abs2poly(poly, degree, newroot) < abs2p) *root = newroot;
            return;
        }
        if (i % 10 == 0) {
            faq = FRAC_JUMPS[(i / 10 - 1) % 10];
            newroot = *root - faq * dx;
        }
        *root = newroot;
    }
    success = false;
}

// Newton polish, transcribed from VBMicrolensing::cmplx_newton_spec.
KGPU_HD inline void kgpu_cmplx_newton_spec(const cx* poly, int degree, cx* root, int& iter, bool& success) {
    const double FRAC_JUMPS[10] = { 0.64109297, 0.91577881, 0.25921289, 0.50487203,
        0.08177045, 0.13653241, 0.306162, 0.37794326, 0.04618805, 0.75132137 };
    const double FRAC_ERR = 2.0e-15;
    const double two_pi = 6.283185307179586;
    cx zero(0, 0);
    iter = 0;
    success = true;
    double stopping_crit2 = 0.0;
    for (int i = 1; i <= KGPU_MAXIT; i++) {
        cx p = poly[degree], dp = zero;
        double ek = 0, absroot = 0;
        if (i % 10 == 1) {
            ek = cabs(poly[degree]);
            absroot = cabs(*root);
            for (int k = degree - 1; k >= 0; k--) {
                dp = p + dp * (*root);
                p = poly[k] + p * (*root);
                ek = absroot * ek + cabs(p);
            }
            stopping_crit2 = (FRAC_ERR * ek) * (FRAC_ERR * ek);
        } else {
            for (int k = degree - 1; k >= 0; k--) {
                dp = p + dp * (*root);
                p = poly[k] + p * (*root);
            }
        }
        iter += 1;
        double abs2p = re(conjg(p) * p);
        if (abs2p == 0.0) return;
        bool good_to_go;
        if (abs2p < stopping_crit2) {
            if (dp == zero) return;
            if (abs2p < 0.01 * stopping_crit2) return;
            good_to_go = true;
        } else {
            good_to_go = false;
        }
        cx dx;
        if (dp == zero) {
            dx = (cabs(*root) + 1.0) * cexp(cx(0.0, FRAC_JUMPS[i % 10] * two_pi));
        } else {
            dx = p / dp;
        }
        cx newroot = *root - dx;
        if (newroot == *root) return;
        if (good_to_go) {
            if (kgpu_abs2poly(poly, degree, newroot) < abs2p) *root = newroot;
            return;
        }
        if (i % 10 == 0) {
            double faq = FRAC_JUMPS[(i / 10 - 1) % 10];
            newroot = *root - faq * dx;
        }
        *root = newroot;
    }
    success = false;
}

// Closed-form quadratic (Viete's formula), transcribed from VBMicrolensing::solve_quadratic_eq.
KGPU_HD inline void kgpu_solve_quadratic_eq(cx& x0, cx& x1, const cx* poly) {
    cx a = poly[2], b = poly[1], c = poly[0];
    cx b2 = b * b;
    cx delta = csqrt(b2 - 4.0 * a * c);
    if (re(conjg(b) * delta) >= 0) x0 = -0.5 * (b + delta);
    else x0 = -0.5 * (b - delta);
    if (x0 == cx(0, 0)) {
        x1 = cx(0, 0);
    } else {
        x1 = c / x0;
        x0 = x0 / a;
    }
}

// Finds all `degree` roots of poly[0..degree] (poly[degree] is the leading coefficient),
// via sequential deflation -- same structure as VBMicrolensing::cmplx_roots_gen, Laguerre instead
// of the 3-mode hybrid (see comment above), always with Newton polishing against the ORIGINAL
// (undeflated) polynomial at the end, same as the reference's polish_roots_after=true path
// (which is what NewImages always requests).
KGPU_HD inline void kgpu_roots_gen(cx* roots, const cx* poly, int degree) {
    cx poly2[KGPU_MAXDEG];
    for (int j = 0; j <= degree; j++) poly2[j] = poly[j];
    for (int j = 0; j < degree; j++) roots[j] = cx(0, 0);

    for (int n = degree; n >= 3; n--) {
        int iter = 0;
        bool success = true;
        kgpu_cmplx_laguerre(poly2, n, &roots[n - 1], iter, success);
        // deflate: divide poly2 by (x - roots[n-1])
        cx coef = poly2[n];
        for (int i = n - 1; i >= 0; i--) {
            cx prev = poly2[i];
            poly2[i] = coef;
            coef = prev + roots[n - 1] * coef;
        }
    }
    kgpu_solve_quadratic_eq(roots[1], roots[0], poly2);

    for (int n = 0; n < degree; n++) {
        int iter = 0;
        bool success = true;
        kgpu_cmplx_newton_spec(poly, degree, &roots[n], iter, success);
    }
}

// ---------------------------------------------------------------------------------------------
// Point-source binary-lens magnification + quadrupole/hexadecapole error estimate.
//
// Mirrors VBMicrolensing::BinaryMag0() + the point-source (theta->th<0) branch of NewImages():
// build the degree-5 lens-equation polynomial for source position (y1,y2) with separation s and
// mass ratio q, find its roots, keep the "good" ones (those that actually solve the lens
// equation, since a quintic has 5 roots but only 3 or 5 are true images depending on whether the
// source is inside or outside the central/secondary caustics), sum |1/dJ| over true images for
// the point-source magnification, and compute corrquad/corrquad2 (the same analytic
// quadrupole/hexadecapole correction-SIZE estimates the reference uses, via the same algebra as
// its "_Jacobians3"/"_Jacobians4" macros) as a cheap, closed-form proxy for "how wrong is the
// point-source approximation here".
//
// s, q here are the RAW (a1, q1) inputs, unconstrained -- this replicates BinaryMag0's own
// internal q1<1 vs q1>=1 branch (q.re is always <=1 after it; a.re is NEGATED when the original
// q1<1, which is the common case). Skipping that branch (an earlier version of this file did,
// by mistake) silently gets the sign of `a` wrong for every q<1 case and produces magnifications
// that are wrong by anywhere from 1e-6 to O(1) relative error depending on how close the source
// is to a caustic -- caught by this file's own validation harness (tests/test_kampylos_gpu_mag.cpp)
// against the real VBMicrolensing reference, not by inspection, so don't trust this sign
// convention from reading the code alone if it's ever touched again -- re-run that test.
struct BinaryMag0Result {
    double mag;       // point-source magnification (sum |1/dJ| over true images); -1 if no images found (shouldn't happen for a real source position)
    double corrquad;   // quadrupole correction-size estimate (same units/scaling as VBMicrolensing's own corrquad right after BinaryMag0 -- NOT yet multiplied by the rho-dependent factor BinaryMag2 applies before testing it)
    double corrquad2;  // hexadecapole correction-size estimate, same caveat
    double safedist;   // squared distance-based safety margin used in the accept/reject test (see binary_mag2_fastpath)
    int n_images;
};

KGPU_HD inline BinaryMag0Result binary_mag0_gpu(double s, double q, double y1v, double y2v) {
    BinaryMag0Result out;
    out.mag = -1.0;
    out.corrquad = 0.0;
    out.corrquad2 = 0.0;
    out.safedist = 10.0;
    out.n_images = 0;

    cx a, qc;
    if (q < 1.0) { a = cx(-s, 0); qc = cx(q, 0); }
    else { a = cx(s, 0); qc = cx(1.0 / q, 0); }
    cx m1 = 1.0 / (1.0 + qc);
    cx m2 = qc * m1;

    // coefs[20..22] = a, m1, m2; coefs[6..11] = powers/products used below, matching
    // VBMicrolensing's own coefs[] layout exactly (NewImages() reuses these indices).
    cx c20 = a, c21 = m1, c22 = m2;
    cx c6 = a * a, c7 = c6 * a, c8 = m2 * m2, c9 = c6 * c8, c10 = a * m2, c11 = a * m1;

    cx y(y1v, y2v);
    cx yfull = y + c11;
    cx yc = conjg(yfull);

    cx c12 = c20 - yc;
    cx c13 = c20 + yfull;
    cx c14 = c13 + yfull;
    cx c15 = conjg(c14);
    cx c16 = c20 * yfull;
    cx c17 = conjg(c16);
    cx c18 = conjg(c12);

    cx poly[6];
    poly[0] = c9 * yfull;
    poly[1] = -c9 + c10 * (c20 + (2.0 * c17 - 2.0 - c6) * yfull);
    poly[2] = c10 * (1.0 + c16 - 2.0 * yc * c13) - (c17 - 1.0) * (c16 * c12 - c18);
    poly[3] = c10 * c15 + (c7 + 2.0 * (1.0 + c6) * yfull - c17 * c14) * yc - c20 * c13;
    poly[4] = -c10 - c12 * (yc * (c13 + c20) - 1.0);
    poly[5] = yc * c12;

    cx zr[5];
    kgpu_roots_gen(zr, poly, 5);

    // Check which roots actually solve the lens equation (same _LL test as the reference:
    // (y - z) + m1/(conj(z)-a) + m2/conj(z) should be ~0 for a true image).
    double good[5];
    for (int i = 0; i < 5; i++) {
        cx z = zr[i];
        cx zc = conjg(z);
        cx ll = (yfull - z) + c21 / (zc - c20) + c22 / zc;
        good[i] = cabs(ll);
    }
    // Rank by goodness (ascending error) to find the 2 worst -- same logic as the reference's
    // running worst1/worst2/worst3 tracker, just done as an explicit sort since there's no
    // benefit to the incremental version here.
    int order[5] = { 0, 1, 2, 3, 4 };
    for (int i = 0; i < 5; i++)
        for (int j = i + 1; j < 5; j++)
            if (good[order[j]] > good[order[i]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
    int worst1 = order[0], worst2 = order[1], worst3 = order[2];
    const double dlmin = 1.0e-4;

    double mag = 0.0;
    double corrquad = 0.0, corrquad2 = 1e200;
    int n_images = 0;
    bool five_roots = !(good[worst2] * dlmin > good[worst3] + 1.e-12);

    for (int i = 0; i < 5; i++) {
        if (!five_roots && (i == worst1 || i == worst2)) continue; // only 3 true images
        cx z = zr[i];
        cx dza = z - c20;
        cx za2 = dza * dza;
        cx zb2 = z * z;
        cx J1 = c21 / za2 + c22 / zb2;
        cx J1c = conjg(J1);
        cx dJ = 1.0 - J1 * J1c;
        cx J2 = -2.0 * (c21 / (za2 * dza) + c22 / (zb2 * z));

        double dJre = dJ.re;
        if (fabs(dJre) < 1.e-9) continue; // degenerate image, same practical exclusion as checkJac in the reference

        mag += fabs(1.0 / dJre);
        n_images++;

        // _Jacobians3: per-image quadrupole-size contribution.
        cx J3 = 6.0 * (c21 / (za2 * za2) + c22 / (zb2 * zb2));
        double dJ2 = dJre * dJre;
        cx za2c = J1c * J1c;
        cx J3s = J3 * za2c;
        double ob2 = (J2.re * J2.re + J2.im * J2.im) * (6.0 - 6.0 * dJre + dJ2);
        cx J2s = J2 * J2 * za2c * J1c;
        double cq = 0.5 * (fabs(ob2 - 6.0 * J2s.re - 2.0 * J3s.re * dJre) + 3.0 * fabs(J2s.im)) / fabs(dJre * dJ2 * dJ2);
        corrquad += cq;
    }

    // _Jacobians4-based corrquad2 (hexadecapole-size estimate from the two worst/closest-image
    // pair) only applies in the 3-true-image case, same as the reference (theta->th<0 branch
    // there always has worst1/worst2 as the pair to use -- for 5 true images the reference
    // leaves corrquad2 at its BinaryMag0-entry value of 0, so this does the same).
    if (!five_roots) {
        cx zalt_terms[2];
        int idxs[2] = { worst1, worst2 };
        double cq2 = -2.0; // sentinel "not yet set"
        for (int k = 0; k < 2; k++) {
            int i = idxs[k];
            cx z = zr[i];
            cx dza = z - c20;
            cx za2 = dza * dza;
            cx zb2 = z * z;
            cx J1 = c21 / za2 + c22 / zb2;
            cx J1c = conjg(J1);
            cx dJ = 1.0 - J1 * J1c;
            cx J2 = -2.0 * (c21 / (za2 * dza) + c22 / (zb2 * z));

            cx zaltc = yc + c21 / dza + c22 / z;
            cx za2b = zaltc - c20;
            cx Jaltc = c21 / (za2b * za2b) + c22 / (zaltc * zaltc);
            cx Jalt = conjg(Jaltc);
            cx JJalt2 = 1.0 - J1c * Jalt;
            cx J3 = J2 * J1c * JJalt2;
            J3 = (J3 - conjg(J3) * Jalt) / (JJalt2 * JJalt2 * dJ.re);
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

    // safedist, transcribed from BinaryMag0's own computation (the q.re<0.01 vs q.re>=0.01
    // branches) -- q.re and a.re here are the POST-BRANCH values (qc, a above), not the raw s,q
    // this function was called with.
    double safedist = 10.0;
    if (qc.re < 0.01) {
        double sd = y1v + (a.re - 1.0 / a.re) * c21.re;
        safedist = sd * sd + y2v * y2v - 4.0 * sqrt(qc.re) / (a.re * a.re);
    } else {
        double sabs = fabs(a.re);
        double Rinf = sabs + 1.0 / sabs + 2.0;
        double d2 = y1v * y1v + y2v * y2v;
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

// Mirrors BinaryMag2's accept/reject test exactly (same Tol=1e-2 VBMicrolensing defaults to,
// same scaling of corrquad/corrquad2 by rho before testing). Returns true if the point-source
// magnification (r.mag) is accurate enough to use directly; false means this point needs the
// CPU-side BinaryMagDark fallback (finite-source/contour integration) -- caller's responsibility
// to route it there, this function only classifies.
KGPU_HD inline bool binary_mag2_fastpath_ok(const BinaryMag0Result& r, double rho, double Tol = 1.e-2) {
    double rho2 = rho * rho;
    double cq = r.corrquad * 6.0 * (rho2 + 1.e-4 * Tol);
    double cq2 = r.corrquad2 * 256.0 * (rho2 + 1.e-8);
    return (cq < Tol) && (cq2 < 1.0) && (r.safedist > 4.0 * rho2);
}

} // namespace kgpu
