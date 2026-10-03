// Measures how much accuracy plain float32 (not double-single emulation -- just native float,
// per Alp's explicit choice after Metal turned out not to support `double` at all, a hardware
// limitation not a syntax issue) actually costs for Kampylos's point-source magnification +
// quadrupole/hexadecapole fast-path test. Same validation methodology as
// test_kampylos_gpu_mag.cpp (the double-precision CUDA/OpenCL version): general random sampling
// across the full parameter space, plus a stress pass concentrated on the hardest regime
// (resonant topology, planetary mass ratios, source near the lens). This is a standalone,
// Metal-independent measurement -- the float math here is plain portable C++, so this answers
// "is float32 good enough" on this machine's CPU, before committing to porting the actual Metal
// kernel.
//
// Build: g++ -O2 -std=c++17 -I ../src -I ../vendor/VBMicrolensing/VBMicrolensing/lib
//   test_kampylos_mag_f32.cpp ../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp
//   -o test_kampylos_mag_f32
#include <cstdio>
#include <cmath>
#include <random>
#include "VBMicrolensingLibrary.h"

// ---------------------------------------------------------------------------------------------
// Float32 re-implementation of kampylos_gpu_mag.cuh's complex arithmetic + quintic solver +
// binary_mag0_gpu/binary_mag2_fastpath_ok -- same algorithm, same structure, every `double`
// changed to `float`, FRAC_ERR/convergence thresholds rescaled for float epsilon (~1.2e-7)
// instead of double epsilon (~2.2e-16).
// ---------------------------------------------------------------------------------------------
struct cxf { float re, im; };
inline cxf cf(float a, float b = 0) { cxf z; z.re = a; z.im = b; return z; }
inline cxf operator+(cxf a, cxf b) { return cf(a.re + b.re, a.im + b.im); }
inline cxf operator-(cxf a, cxf b) { return cf(a.re - b.re, a.im - b.im); }
inline cxf operator*(cxf a, cxf b) { return cf(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
inline cxf operator/(cxf a, cxf b) {
    float md = b.re * b.re + b.im * b.im;
    return cf((a.re * b.re + a.im * b.im) / md, (a.im * b.re - a.re * b.im) / md);
}
inline cxf operator*(cxf z, float a) { return cf(z.re * a, z.im * a); }
inline cxf operator-(cxf z) { return cf(-z.re, -z.im); }
inline bool cxeq(cxf a, cxf b) { return a.re == b.re && a.im == b.im; }
inline float cabsf_(cxf z) { return sqrtf(z.re * z.re + z.im * z.im); }
inline cxf conjf_(cxf z) { return cf(z.re, -z.im); }
inline cxf csqrtf_(cxf z) {
    float md = sqrtf(z.re * z.re + z.im * z.im);
    if (md <= 0) return cf(0, 0);
    return cf(sqrtf((md + z.re) / 2), sqrtf((md - z.re) / 2) * ((z.im > 0) ? 1.0f : -1.0f));
}
inline cxf cexpf_(cxf z) {
    float r = expf(z.re);
    return cf(r * cosf(z.im), r * sinf(z.im));
}

#define MAXIT_F 2000
inline float abs2polyf(const cxf* poly, int degree, cxf z) {
    cxf pv = poly[degree];
    for (int k = degree - 1; k >= 0; k--) pv = poly[k] + z * pv;
    return (conjf_(pv) * pv).re;
}
inline void laguerref(const cxf* poly, int degree, cxf* root) {
    const float FRAC_JUMPS[10] = { 0.64109297f, 0.91577881f, 0.25921289f, 0.50487203f,
        0.08177045f, 0.13653241f, 0.306162f, 0.37794326f, 0.04618805f, 0.75132137f };
    const float FRAC_ERR = 2.0e-6f; // rescaled for float epsilon (double version used 2e-15)
    const float two_pi = 6.283185307f;
    cxf c_one = cf(1, 0), zero = cf(0, 0);
    float one_nth = 1.0f / degree;
    float n_1_nth = (degree - 1.0f) * one_nth;
    float two_n_div_n_1 = 2.0f / n_1_nth;
    cxf c_one_nth = cf(one_nth, 0.0f);
    for (int i = 1; i <= MAXIT_F; i++) {
        float ek = cabsf_(poly[degree]);
        float absroot = cabsf_(*root);
        cxf p = poly[degree], dp = zero, d2p_half = zero;
        for (int k = degree - 1; k >= 0; k--) {
            d2p_half = dp + d2p_half * (*root);
            dp = p + dp * (*root);
            p = poly[k] + p * (*root);
            ek = absroot * ek + cabsf_(p);
        }
        float abs2p = (conjf_(p) * p).re;
        if (abs2p == 0) return;
        float stopping_crit2 = (FRAC_ERR * ek) * (FRAC_ERR * ek);
        bool good_to_go = false;
        if (abs2p < stopping_crit2) {
            if (abs2p < 0.01f * stopping_crit2) return;
            good_to_go = true;
        }
        cxf denom = zero, dx;
        if (!cxeq(dp, zero)) {
            cxf fac_newton = p / dp;
            cxf fac_extra = d2p_half / dp;
            cxf F_half = fac_newton * fac_extra;
            cxf denom_sqrt = csqrtf_(c_one - F_half * two_n_div_n_1);
            denom = c_one_nth + denom_sqrt * n_1_nth;
            if (!cxeq(denom, zero)) dx = fac_newton / denom;
        }
        if (cxeq(denom, zero)) dx = cexpf_(cf(0.0f, FRAC_JUMPS[i % 10] * two_pi)) * (absroot + 1.0f);
        cxf newroot = *root - dx;
        if (cxeq(newroot, *root)) return;
        if (good_to_go) { if (abs2polyf(poly, degree, newroot) < abs2p) *root = newroot; return; }
        if (i % 10 == 0) { float faq = FRAC_JUMPS[(i / 10 - 1) % 10]; newroot = *root - dx * faq; }
        *root = newroot;
    }
}
inline void newtonspecf(const cxf* poly, int degree, cxf* root) {
    const float FRAC_JUMPS[10] = { 0.64109297f, 0.91577881f, 0.25921289f, 0.50487203f,
        0.08177045f, 0.13653241f, 0.306162f, 0.37794326f, 0.04618805f, 0.75132137f };
    const float FRAC_ERR = 2.0e-6f;
    const float two_pi = 6.283185307f;
    cxf zero = cf(0, 0);
    float stopping_crit2 = 0.0f;
    for (int i = 1; i <= MAXIT_F; i++) {
        cxf p = poly[degree], dp = zero;
        if (i % 10 == 1) {
            float ek = cabsf_(poly[degree]);
            float absroot = cabsf_(*root);
            for (int k = degree - 1; k >= 0; k--) { dp = p + dp * (*root); p = poly[k] + p * (*root); ek = absroot * ek + cabsf_(p); }
            stopping_crit2 = (FRAC_ERR * ek) * (FRAC_ERR * ek);
        } else {
            for (int k = degree - 1; k >= 0; k--) { dp = p + dp * (*root); p = poly[k] + p * (*root); }
        }
        float abs2p = (conjf_(p) * p).re;
        if (abs2p == 0.0f) return;
        bool good_to_go = false;
        if (abs2p < stopping_crit2) {
            if (cxeq(dp, zero)) return;
            if (abs2p < 0.01f * stopping_crit2) return;
            good_to_go = true;
        }
        cxf dx;
        if (cxeq(dp, zero)) dx = cexpf_(cf(0.0f, FRAC_JUMPS[i % 10] * two_pi)) * (cabsf_(*root) + 1.0f);
        else dx = p / dp;
        cxf newroot = *root - dx;
        if (cxeq(newroot, *root)) return;
        if (good_to_go) { if (abs2polyf(poly, degree, newroot) < abs2p) *root = newroot; return; }
        if (i % 10 == 0) { float faq = FRAC_JUMPS[(i / 10 - 1) % 10]; newroot = *root - dx * faq; }
        *root = newroot;
    }
}
inline void solvequadf(cxf& x0, cxf& x1, const cxf* poly) {
    cxf a = poly[2], b = poly[1], c = poly[0];
    cxf b2 = b * b;
    cxf delta = csqrtf_(b2 - a * c * 4.0f);
    if ((conjf_(b) * delta).re >= 0) x0 = -((b + delta) * 0.5f);
    else x0 = -((b - delta) * 0.5f);
    if (cxeq(x0, cf(0, 0))) { x1 = cf(0, 0); }
    else { x1 = c / x0; x0 = x0 / a; }
}
inline void rootsgenf(cxf* roots, const cxf* poly, int degree) {
    cxf poly2[6];
    for (int j = 0; j <= degree; j++) poly2[j] = poly[j];
    for (int j = 0; j < degree; j++) roots[j] = cf(0, 0);
    for (int n = degree; n >= 3; n--) {
        laguerref(poly2, n, &roots[n - 1]);
        cxf coef = poly2[n];
        for (int i = n - 1; i >= 0; i--) { cxf prev = poly2[i]; poly2[i] = coef; coef = prev + roots[n - 1] * coef; }
    }
    solvequadf(roots[1], roots[0], poly2);
    for (int n = 0; n < degree; n++) newtonspecf(poly, degree, &roots[n]);
}

struct Mag0ResultF { float mag, corrquad, corrquad2, safedist; int n_images; };

inline Mag0ResultF binary_mag0_f32(float s, float q, float y1v, float y2v) {
    Mag0ResultF out; out.mag = -1.0f; out.corrquad = 0; out.corrquad2 = 0; out.safedist = 10.0f; out.n_images = 0;
    cxf a, qc;
    if (q < 1.0f) { a = cf(-s, 0); qc = cf(q, 0); } else { a = cf(s, 0); qc = cf(1.0f / q, 0); }
    cxf m1 = cf(1, 0) / (cf(1, 0) + qc);
    cxf m2 = qc * m1;
    cxf c20 = a, c21 = m1, c22 = m2;
    cxf c6 = a * a, c7 = c6 * a, c8 = m2 * m2, c9 = c6 * c8, c10 = a * m2, c11 = a * m1;
    cxf y = cf(y1v, y2v);
    cxf yfull = y + c11;
    cxf yc = conjf_(yfull);
    cxf c12 = c20 - yc, c13 = c20 + yfull, c14 = c13 + yfull, c15 = conjf_(c14);
    cxf c16 = c20 * yfull, c17 = conjf_(c16), c18 = conjf_(c12);
    cxf poly[6];
    poly[0] = c9 * yfull;
    poly[1] = -c9 + c10 * (c20 + (c17 * 2.0f - (cf(2, 0) + c6)) * yfull);
    poly[2] = c10 * (cf(1, 0) + c16 - (yc * c13) * 2.0f) - (c17 - cf(1, 0)) * (c16 * c12 - c18);
    poly[3] = c10 * c15 + (c7 + (cf(1, 0) + c6) * yfull * 2.0f - c17 * c14) * yc - c20 * c13;
    poly[4] = -c10 - c12 * (yc * (c13 + c20) - cf(1, 0));
    poly[5] = yc * c12;
    cxf zr[5];
    rootsgenf(zr, poly, 5);
    float good[5];
    for (int i = 0; i < 5; i++) {
        cxf z = zr[i]; cxf zc = conjf_(z);
        cxf ll = (yfull - z) + (c21 / (zc - c20) + c22 / zc);
        good[i] = cabsf_(ll);
    }
    int order[5] = { 0, 1, 2, 3, 4 };
    for (int i = 0; i < 5; i++) for (int j = i + 1; j < 5; j++) if (good[order[j]] > good[order[i]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
    int worst1 = order[0], worst2 = order[1], worst3 = order[2];
    const float dlmin = 1.0e-4f;
    float mag = 0.0f; int n_images = 0;
    bool five_roots = !(good[worst2] * dlmin > good[worst3] + 1.e-12f);
    float corrquad = 0.0f;
    for (int i = 0; i < 5; i++) {
        if (!five_roots && (i == worst1 || i == worst2)) continue;
        cxf z = zr[i]; cxf dza = z - c20; cxf za2 = dza * dza; cxf zb2 = z * z;
        cxf J1 = c21 / za2 + c22 / zb2; cxf J1c = conjf_(J1);
        cxf dJ = cf(1, 0) - J1 * J1c;
        cxf J2 = -((c21 / (za2 * dza) + c22 / (zb2 * z)) * 2.0f);
        float dJre = dJ.re;
        if (fabsf(dJre) < 1.e-9f) continue;
        mag += fabsf(1.0f / dJre); n_images++;
        cxf J3 = (c21 / (za2 * za2) + c22 / (zb2 * zb2)) * 6.0f;
        float dJ2 = dJre * dJre;
        cxf za2c = J1c * J1c;
        cxf J3s = J3 * za2c;
        float ob2 = (J2.re * J2.re + J2.im * J2.im) * (6.0f - 6.0f * dJre + dJ2);
        cxf J2s = J2 * J2 * za2c * J1c;
        float cq = 0.5f * (fabsf(ob2 - 6.0f * J2s.re - 2.0f * J3s.re * dJre) + 3.0f * fabsf(J2s.im)) / fabsf(dJre * dJ2 * dJ2);
        corrquad += cq;
    }
    float corrquad2;
    if (!five_roots) {
        int idxs[2] = { worst1, worst2 }; float cq2 = -2.0f;
        for (int k = 0; k < 2; k++) {
            int i = idxs[k];
            cxf z = zr[i]; cxf dza = z - c20; cxf za2 = dza * dza; cxf zb2 = z * z;
            cxf J1 = c21 / za2 + c22 / zb2; cxf J1c = conjf_(J1);
            cxf dJ = cf(1, 0) - J1 * J1c;
            cxf J2 = -((c21 / (za2 * dza) + c22 / (zb2 * z)) * 2.0f);
            cxf zaltc = yc + (c21 / dza + c22 / z);
            cxf za2b = zaltc - c20;
            cxf Jaltc = c21 / (za2b * za2b) + c22 / (zaltc * zaltc);
            cxf Jalt = conjf_(Jaltc);
            cxf JJalt2 = cf(1, 0) - J1c * Jalt;
            cxf J3 = J2 * J1c * JJalt2;
            J3 = (J3 - conjf_(J3) * Jalt) / ((JJalt2 * JJalt2) * dJ.re);
            float cq = (dJ.re < -10) ? -1.0f : (J3.re * J3.re + J3.im * J3.im);
            if (k == 0) cq2 = cq; else { if (cq2 < 0 || cq < 0) cq2 = 0; else if (cq > cq2) cq2 = cq; }
        }
        corrquad2 = cq2;
    } else corrquad2 = 0.0f;
    float safedist;
    if (qc.re < 0.01f) {
        float sd = y1v + (a.re - 1.0f / a.re) * c21.re;
        safedist = sd * sd + y2v * y2v - 4.0f * sqrtf(qc.re) / (a.re * a.re);
    } else {
        float sabs = fabsf(a.re); float Rinf = sabs + 1.0f / sabs + 2.0f;
        float d2 = y1v * y1v + y2v * y2v; safedist = 10.0f;
        if (d2 > Rinf * Rinf) { float d = sqrtf(d2); safedist = (d - Rinf) * (d - Rinf); }
    }
    out.mag = (n_images > 0) ? mag : -1.0f;
    out.corrquad = corrquad; out.corrquad2 = corrquad2; out.safedist = safedist; out.n_images = n_images;
    return out;
}
inline bool fastpath_f32(const Mag0ResultF& r, float rho, float Tol = 1.e-2f) {
    float rho2 = rho * rho;
    float cq = r.corrquad * 6.0f * (rho2 + 1.e-4f * Tol);
    float cq2 = r.corrquad2 * 256.0f * (rho2 + 1.e-8f);
    return (cq < Tol) && (cq2 < 1.0f) && (r.safedist > 4.0f * rho2);
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
        Mag0ResultF mine = binary_mag0_f32((float)s, (float)q, (float)y1, (float)y2);

        if (ref_mag0 < 0 || mine.mag < 0) { local_noimage++; continue; }

        double relerr = fabs((double)mine.mag - ref_mag0) / ref_mag0;
        if (relerr > local_worst) local_worst = relerr;
        if (relerr > mag_tol) {
            local_mismatch++;
            if (local_mismatch <= 5) printf("  [%s] MAG MISMATCH: s=%.4f q=%.4g y1=%.4f y2=%.4f ref=%.8f f32=%.8f relerr=%.3e\n",
                label, s, q, y1, y2, ref_mag0, mine.mag, relerr);
        }

        bool my_ok = fastpath_f32(mine, (float)rho);
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

    // Same two passes as test_kampylos_gpu_mag.cpp's double-precision validation: general
    // (full parameter space) and a stress pass on the hardest regime (resonant topology,
    // planetary mass ratio, source near the lens).
    run_pass("general", vbm, rng, -1.0, 1.0, -6.0, 0.0, -2.0, 2.0, -4.0, -0.3, 20000, 1e-4);
    run_pass("resonant/planetary", vbm, rng, -0.05, 0.05, -5.0, -1.0, -0.6, 0.6, -4.0, -1.5, 50000, 1e-4);

    printf("\n=== TOTAL: %d checked, %d no-image, %d mag mismatches (relerr>1e-4), worst relerr %.3e, %d UNSAFE decisions ===\n",
        n_checked, n_noimage, n_mismatch, worst_relerr, n_decision_unsafe);
    bool pass = (n_decision_unsafe == 0) && (worst_relerr < 1e-3);
    printf("\n%s\n", pass ? "PASS (float32 usable, with caveats noted above)" : "FAIL (float32 not safe enough as-is)");
    return pass ? 0 : 1;
}
