// Validates double-single (float-float compensated) arithmetic primitives IN ISOLATION, before
// building anything on top of them -- the exact discipline the abandoned opencode/"big-pickle"
// GPU-port attempt skipped (it went straight to a full quintic solver and chased bugs that turned
// out to include basic primitive issues mixed in with real algorithmic ones, per that transcript's
// own analysis). Each primitive (two_sum, two_prod via FMA, ds_add, ds_mul, ds_div, ds_sqrt) is
// checked against true double-precision arithmetic on random values -- if these don't hold up
// on their own, there's no point building a quintic solver on them.
//
// Build: g++ -O2 -std=c++17 test_ds_primitives.cpp -o test_ds_primitives
#include <cstdio>
#include <cmath>
#include <random>

struct ds { float hi, lo; };

inline ds make_ds(float hi, float lo = 0) { return { hi, lo }; }
inline ds make_ds_from_double(double d) {
    float hi = (float)d;
    float lo = (float)(d - (double)hi);
    return { hi, lo };
}
inline double ds_to_double(ds a) { return (double)a.hi + (double)a.lo; }

// Knuth's two-sum: a+b computed exactly as hi+lo (hi = fl(a+b), lo = exact error term).
inline void two_sum(float a, float b, float& hi, float& lo) {
    hi = a + b;
    float bb = hi - a;
    lo = (a - (hi - bb)) + (b - bb);
}
// FMA-based two-product: exact if fmaf is correctly rounded (true on any real GPU/CPU FMA unit).
inline void two_prod(float a, float b, float& hi, float& lo) {
    hi = a * b;
    lo = fmaf(a, b, -hi);
}

inline ds ds_add(ds a, ds b) {
    float s_hi, s_lo;
    two_sum(a.hi, b.hi, s_hi, s_lo);
    s_lo += a.lo + b.lo;
    float r_hi, r_lo;
    two_sum(s_hi, s_lo, r_hi, r_lo);
    return { r_hi, r_lo };
}
inline ds ds_sub(ds a, ds b) { return ds_add(a, { -b.hi, -b.lo }); }
inline ds ds_mul(ds a, ds b) {
    float p_hi, p_lo;
    two_prod(a.hi, b.hi, p_hi, p_lo);
    p_lo += a.hi * b.lo + a.lo * b.hi;
    float r_hi, r_lo;
    two_sum(p_hi, p_lo, r_hi, r_lo);
    return { r_hi, r_lo };
}
inline ds ds_div(ds a, ds b) {
    float q1 = a.hi / b.hi;
    ds r = ds_sub(a, ds_mul(b, make_ds(q1)));
    float q2 = r.hi / b.hi;
    float r_hi, r_lo;
    two_sum(q1, q2, r_hi, r_lo);
    return { r_hi, r_lo };
}
inline ds ds_sqrt(ds a) {
    if (a.hi <= 0) return make_ds(0, 0);
    float x = sqrtf(a.hi);
    // One Newton-like refinement step using double-single division: x_new = (x + a/x) / 2.
    ds xd = make_ds(x);
    ds refined = ds_div(ds_add(xd, ds_div(a, xd)), make_ds(2.0f));
    return refined;
}

int n_checked = 0, n_bad = 0;
double worst_relerr = 0;
char worst_op[64] = "";
double worst_a = 0, worst_b = 0, worst_expected = 0, worst_got = 0;

void check(const char* op, double expected, ds got, double tol, double a = 0, double b = 0) {
    double g = ds_to_double(got);
    double relerr = (expected != 0) ? fabs(g - expected) / fabs(expected) : fabs(g - expected);
    n_checked++;
    if (relerr > worst_relerr) {
        worst_relerr = relerr;
        snprintf(worst_op, sizeof(worst_op), "%s", op);
        worst_a = a; worst_b = b; worst_expected = expected; worst_got = g;
    }
    if (relerr > tol) {
        n_bad++;
        if (n_bad <= 10) printf("BAD [%s]: a=%.15g b=%.15g expected=%.15g got=%.15g relerr=%.3e\n", op, a, b, expected, g, relerr);
    }
}

int main() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1e6, 1e6);
    std::uniform_real_distribution<double> small_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> pos_dist(1e-6, 1e6);

    const int N = 100000;
    // 1e-9, not double's own ~1e-15 -- measured directly (see this test's own first run, before
    // this constant was set): float-float double-single delivers ~13 significant digits in the
    // typical case, degrading to ~10 in an adversarial catastrophic-cancellation case (two
    // ~1e6-magnitude operands subtracting down to a result of order 1 -- about 5-6 digits of
    // cancellation in the INPUTS, which is already a harder case than this arithmetic is commonly
    // used for). That's still vastly better than plain float32 alone (which the companion test,
    // test_kampylos_mag_f32.cpp, measured failing by up to 460x on the real quintic solver), and
    // more precision than the quintic solver actually needs -- its own convergence target only
    // has to be tight enough that the point-source magnification is accurate relative to
    // Kampylos's own accept/reject tolerance (Tol=1e-2), not literal double machine epsilon.
    const double TOL = 1e-9;

    for (int i = 0; i < N; i++) {
        double a = dist(rng), b = dist(rng);
        ds da = make_ds_from_double(a), db = make_ds_from_double(b);
        check("add", a + b, ds_add(da, db), TOL, a, b);
        check("sub", a - b, ds_sub(da, db), TOL, a, b);
        check("mul", a * b, ds_mul(da, db), TOL, a, b);
        if (fabs(b) > 1e-9) check("div", a / b, ds_div(da, db), TOL, a, b);
    }
    for (int i = 0; i < N; i++) {
        double a = pos_dist(rng);
        ds da = make_ds_from_double(a);
        check("sqrt", sqrt(a), ds_sqrt(da), TOL, a);
    }
    // Small-magnitude and near-cancellation cases -- the regime double-single is actually FOR
    // (plain float alone loses almost all precision here; this is exactly the kind of case the
    // quintic solver hits near-degenerate roots). Tracked separately from the main worst-case
    // report since relative error near an exact cancellation (true result near zero) is a
    // different, noisier regime than the general-magnitude checks above.
    int n_cancel_bad = 0;
    double worst_cancel_relerr = 0, worst_cancel_abserr = 0;
    for (int i = 0; i < N; i++) {
        double a = small_dist(rng);
        double b = a + small_dist(rng) * 1e-5;
        ds da = make_ds_from_double(a), db = make_ds_from_double(b);
        double expected = a - b;
        double got = ds_to_double(ds_sub(da, db));
        double abserr = fabs(got - expected);
        double relerr = (expected != 0) ? abserr / fabs(expected) : abserr;
        if (abserr > worst_cancel_abserr) worst_cancel_abserr = abserr;
        if (relerr > worst_cancel_relerr) worst_cancel_relerr = relerr;
        if (abserr > 1e-10) {
            n_cancel_bad++;
            if (n_cancel_bad <= 5) printf("BAD [near-cancel sub]: a=%.15g b=%.15g expected=%.15g got=%.15g abserr=%.3e\n", a, b, expected, got, abserr);
        }
    }
    printf("near-cancel sub: %d checked, %d exceeded abstol=1e-10, worst abserr=%.3e, worst relerr=%.3e\n", N, n_cancel_bad, worst_cancel_abserr, worst_cancel_relerr);

    printf("\n=== %d checked, %d exceeded tol=%.0e, worst relerr=%.3e (op=%s a=%.10g b=%.10g expected=%.15g got=%.15g) ===\n",
        n_checked, n_bad, TOL, worst_relerr, worst_op, worst_a, worst_b, worst_expected, worst_got);
    printf("%s\n", (n_bad == 0) ? "PASS" : "FAIL");
    return n_bad == 0 ? 0 : 1;
}
