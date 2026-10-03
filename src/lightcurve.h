#pragma once
// Photometry data loading and the flux/chi-squared machinery shared by both the single-lens
// pre-fit and the binary-lens grid search.
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

struct DataPoint {
    double t;      // time, HJD (or HJD - 2450000, whatever the source file uses -- kept
                    // consistent within one event, that's all that matters for fitting)
    double flux;
    double sigma;  // flux uncertainty, same units as flux
};

// Loads "time value error" (whitespace-separated, '#'-prefixed comment lines skipped). If
// input_is_mag is true, value/error are magnitude/mag-error and get converted to flux via
// flux = 10^(-0.4*(mag - zp)), sigma_flux = flux * ln(10) * 0.4 * mag_err -- the standard
// linear-error-propagation approximation, valid for the sub-0.1-mag errors typical of these
// survey alert-system light curves. zp is arbitrary (only relative flux matters for fitting,
// since fs/fb are fitted anyway) -- 18.0 is just a convenient round number, not a real
// zeropoint from any specific instrument.
inline std::vector<DataPoint> load_photometry(const std::string& path, bool input_is_mag) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<DataPoint> out;
    std::string line;
    const double zp = 18.0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double t, v, e;
        if (!(ss >> t >> v >> e)) continue;
        DataPoint dp;
        dp.t = t;
        if (input_is_mag) {
            dp.flux = pow(10.0, -0.4 * (v - zp));
            dp.sigma = dp.flux * log(10.0) * 0.4 * e;
        } else {
            dp.flux = v;
            dp.sigma = e;
        }
        if (dp.sigma > 0 && std::isfinite(dp.flux)) out.push_back(dp);
    }
    return out;
}

struct FluxFit {
    double fs, fb;   // source flux, blend flux
    double chi2;
};

// Given a magnification model evaluated at every data point, finds the best-fit (fs, fb) in
// closed form (model flux = fs*A + fb is linear in fs, fb for fixed A) and returns the
// resulting chi-squared. This is standard practice in microlensing fitting: it removes two
// parameters from every nonlinear optimization step for free, since they never need a
// numerical search to begin with.
inline FluxFit linear_flux_fit(const std::vector<DataPoint>& data, const std::vector<double>& mag) {
    double S_AA = 0, S_A1 = 0, S_11 = 0, S_Ay = 0, S_1y = 0;
    size_t n = data.size();
    for (size_t i = 0; i < n; i++) {
        double w = 1.0 / (data[i].sigma * data[i].sigma);
        double A = mag[i];
        S_AA += w * A * A;
        S_A1 += w * A;
        S_11 += w;
        S_Ay += w * A * data[i].flux;
        S_1y += w * data[i].flux;
    }
    // When A (the model magnification) is nearly constant across the data -- e.g. a PSPL/binary
    // trial whose timescale is much shorter than the data's own cadence, so essentially no point
    // actually samples the event -- the 2x2 normal equations above become near-singular, and
    // fs/fb are jointly almost unconstrained. Solving that system exactly still finds the true
    // minimum of the (nearly meaningless) chi2 surface for that A, but a *local* optimizer
    // driving (t0, u0, tE) will discover this and exploit it: it can always walk towards
    // whatever makes the design more singular, since chi2 keeps improving as fs/fb race off to
    // huge, cancelling values that overfit noise rather than fit the real signal. This was
    // confirmed directly and is adversarial, not a one-off: raising the degeneracy threshold
    // from an absolute 1e-300 cutoff to a relative one, and separately capping |fs|/|fb| against
    // the data's own flux range, both got defeated in turn -- the optimizer just walked up to
    // whichever cutoff was in place and rode its edge (e.g. landing fs/fb in the hundreds right
    // at a 1e-6 relative-determinant cutoff). Any hard cutoff creates a cliff the optimizer can
    // climb towards from the permissive side.
    //
    // The fix is to remove the cliff rather than move it: Tikhonov (ridge) regularization adds a
    // small, scale-relative damping term to the diagonal of the normal equations. This bounds
    // fs/fb continuously as the design approaches singularity -- there is no longer a boundary
    // where chi2 keeps improving right up to a jump, because the ridge term costs a little more
    // bias the closer the fit leans on an ill-constrained direction, smoothly outweighing
    // whatever spurious chi2 "gain" a near-singular design offered. lambda is tied to S_11 (the
    // sum of weights -- the problem's natural scale) rather than an absolute value, and is small
    // enough (1e-3) to leave any well-conditioned fit (real signal, S_A1^2 well below S_AA*S_11)
    // essentially unaffected.
    double lambda = 1e-3 * S_11;
    double S_AA_r = S_AA + lambda, S_11_r = S_11 + lambda;
    double det = S_AA_r * S_11_r - S_A1 * S_A1;
    FluxFit out;
    out.fs = (S_11_r * S_Ay - S_A1 * S_1y) / det;
    out.fb = (S_AA_r * S_1y - S_A1 * S_Ay) / det;
    double chi2 = 0;
    for (size_t i = 0; i < n; i++) {
        double w = 1.0 / (data[i].sigma * data[i].sigma);
        double resid = data[i].flux - (out.fs * mag[i] + out.fb);
        chi2 += w * resid * resid;
    }
    out.chi2 = chi2;
    return out;
}
