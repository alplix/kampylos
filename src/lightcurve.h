#pragma once
// Photometry data loading and the flux/chi-squared machinery shared by both the single-lens
// pre-fit and the binary-lens grid search.
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>

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
    double det = S_AA * S_11 - S_A1 * S_A1;
    FluxFit out;
    if (fabs(det) < 1e-300) {
        // Degenerate (e.g. all magnifications identical) -- fall back to fs=0 fit.
        out.fs = 0;
        out.fb = S_11 > 0 ? S_1y / S_11 : 0;
    } else {
        out.fs = (S_11 * S_Ay - S_A1 * S_1y) / det;
        out.fb = (S_AA * S_1y - S_A1 * S_Ay) / det;
    }
    double chi2 = 0;
    for (size_t i = 0; i < n; i++) {
        double w = 1.0 / (data[i].sigma * data[i].sigma);
        double resid = data[i].flux - (out.fs * mag[i] + out.fb);
        chi2 += w * resid * resid;
    }
    out.chi2 = chi2;
    return out;
}
