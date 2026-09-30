#pragma once
// Cheap single-lens pre-fit: gives the binary-lens grid search a real (t0, u0, tE) anchor
// instead of starting every grid cell from a blind guess. Seeded from the data itself (peak
// time, a few u0/tE guesses spanning the plausible range) rather than requiring the caller to
// know anything about the event in advance.
#include <vector>
#include <algorithm>
#include "simplex.h"
#include "lightcurve.h"
#include "../vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.h"

struct PsplFit {
    double t0, u0, tE, chi2;
};

inline PsplFit pspl_prefit(VBMicrolensing& vbm, const std::vector<DataPoint>& data) {
    // Seed t0 from the highest-flux point (a real anchor -- the peak of a microlensing event
    // is, almost by definition, near its highest observed flux).
    double t0_seed = data[0].t;
    double max_flux = data[0].flux;
    for (const auto& d : data) {
        if (d.flux > max_flux) { max_flux = d.flux; t0_seed = d.t; }
    }
    double t_span = data.back().t - data.front().t;

    auto objective = [&](const std::vector<double>& p) -> double {
        double t0 = p[0], u0 = fabs(p[1]), tE = fabs(p[2]);
        if (tE < 0.05 || u0 < 1e-4) return 1e18;
        std::vector<double> mag(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            double u = sqrt(u0 * u0 + ((data[i].t - t0) / tE) * ((data[i].t - t0) / tE));
            mag[i] = vbm.PSPLMag(u);
        }
        return linear_flux_fit(data, mag).chi2;
    };

    PsplFit best;
    best.chi2 = 1e300;
    // A handful of u0/tE seed combinations spanning the plausible range for an alert-survey
    // event -- wide enough to not depend on already knowing the answer, cheap enough (PSPL is
    // fast, no contour integration) that trying several is basically free next to the binary
    // search that follows.
    double u0_seeds[] = { 0.01, 0.05, 0.2, 0.5 };
    double tE_fracs[] = { 0.05, 0.15, 0.35 };
    for (double u0s : u0_seeds) {
        for (double tEf : tE_fracs) {
            double tE_seed = std::max(1.0, t_span * tEf);
            std::vector<double> x0 = { t0_seed, u0s, tE_seed };
            std::vector<double> step = { std::max(1.0, tE_seed * 0.1), u0s * 0.3, tE_seed * 0.3 };
            SimplexResult res = nelder_mead(objective, x0, step, 800, 1e-10, 1e-9);
            if (res.fval < best.chi2) {
                best.t0 = res.x[0]; best.u0 = fabs(res.x[1]); best.tE = fabs(res.x[2]); best.chi2 = res.fval;
            }
        }
    }
    return best;
}
