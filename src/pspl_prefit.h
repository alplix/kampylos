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
    // Independent, purely data-driven fallback anchor -- no optimization involved at all, so it
    // cannot land in the degenerate corners that fool the optimized (t0, u0, tE) above (confirmed
    // on real test data: that optimizer can converge to either a near-constant-at-huge-tE-and-
    // tiny-u0 pedestal, or a near-zero-tE spike invisible between cadence gaps, both of which can
    // numerically out-chi2 a genuine, well-localized fit). t0_raw is just the peak-flux time;
    // tE_raw is derived from how wide the excess-above-baseline region around that peak actually
    // is, so it reflects the event's real visible timescale rather than a value the optimizer
    // walked to.
    double t0_raw, tE_raw;
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
        // tE far beyond the data's own time span is just as unconstrained as tE far below the
        // cadence: the event's rise/fall is never actually sampled, so u stays near-constant at
        // u0 across every point and PSPLMag(u0) can be driven arbitrarily high by u0->0, winning
        // a lower chi2 from a degenerate near-constant-but-huge magnification rather than from
        // genuinely explaining the light curve. Confirmed on real test data: u0 collapsed to its
        // floor (1e-4) and tE blew up to ~8x the data span. Bounding tE above (as well as below)
        // keeps the search inside timescales the data can actually constrain.
        if (tE < 0.05 || tE > 5.0 * t_span || u0 < 1e-4) return 1e18;
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

    best.t0_raw = t0_seed;
    {
        std::vector<double> fluxes;
        fluxes.reserve(data.size());
        for (const auto& d : data) fluxes.push_back(d.flux);
        std::sort(fluxes.begin(), fluxes.end());
        double baseline = fluxes[fluxes.size() / 2];  // median -- robust as long as the event
                                                        // doesn't cover most of the baseline.
        double typical_sigma = data[data.size() / 2].sigma;
        double thresh = baseline + 3.0 * typical_sigma;
        size_t peak_idx = 0;
        for (size_t i = 0; i < data.size(); i++) if (data[i].t == t0_seed) { peak_idx = i; break; }
        size_t lo = peak_idx, hi = peak_idx;
        while (lo > 0 && data[lo - 1].flux > thresh) lo--;
        while (hi + 1 < data.size() && data[hi + 1].flux > thresh) hi++;
        double width = data[hi].t - data[lo].t;
        double tE_raw = width / 2.355;  // excess-region FWHM -> Gaussian-equivalent sigma, used
                                          // only as an order-of-magnitude timescale seed.
        tE_raw = std::max(tE_raw, 1.0);
        tE_raw = std::min(tE_raw, t_span / 3.0);
        best.tE_raw = tE_raw;
    }
    return best;
}
