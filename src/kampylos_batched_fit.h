#pragma once
// Backend-independent batched multistart driver, shared by the CUDA (gpu_fit_binary.cu),
// OpenCL (kampylos_opencl_fit.cpp) and Metal (kampylos_metal_fit.mm) paths -- they used to carry
// three copies of this loop. Each backend only supplies `eval`, a callable that takes a batch of
// candidates (all at the same grid cell) and returns one chi2 per candidate, same order.
//
// Same algorithm as the CPU path (binary_fit.h's fit_binary_local_steps over a BinarySeedSet):
// one Nelder-Mead pass per seed (max_iter 400, ftol 1e-9, xtol 1e-8), then up to n_restarts
// restart passes from the converged point with steps {0.3, 0.02, 0.08, 0.15, 0.2}
// (ftol 1e-10, xtol 1e-9). As on the CPU, a seed only gets another restart while its previous
// restart improved it (2026-10-08: the GPU drivers used to restart every seed n_restarts times
// unconditionally, which cost extra GPU rounds and made CPU/GPU cell-by-cell comparisons
// needlessly loose).
#include <vector>
#include <cmath>
#include "nelder_mead_stepper.h"
#include "kampylos_gpu_types.h"
#include "binary_fit.h"

static inline Candidate kampylos_to_candidate(const std::vector<double>& x) {
    Candidate c;
    c.t0 = x[0]; c.u0 = x[1]; c.log_tE = x[2]; c.alpha = x[3]; c.log_rho = x[4];
    return c;
}

// Drives N independent NelderMeadStepper instances to completion, batching every round's
// pending points across all of them into one eval() call.
template <class Eval>
std::vector<SimplexResultLite> kampylos_batched_nelder_mead(
    Eval& eval,
    const std::vector<std::vector<double>>& x0s,
    const std::vector<std::vector<double>>& steps,
    int max_iter, double ftol, double xtol
) {
    int n_seeds = (int)x0s.size();
    std::vector<NelderMeadStepper> steppers;
    steppers.reserve(n_seeds);
    for (int i = 0; i < n_seeds; i++) steppers.emplace_back(x0s[i], steps[i], max_iter, ftol, xtol);

    for (;;) {
        std::vector<Candidate> batch;
        std::vector<int> pending_count(n_seeds, 0);
        bool any_active = false;
        for (int s_i = 0; s_i < n_seeds; s_i++) {
            if (steppers[s_i].done()) continue;
            any_active = true;
            const auto& pts = steppers[s_i].pending_points();
            pending_count[s_i] = (int)pts.size();
            for (const auto& p : pts) batch.push_back(kampylos_to_candidate(p));
        }
        if (!any_active) break;

        std::vector<double> chi2 = eval(batch);

        int offset = 0;
        for (int s_i = 0; s_i < n_seeds; s_i++) {
            if (pending_count[s_i] == 0) continue;
            std::vector<double> sub(chi2.begin() + offset, chi2.begin() + offset + pending_count[s_i]);
            offset += pending_count[s_i];
            steppers[s_i].submit_results(sub);
        }
    }

    std::vector<SimplexResultLite> out;
    out.reserve(n_seeds);
    for (auto& st : steppers) out.push_back(st.result());
    return out;
}

// Runs a whole seed set at one grid cell and returns the best parameter vector (log_s/log_q
// are set to the given natural logs; fs/fb are left 0 -- the caller finalises on the host).
template <class Eval>
BinaryFitResult kampylos_batched_fit_seeds(Eval& eval, double ln_s, double ln_q,
                                           const BinarySeedSet& seeds, int n_restarts = 2) {
    std::vector<SimplexResultLite> current = kampylos_batched_nelder_mead(eval, seeds.x0, seeds.step, 400, 1e-9, 1e-8);
    std::vector<int> active(current.size(), 1);
    for (int restart = 0; restart < n_restarts; restart++) {
        std::vector<std::vector<double>> rx0, rsteps;
        std::vector<int> idx;
        for (size_t i = 0; i < current.size(); i++) {
            if (!active[i]) continue;
            idx.push_back((int)i);
            rx0.push_back(current[i].x);
            rsteps.push_back({ 0.3, 0.02, 0.08, 0.15, 0.2 });
        }
        if (idx.empty()) break;
        auto next = kampylos_batched_nelder_mead(eval, rx0, rsteps, 400, 1e-10, 1e-9);
        for (size_t k = 0; k < idx.size(); k++) {
            int i = idx[k];
            if (next[k].fval < current[i].fval) current[i] = next[k];
            else active[i] = 0;
        }
    }

    int best = 0;
    for (size_t i = 1; i < current.size(); i++) if (current[i].fval < current[best].fval) best = (int)i;

    BinaryFitResult out;
    out.log_s = ln_s; out.log_q = ln_q;
    out.t0 = current[best].x[0]; out.u0 = current[best].x[1]; out.tE = exp(current[best].x[2]);
    out.alpha = current[best].x[3]; out.rho = exp(current[best].x[4]);
    out.chi2 = current[best].fval;
    out.fs = 0; out.fb = 0;
    return out;
}

// The 96-seed multistart around one anchor (legacy fit_binary_multistart seeds), as a seed set.
inline BinarySeedSet kampylos_legacy_multistart_seeds(double t0_anchor, double u0_anchor, double tE_anchor,
                                                      double rho_seed = 1e-3, int n_alpha_seeds = 8) {
    BinarySeedSet set;
    const double rho_seeds[] = { rho_seed, 1e-4, 1e-2 };
    const double u0_mag_seeds[] = { std::fabs(u0_anchor), 0.02 };
    for (int i = 0; i < n_alpha_seeds; i++) {
        double alpha_seed = 2.0 * M_PI * i / n_alpha_seeds;
        for (double u0_mag : u0_mag_seeds)
            for (double u0_sign : { 1.0, -1.0 })
                for (double rs : rho_seeds)
                    set.add({ t0_anchor, u0_sign * u0_mag, log(tE_anchor), alpha_seed, log(rs) },
                            { std::max(0.5, tE_anchor * 0.05), 0.05, 0.2, 0.3, 0.5 });
    }
    return set;
}

// Flux-centring used by every GPU upload: the weighted mean flux. Subtracting a constant from
// all fluxes changes nothing but fb (fs, chi2 identical) and removes most of the cancellation
// in the chi2 = S_yy - fs*S_Ay - fb*S_1y identity -- essential for Metal's float sums.
inline double kampylos_flux_offset(const std::vector<DataPoint>& data) {
    double sw = 0, swy = 0;
    for (const auto& d : data) { double w = 1.0 / (d.sigma * d.sigma); sw += w; swy += w * d.flux; }
    return sw > 0 ? swy / sw : 0.0;
}
