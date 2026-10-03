#pragma once
// A "steppable" re-implementation of simplex.h's nelder_mead(): same algorithm (alpha=1,
// gamma=2, rho=0.5, sigma=0.5 -- textbook Nelder-Mead), but instead of blocking until converged,
// it exposes "what point(s) do you need evaluated next" / "here are the results, what's next"
// so many independent instances can be driven in lockstep rounds, each round's pending points
// across ALL instances batched into one evaluation call -- which is the whole reason this exists:
// kampylos's GPU kernel evaluates a BATCH of candidates against a light curve in one launch, and
// fit_binary_multistart() needs 96 independent Nelder-Mead searches per grid cell, each wanting
// its own next point at its own pace. Driving them one at a time (96 separate blocking
// nelder_mead() calls, as the original CPU code does) would leave the GPU evaluating a single
// candidate per launch -- correct, but throwing away all the batching a GPU is for.
//
// Validated (tests/test_nelder_mead_stepper.cpp) to reach the IDENTICAL final point/value as
// simplex.h's own nelder_mead() on the same problems -- this is a refactor of a proven algorithm,
// not a new one, so that equivalence is the thing that actually needs checking, not the math.
#include <vector>
#include <algorithm>
#include <cmath>

enum class NMPhase { NeedInit, NeedReflect, NeedExpand, NeedContract, NeedShrink, Done };

struct SimplexResultLite {
    std::vector<double> x;
    double fval;
    int iterations;
    bool converged;
};

class NelderMeadStepper {
public:
    NelderMeadStepper(std::vector<double> x0, std::vector<double> step, int max_iter = 2000, double ftol = 1e-10, double xtol = 1e-10)
        : n((int)x0.size()), n1((int)x0.size() + 1), max_iter(max_iter), ftol(ftol), xtol(xtol),
          simplex(n1, x0), fval(n1, 0.0), phase(NMPhase::NeedInit), iter(0), converged(false)
    {
        for (int i = 0; i < n; i++) simplex[i + 1][i] += step[i];
        pending = simplex; // NeedInit: caller must evaluate all n1 vertices, in simplex order
    }

    bool done() const { return phase == NMPhase::Done; }

    // Points this instance needs evaluated right now, in the order submit_results() expects
    // their values back.
    const std::vector<std::vector<double>>& pending_points() const { return pending; }

    // Call once pending_points() is non-empty and you have f() for each of them, in order.
    void submit_results(const std::vector<double>& fvals) {
        switch (phase) {
        case NMPhase::NeedInit:
            fval = fvals; // one per vertex, n1 of them
            begin_iteration();
            break;
        case NMPhase::NeedReflect: {
            fr = fvals[0];
            if (fr < fval[0]) {
                for (int j = 0; j < n; j++) xe[j] = centroid[j] + gamma * (xr[j] - centroid[j]);
                pending = { xe };
                phase = NMPhase::NeedExpand;
            } else if (fr < fval[n - 1]) {
                simplex[n] = xr; fval[n] = fr;
                begin_iteration();
            } else {
                for (int j = 0; j < n; j++) xc[j] = centroid[j] + rho * (worst[j] - centroid[j]);
                pending = { xc };
                phase = NMPhase::NeedContract;
            }
            break;
        }
        case NMPhase::NeedExpand: {
            double fe = fvals[0];
            if (fe < fr) { simplex[n] = xe; fval[n] = fe; }
            else { simplex[n] = xr; fval[n] = fr; }
            begin_iteration();
            break;
        }
        case NMPhase::NeedContract: {
            double fc = fvals[0];
            if (fc < fval[n]) {
                simplex[n] = xc; fval[n] = fc;
                begin_iteration();
            } else {
                for (int i = 1; i < n1; i++) {
                    for (int j = 0; j < n; j++) simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                }
                pending = std::vector<std::vector<double>>(simplex.begin() + 1, simplex.end());
                phase = NMPhase::NeedShrink;
            }
            break;
        }
        case NMPhase::NeedShrink: {
            for (int i = 1; i < n1; i++) fval[i] = fvals[i - 1];
            begin_iteration();
            break;
        }
        case NMPhase::Done:
            break; // no-op -- caller shouldn't be submitting once done, but ignore rather than crash
        }
    }

    SimplexResultLite result() const {
        int best = 0;
        for (int i = 1; i < n1; i++) if (fval[i] < fval[best]) best = i;
        return { simplex[best], fval[best], iter, converged };
    }

private:
    int n, n1, max_iter;
    double ftol, xtol;
    const double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;

    std::vector<std::vector<double>> simplex;
    std::vector<double> fval;
    std::vector<std::vector<double>> pending;
    NMPhase phase;
    int iter;
    bool converged;

    std::vector<double> centroid, worst, xr, xe, xc;
    double fr = 0;

    // Sorts the simplex, checks convergence, and either finishes (phase=Done) or sets up the
    // next iteration's reflection point request (phase=NeedReflect) -- the one piece of logic
    // every exit path (init, plain accept, expand, contract, shrink) funnels through, exactly
    // mirroring simplex.h's single per-iteration top-of-loop block.
    void begin_iteration() {
        if (iter >= max_iter) { phase = NMPhase::Done; return; }

        std::vector<int> order(n1);
        for (int i = 0; i < n1; i++) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return fval[a] < fval[b]; });
        std::vector<std::vector<double>> s2(n1);
        std::vector<double> f2(n1);
        for (int i = 0; i < n1; i++) { s2[i] = simplex[order[i]]; f2[i] = fval[order[i]]; }
        simplex = s2; fval = f2;

        double fspread = fval[n] - fval[0];
        double xspread = 0;
        for (int i = 1; i < n1; i++) {
            double d = 0;
            for (int j = 0; j < n; j++) { double diff = simplex[i][j] - simplex[0][j]; d += diff * diff; }
            xspread = std::max(xspread, std::sqrt(d));
        }
        if (fspread < ftol && xspread < xtol) { converged = true; phase = NMPhase::Done; return; }

        centroid.assign(n, 0.0);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) centroid[j] += simplex[i][j];
        for (int j = 0; j < n; j++) centroid[j] /= n;

        worst = simplex[n];
        xr.resize(n); xe.resize(n); xc.resize(n);
        for (int j = 0; j < n; j++) xr[j] = centroid[j] + alpha * (centroid[j] - worst[j]);

        iter++;
        pending = { xr };
        phase = NMPhase::NeedReflect;
    }
};
