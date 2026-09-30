#pragma once
// Minimal, self-contained Nelder-Mead simplex minimizer. No external dependency: the fitting
// problem here (a handful of continuous parameters, a cheap-ish objective, no gradients
// available since VBMicrolensing doesn't expose derivatives) is exactly what Nelder-Mead is
// for, and pulling in a whole optimization library for this would be overkill.
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>

struct SimplexResult {
    std::vector<double> x;
    double fval;
    int iterations;
    bool converged;
};

// f: objective to minimize. x0: starting point. step: initial per-dimension step size (the
// simplex's first edge length along each axis). max_iter: safety cap. ftol/xtol: convergence
// thresholds on function-value spread and vertex spread across the simplex.
inline SimplexResult nelder_mead(
    const std::function<double(const std::vector<double>&)>& f,
    std::vector<double> x0,
    std::vector<double> step,
    int max_iter = 2000,
    double ftol = 1e-10,
    double xtol = 1e-10
) {
    int n = (int)x0.size();
    int n1 = n + 1;
    std::vector<std::vector<double>> simplex(n1, x0);
    for (int i = 0; i < n; i++) simplex[i + 1][i] += step[i];

    std::vector<double> fval(n1);
    for (int i = 0; i < n1; i++) fval[i] = f(simplex[i]);

    const double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;

    int iter = 0;
    bool converged = false;
    for (; iter < max_iter; iter++) {
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
        if (fspread < ftol && xspread < xtol) { converged = true; break; }

        std::vector<double> centroid(n, 0.0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) centroid[j] += simplex[i][j];
        }
        for (int j = 0; j < n; j++) centroid[j] /= n;

        std::vector<double> worst = simplex[n];
        std::vector<double> xr(n);
        for (int j = 0; j < n; j++) xr[j] = centroid[j] + alpha * (centroid[j] - worst[j]);
        double fr = f(xr);

        if (fr < fval[0]) {
            std::vector<double> xe(n);
            for (int j = 0; j < n; j++) xe[j] = centroid[j] + gamma * (xr[j] - centroid[j]);
            double fe = f(xe);
            if (fe < fr) { simplex[n] = xe; fval[n] = fe; }
            else { simplex[n] = xr; fval[n] = fr; }
        } else if (fr < fval[n - 1]) {
            simplex[n] = xr; fval[n] = fr;
        } else {
            std::vector<double> xc(n);
            for (int j = 0; j < n; j++) xc[j] = centroid[j] + rho * (worst[j] - centroid[j]);
            double fc = f(xc);
            if (fc < fval[n]) { simplex[n] = xc; fval[n] = fc; }
            else {
                for (int i = 1; i < n1; i++) {
                    for (int j = 0; j < n; j++) simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                    fval[i] = f(simplex[i]);
                }
            }
        }
    }

    int best = 0;
    for (int i = 1; i < n1; i++) if (fval[i] < fval[best]) best = i;
    return SimplexResult{ simplex[best], fval[best], iter, converged };
}
