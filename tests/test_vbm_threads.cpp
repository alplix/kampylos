// Checks that VBMicrolensing can be called from several threads at once through separate
// instances -- which it can only after vendor/patch_vbm_thread_local.py (its function-local
// statics are process-wide otherwise). Four threads each evaluate BinaryMag2 on their own
// random parameter stream; every value must equal the sequential evaluation of the same stream,
// and the process must not crash. With the unpatched v5.6 library this test crashes or reports
// mismatches within seconds; with the patched one it passes.
//
// Build (from the repo root):
//   g++ -O2 -std=c++17 -I vendor/VBMicrolensing/VBMicrolensing/lib tests/test_vbm_threads.cpp \
//       vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp -lpthread -o test_vbm_threads
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include <thread>
#include <vector>
#include "VBMicrolensingLibrary.h"

struct Trial { double s, q, y1, y2, rho; };

static std::vector<Trial> make_trials(unsigned seed, int n) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0, 1);
    std::vector<Trial> t(n);
    for (auto& x : t) {
        x.s = pow(10.0, -0.5 + 1.0 * U(rng)); x.q = pow(10.0, -4.0 + 4.0 * U(rng));
        x.y1 = -1.5 + 3.0 * U(rng); x.y2 = -1.5 + 3.0 * U(rng); x.rho = pow(10.0, -4.0 + 3.0 * U(rng));
    }
    return t;
}

static void evaluate(const std::vector<Trial>& t, std::vector<double>& out) {
    VBMicrolensing vbm;
    vbm.Tol = 1e-2;
    out.resize(t.size());
    for (size_t i = 0; i < t.size(); i++) out[i] = vbm.BinaryMag2(t[i].s, t[i].q, t[i].y1, t[i].y2, t[i].rho);
}

int main(int argc, char** argv) {
    const int n_threads = 4, n_trials = argc > 1 ? atoi(argv[1]) : 3000;
#ifdef VBM_THREAD_LOCAL_STATICS
    printf("library reports VBM_THREAD_LOCAL_STATICS\n");
#else
    printf("library is NOT patched (VBM_THREAD_LOCAL_STATICS undefined) -- expect a crash or mismatches\n");
#endif
    std::vector<std::vector<Trial>> trials(n_threads);
    std::vector<std::vector<double>> reference(n_threads), parallel(n_threads);
    for (int k = 0; k < n_threads; k++) { trials[k] = make_trials(100 + k, n_trials); evaluate(trials[k], reference[k]); }

    std::vector<std::thread> workers;
    for (int k = 0; k < n_threads; k++) workers.emplace_back([&, k]() { evaluate(trials[k], parallel[k]); });
    for (auto& w : workers) w.join();

    // The library is not bit-reproducible even sequentially (its root polishing uses a
    // random_device-seeded engine; two passes in one process differ by ~1e-16 on a few points),
    // so compare with a tolerance far below anything a shared-scratch race produces.
    long mismatches = 0;
    double worst = 0;
    for (int k = 0; k < n_threads; k++)
        for (int i = 0; i < n_trials; i++) {
            double rel = fabs(parallel[k][i] - reference[k][i]) / std::max(fabs(reference[k][i]), 1e-300);
            if (!(rel <= 1e-9)) mismatches++;
            if (rel > worst) worst = rel;
        }
    printf("%d threads x %d BinaryMag2 calls: %ld mismatches vs sequential (worst relative difference %.3g)\n",
           n_threads, n_trials, mismatches, worst);
    printf("\n%s\n", mismatches ? "FAIL" : "PASS");
    return mismatches ? 1 : 0;
}
