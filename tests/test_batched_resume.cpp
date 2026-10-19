// Checks the GPU apps' batched Nelder-Mead driver's in-cell checkpoint (src/kampylos_batched_fit.h)
// with a CPU evaluator standing in for the GPU kernel:
//   1. the driver state round-trips through its text form exactly;
//   2. a cell "killed" after random rounds -- repeatedly, so kills land in the initial pass and
//      in the restart passes -- and resumed from the checkpoint each time ends bit-identical to
//      one uninterrupted run, and a checkpoint for another cell is ignored;
//   3. the same with the rounds split into 3-candidate evaluation calls, so kills and
//      checkpoints land inside rounds (the `partial` state, 2026-10-19);
//   4. progress is reported monotonically from 0 to 1.
// Uses the committed synthetic planet light curve and a handful of its seeds (minutes).
//
// Build (from the repo root):
//   g++ -O2 -std=c++17 -I src -I vendor/VBMicrolensing/VBMicrolensing/lib tests/test_batched_resume.cpp \
//       vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.cpp -o test_batched_resume
//   ./test_batched_resume tests/regression/synthetic_planet.dat
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include "kampylos_wu.h"
#include "kampylos_batched_fit.h"

static int fails = 0;
static void check(bool ok, const char* what) {
    printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) fails++;
}
static bool same_result(const BinaryFitResult& a, const BinaryFitResult& b) { return memcmp(&a, &b, sizeof(BinaryFitResult)) == 0; }

// The CPU objective, batched: what the GPU kernels compute (chi2 of each candidate at this cell).
struct CpuEval {
    VBMicrolensing vbm;
    const std::vector<DataPoint>& data;
    double ln_s, ln_q, fs_bound;
    std::vector<double> mag;
    long calls = 0;
    CpuEval(const std::vector<DataPoint>& d, double s, double q) : data(d), ln_s(s), ln_q(q), fs_bound(flux_fit_bound(d)), mag(d.size()) {}
    std::vector<double> operator()(const std::vector<Candidate>& batch) {
        std::vector<double> out(batch.size());
        for (size_t k = 0; k < batch.size(); k++) {
            const Candidate& c = batch[k];
            double tE = exp(c.log_tE), rho = exp(c.log_rho);
            if (tE < 0.01 || tE > 10000.0 || rho < 1e-6 || rho > 1.0) { out[k] = 1e18; continue; }
            double pr[7] = { ln_s, ln_q, c.u0, c.alpha, c.log_rho, c.log_tE, c.t0 };
            bool ok = true;
            for (size_t i = 0; i < data.size() && ok; i++) { mag[i] = vbm.BinaryLightCurve(pr, data[i].t); if (!std::isfinite(mag[i]) || mag[i] <= 0) ok = false; }
            out[k] = ok ? linear_flux_fit(data, mag, fs_bound).chi2 : 1e18;
            calls++;
        }
        return out;
    }
};

struct MemoryCheckpoint {
    bool present = false;
    std::string text;
    long rounds = 0, stop_after = -1;
    double last_progress = -1;
    bool monotonic = true;
    KampylosBatchedHooks hooks() {
        KampylosBatchedHooks h;
        h.read_checkpoint = [this](std::string& t) { if (!present) return false; t = text; return true; };
        h.write_checkpoint = [this](const std::string& t) { text = t; present = true; };
        h.time_to_checkpoint = []() { return true; };
        h.checkpoint_completed = []() {};
        h.progress = [this](double p) { if (p < last_progress || p < 0 || p > 1) monotonic = false; last_progress = p; };
        h.keep_going = [this]() { rounds++; return stop_after < 0 || rounds < stop_after; };
        return h;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <photometry (flux)>\n", argv[0]); return 2; }
    std::vector<DataPoint> data = load_photometry(argv[1], false);
    printf("loaded %zu points\n", data.size());
    KampylosWuAnchor anchor = kampylos_compute_anchor(data);
    const double ln_s = log(pow(10.0, 0.2)), ln_q = log(1e-3);
    BinarySeedSet all = kampylos_make_seeds(anchor.seeds, exp(ln_s), exp(ln_q));
    BinarySeedSet seeds;
    for (size_t k = 0; k < all.size(); k += all.size() / 6) seeds.add(all.x0[k], all.step[k]);
    printf("   %zu of %zu seeds\n", seeds.size(), all.size());

    printf("== uninterrupted run\n");
    CpuEval eval(data, ln_s, ln_q);
    MemoryCheckpoint plain;
    plain.hooks();
    MemoryCheckpoint none;
    KampylosBatchedHooks nh = none.hooks();
    BinaryFitResult full = kampylos_batched_fit_seeds(eval, ln_s, ln_q, seeds, 2, &nh, 5, stdout);
    printf("   chi2 %.6f after %ld rounds, %ld evaluations\n", full.chi2, none.rounds, eval.calls);
    check(none.monotonic && none.last_progress > 0.99, "progress monotonic and reaches ~1");
    check(none.present, "a checkpoint was written");

    printf("== state round trip\n");
    {
        KampylosBatchedState s;
        check(kampylos_batched_state_from_text(none.text, &s), "the last checkpoint parses");
        check(kampylos_batched_state_to_text(s) == none.text, "re-serialising gives the identical text");
        check(!kampylos_batched_state_from_text(none.text.substr(0, none.text.size() / 2), &s), "a torn checkpoint is rejected");
    }

    printf("== killed every ~%ld rounds and resumed\n", none.rounds / 7);
    {
        const long total_rounds = none.rounds;
        MemoryCheckpoint cp;
        bool completed = false;
        int runs = 0, resumed_runs = 0;
        BinaryFitResult r{};
        long stop = total_rounds / 7 + 3;
        while (!completed && runs < 40) {
            cp.rounds = 0;
            cp.stop_after = stop;
            KampylosBatchedHooks h = cp.hooks();
            CpuEval e(data, ln_s, ln_q);
            r = kampylos_batched_fit_seeds(e, ln_s, ln_q, seeds, 2, &h, 5, nullptr, &completed);
            runs++;
            if (!completed) resumed_runs++;
            stop = total_rounds / 5 + (runs * 37) % 50;   // vary the kill point
        }
        printf("   %d runs (%d of them interrupted)\n", runs, resumed_runs);
        check(completed, "eventually completes");
        check(resumed_runs >= 3, "at least three kills happened");
        check(same_result(full, r), "result bit-identical to the uninterrupted run");
    }

    printf("== killed inside rounds (3 candidates per evaluation call) and resumed\n");
    {
        MemoryCheckpoint cp;
        bool completed = false;
        int runs = 0, resumed_runs = 0, partial_checkpoints = 0;
        bool partial_round_trip = true;
        BinaryFitResult r{};
        long stop = 11;   // in evaluation calls now, most of them inside a round
        while (!completed && runs < 400) {
            cp.rounds = 0;
            cp.stop_after = stop;
            KampylosBatchedHooks h = cp.hooks();
            h.eval_chunk = 3;
            CpuEval e(data, ln_s, ln_q);
            r = kampylos_batched_fit_seeds(e, ln_s, ln_q, seeds, 2, &h, 5, nullptr, &completed);
            runs++;
            if (!completed) {
                resumed_runs++;
                KampylosBatchedState st;
                if (kampylos_batched_state_from_text(cp.text, &st) && !st.partial.empty()) {
                    partial_checkpoints++;
                    if (kampylos_batched_state_to_text(st) != cp.text) partial_round_trip = false;
                }
            }
            stop = 7 + (runs * 53) % 90;   // vary the kill point
        }
        printf("   %d runs (%d of them interrupted, %d inside a round)\n", runs, resumed_runs, partial_checkpoints);
        check(completed, "eventually completes");
        check(partial_checkpoints >= 3, "at least three kills landed inside a round");
        check(partial_round_trip, "checkpoints taken inside a round round-trip exactly");
        check(same_result(full, r), "result bit-identical to the uninterrupted run");
    }

    printf("== checkpoint for another cell\n");
    {
        MemoryCheckpoint other;
        other.present = true;
        other.text = none.text;   // cell 5's final state
        KampylosBatchedHooks h = other.hooks();
        CpuEval e(data, ln_s, ln_q);
        BinaryFitResult r = kampylos_batched_fit_seeds(e, ln_s, ln_q, seeds, 2, &h, 6, nullptr);
        check(same_result(full, r), "ignored: cell 6 computed from scratch gives the same answer");
    }

    printf("\n%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
