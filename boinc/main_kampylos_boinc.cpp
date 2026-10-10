// Kampylos, wrapped as a real BOINC app. Unlike the puzzle apps (which checkpoint mid-search
// on a fixed interval), the natural checkpoint granularity here is one grid cell: each cell
// already takes minutes (a full multistart binary-lens fit), so checkpointing between cells
// rather than inside one is both simple and cheap relative to the work it saves.
//
// Two BOINC input files: "photometry" (one event's light curve, shared across every WU that
// covers a sub-range of that event's grid) and "in" (this WU's own small grid sub-range, plus
// the optional grid-convention marker -- see vendor/kampylos/src/kampylos_wu.h). Output ("out")
// is a two-line header (column names with the grid convention, and the single-lens anchor)
// followed by one line per grid cell, appended as each cell finishes.
//
// v2 (2026-10-08): log10 grid marker, exact (unregularized) flux fit, PSPL/FSPL/FSPL+LD anchor
// in the result file, single-lens-reproducing seeds, per-cell few-points diagnostics. Shared
// plumbing lives in kampylos_boinc_common.h; this file only picks the CPU fitter.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Kampylos's headers (and the VBMicrolensing library they pull in) must come BEFORE
// boinc_api.h/filesys.h on Windows: those drag in windows.h, which #defines a pile of
// short, generic-looking macros for dialog-resource constants -- including "scr2", which
// collides with a local variable of the same name inside VBMicrolensingLibrary.h and breaks
// the build with a bizarre "expected unqualified-id before numeric constant" error. Neither
// group of headers actually depends on the other at the header level (only main() below calls
// real BOINC API functions), so including them in this order avoids the collision entirely
// instead of patching vendored third-party source to dodge windows.h's namespace pollution.
#include "../src/kampylos_wu.h"

#include "kampylos_boinc_common.h"

#define KAMPYLOS_VERSION "4"

// v3 (2026-10-09, forum threads 72 and 87): a CPU task is often ONE grid cell, and one cell can run
// for many hours (dozens of seeds x a restart-backed simplex over thousands of data points), while
// the only checkpoint was between cells -- so a restart lost everything and progress sat at the
// client's estimate (99.99%) for hours. Now the seed loop of fit_binary_seeds() runs here with
// progress after every seed and a seed-level checkpoint: cell index, next seed, best fit so far
// (%.17g, so the doubles round-trip exactly). Seeds are deterministic and the best chi2 wins, so a
// resumed cell gives exactly the uninterrupted result. The checkpoint is only used for the cell it
// names; if the cell checkpoint is older, the cell is simply redone.
static const char *SEED_CHECKPOINT_NAME = "kampylos_seed_checkpoint";

static bool read_seed_checkpoint(int *cell, size_t *next, BinaryFitResult *best) {
    char path[512];
    if (boinc_resolve_filename(SEED_CHECKPOINT_NAME, path, sizeof(path))) return false;
    FILE *f = boinc_fopen(path, "r");
    if (!f) return false;
    unsigned long nx = 0;
    bool ok = fscanf(f, "%d %lu %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", cell, &nx, &best->log_s, &best->log_q,
                     &best->t0, &best->u0, &best->tE, &best->alpha, &best->rho, &best->fs, &best->fb, &best->chi2) == 12;
    fclose(f);
    *next = nx;
    return ok;
}

static void write_seed_checkpoint(int cell, size_t next, const BinaryFitResult &b) {
    char path[512];
    if (boinc_resolve_filename(SEED_CHECKPOINT_NAME, path, sizeof(path))) return;
    std::string tmp = std::string(path) + ".tmp";
    FILE *f = boinc_fopen(tmp.c_str(), "w");
    if (!f) return;
    fprintf(f, "%d %lu %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", cell, (unsigned long)next,
            b.log_s, b.log_q, b.t0, b.u0, b.tE, b.alpha, b.rho, b.fs, b.fb, b.chi2);
    fclose(f);
    boinc_rename(tmp.c_str(), path);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    fprintf(stderr, "Kampylos (gravitational microlensing binary-lens grid search) v" KAMPYLOS_VERSION "\n");
    fprintf(stderr, "bitboinc -- https://bitboinc.athena.org.tr/\n");
    fprintf(stderr, "coded by Alperen Yavuz\n");

    BOINC_OPTIONS options;
    boinc_options_defaults(options);
    options.main_program = true;
    options.check_heartbeat = true;
    options.handle_process_control = true;
    int rc = boinc_init_options(&options);
    if (rc) {
        fprintf(stderr, "boinc_init_options failed: %d\n", rc);
        return rc;
    }

    KampylosBoincJob job;
    if (kampylos_boinc_load_inputs(job, true)) { boinc_finish(1); return 1; }
    kampylos_boinc_compute_anchor(job);
    if (kampylos_boinc_open_output(job, "cpu")) { boinc_finish(1); return 1; }

    VBMicrolensing vbm;
    int cell = job.start_cell;   // cells are fitted in order, one call per cell
    return kampylos_boinc_run(job, vbm, [&](double ln_s, double ln_q, const BinarySeedSet &seeds) {
        // same loop as fit_binary_seeds() (vendor/kampylos/src/binary_fit.h), with progress and
        // a seed-level checkpoint
        const size_t n = seeds.x0.size();
        BinaryFitResult best;
        best.chi2 = 1e300;
        size_t k0 = 0;
        int cp_cell;
        size_t cp_next;
        BinaryFitResult cp_best;
        if (read_seed_checkpoint(&cp_cell, &cp_next, &cp_best) && cp_cell == cell && cp_next <= n) {
            best = cp_best;
            k0 = cp_next;
            fprintf(stderr, "resuming cell %d at seed %lu of %lu\n", cell, (unsigned long)k0, (unsigned long)n);
        }
        // v12: "part i K" in the input line = this work unit fits only the seeds with k % K == i (interleaved,
        // so the K parts cost about the same); the server keeps the lowest chi2 per cell over the K results.
        const size_t K = (size_t)job.in.nparts, part = (size_t)job.in.part;
        size_t own_total = 0, own_before = 0;
        for (size_t k = 0; k < n; k++) if (k % K == part) { own_total++; if (k < k0) own_before++; }
        size_t own_done = own_before;
        for (size_t k = k0; k < n; k++) {
            if (k % K != part) continue;
            BinaryFitResult r = fit_binary_local_steps(vbm, job.data, ln_s, ln_q, seeds.x0[k], seeds.step[k]);
            if (r.chi2 < best.chi2) best = r;
            own_done++;
            boinc_fraction_done((cell + (double)own_done / (double)(own_total ? own_total : 1)) / (double)job.total);
            if (boinc_time_to_checkpoint()) {
                write_seed_checkpoint(cell, k + 1, best);
                boinc_checkpoint_completed();
            }
        }
        cell++;
        return best;
    });
}
