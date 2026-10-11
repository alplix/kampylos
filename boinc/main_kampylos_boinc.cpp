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
//
// v3 (2026-10-09, forum threads 72 and 87): a CPU task is often ONE grid cell, and one cell can run
// for many hours, while the only checkpoint was between cells. The seed loop now lives in
// src/kampylos_cpu_cell.h with progress after every seed and a seed-level checkpoint; this file
// only maps its hooks onto the BOINC API (kampylos_boinc_cell_hooks).
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
#include "../src/kampylos_cpu_cell.h"

#include "kampylos_boinc_common.h"

// The version printed to stderr. Build with -DKAMPYLOS_VERSION='"12"' so it matches the
// version_num the server assigns to the binary; the default is only a fallback.
#ifndef KAMPYLOS_VERSION
#define KAMPYLOS_VERSION "4"
#endif

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
        KampylosCellHooks hooks = kampylos_boinc_cell_hooks(cell, job.total);
        BinaryFitResult best = kampylos_fit_cell_resumable(vbm, job.data, ln_s, ln_q, seeds, cell,
                                                           job.in.part, job.in.nparts, hooks, stderr);
        cell++;
        return best;
    });
}
