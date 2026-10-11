#pragma once
// BOINC plumbing shared by the four Kampylos app mains (CPU main_kampylos_boinc.cpp, CUDA
// main_kampylos_gpu_boinc.cpp, OpenCL main_kampylos_opencl_boinc.cpp, Metal
// main_kampylos_metal_boinc.cpp) -- input parsing, photometry loading, the single-lens anchor,
// output header/resume handling, checkpointing. The only per-backend code left in each main is
// device setup and the one call that fits a grid cell, so the four apps cannot drift apart in
// file format or grid convention again (the ln/log10 bug was in all four copies).
//
// INCLUDE ORDER: the including main must include vendor/kampylos/src/kampylos_wu.h (and its
// backend header) BEFORE this file: boinc_api.h drags in windows.h on Windows, whose macros
// (e.g. "scr2") collide with names inside VBMicrolensingLibrary.h -- see the note in
// main_kampylos_boinc.cpp.
#include <cstdio>
#include <string>
#include <vector>
#include "boinc_api.h"
#include "filesys.h"

static const char *KAMPYLOS_CHECKPOINT_NAME = "kampylos_checkpoint";

// Checkpoint format: a single integer, the number of grid cells already completed (and
// therefore already written to "out"). Row-major over (is, iq).
static int kampylos_read_checkpoint() {
    char path[512];
    if (boinc_resolve_filename(KAMPYLOS_CHECKPOINT_NAME, path, sizeof(path))) return 0;
    FILE *f = boinc_fopen(path, "r");
    if (!f) return 0;
    int done = 0;
    if (fscanf(f, "%d", &done) != 1) done = 0;
    fclose(f);
    return done;
}

// Written through a temporary file and renamed, so a crash can never leave a torn checkpoint behind.
static void kampylos_write_checkpoint(int done) {
    char path[512];
    if (boinc_resolve_filename(KAMPYLOS_CHECKPOINT_NAME, path, sizeof(path))) return;
    std::string tmp = std::string(path) + ".tmp";
    FILE *f = boinc_fopen(tmp.c_str(), "w");
    if (!f) return;
    fprintf(f, "%d\n", done);
    fclose(f);
    boinc_rename(tmp.c_str(), path);
}

// The CPU app's seed-level checkpoint (see main_kampylos_boinc.cpp) belongs to the cell being fitted; it is removed
// as soon as that cell is finished, so a stale one can never be mistaken for a later cell's.
static const char *KAMPYLOS_SEED_CHECKPOINT_NAME = "kampylos_seed_checkpoint";
static void kampylos_remove_seed_checkpoint() {
    char path[512];
    if (boinc_resolve_filename(KAMPYLOS_SEED_CHECKPOINT_NAME, path, sizeof(path))) return;
    boinc_delete_file(path);
}

// Number of complete result lines (finished cells) in an existing "out", or -1 if it has no valid format-2 header.
// The lines are the authoritative record of finished cells: each one is written and flushed only after its cell is done.
static int kampylos_count_done_cells(const char *path) {
    FILE *f = boinc_fopen(path, "rb");
    if (!f) return -1;
    std::string content;
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) content.append(buf, n);
    fclose(f);
    if (content.compare(0, 7, "# grid=") != 0) return -1;
    int lines = 0;
    size_t pos = 0;
    while (pos < content.size()) {
        size_t eol = content.find('\n', pos);
        if (eol == std::string::npos) break;           // torn last line
        if (content[pos] != '#') lines++;
        pos = eol + 1;
    }
    return lines;
}

struct KampylosBoincJob {
    KampylosWuInput in;
    std::vector<DataPoint> data;
    KampylosWuAnchor anchor;
    FILE *fout = nullptr;
    int start_cell = 0, total = 0;
};

// Resolves/parses "in" and loads "photometry". Returns 0 on success; on failure prints the
// reason and returns non-zero (the caller does boinc_finish(1)).
// allow_parts: only the CPU app can fit a part of each cell's seeds (input token "part i K", v12); the batched GPU
// backends refuse such a work unit instead of silently computing every seed.
static int kampylos_boinc_load_inputs(KampylosBoincJob &job, bool allow_parts = false) {
    char phot_path[512], in_path[512];
    if (boinc_resolve_filename("photometry", phot_path, sizeof(phot_path))) {
        fprintf(stderr, "can't resolve input file 'photometry'\n");
        return 1;
    }
    if (boinc_resolve_filename("in", in_path, sizeof(in_path))) {
        fprintf(stderr, "can't resolve input file 'in'\n");
        return 1;
    }
    FILE *fin = boinc_fopen(in_path, "r");
    if (!fin) {
        fprintf(stderr, "can't open input file %s\n", in_path);
        return 1;
    }
    std::string err;
    bool ok = kampylos_read_input_file(fin, &job.in, &err);
    fclose(fin);
    if (!ok) {
        fprintf(stderr, "malformed input file: %s\n", err.c_str());
        return 1;
    }
    if (job.in.nparts > 1 && !allow_parts) {
        fprintf(stderr, "this backend cannot run a seed-split work unit (part %d of %d)\n", job.in.part, job.in.nparts);
        return 1;
    }
    try {
        job.data = load_photometry(phot_path, job.in.is_mag);
    } catch (const std::exception &e) {
        fprintf(stderr, "error loading photometry: %s\n", e.what());
        return 1;
    }
    if (job.data.size() < 20) {
        fprintf(stderr, "error: only %zu usable data points (need at least 20)\n", job.data.size());
        return 1;
    }
    job.total = job.in.n_s * job.in.n_q;
    const char *g = kampylos_grid_name(job.in.log10_grid);
    fprintf(stderr, "Loaded %zu data points; grid %dx%d (%s), %s_s [%.4f,%.4f], %s_q [%.4f,%.4f]\n",
            job.data.size(), job.in.n_s, job.in.n_q, g, g, job.in.s_min, job.in.s_max, g, job.in.q_min, job.in.q_max);
    return 0;
}

static void kampylos_boinc_compute_anchor(KampylosBoincJob &job) {
    job.anchor = kampylos_compute_anchor(job.data);
    kampylos_log_anchor(stderr, job.anchor);
}

// Opens "out": fresh (header written) on a first run, trimmed back to the checkpointed cells on
// a resume. Returns 0 on success.
static int kampylos_boinc_open_output(KampylosBoincJob &job, const char *backend) {
    char out_path[512];
    if (boinc_resolve_filename("out", out_path, sizeof(out_path))) {
        fprintf(stderr, "can't resolve output file 'out'\n");
        return 1;
    }
    // Resume point = the number of complete cell lines already in "out" (they are the record of finished cells; the
    // cell checkpoint file can only be older or missing, never ahead). Nothing finished is ever thrown away.
    int in_file = kampylos_count_done_cells(out_path);
    int cp = kampylos_read_checkpoint();
    job.start_cell = in_file > 0 ? in_file : 0;
    if (job.start_cell > job.total) job.start_cell = job.total;
    if (job.start_cell > 0 && !kampylos_trim_output(out_path, job.start_cell)) {
        fprintf(stderr, "output file unreadable after %d cells -- restarting from cell 0\n", job.start_cell);
        job.start_cell = 0;
    }
    if (cp != job.start_cell) fprintf(stderr, "cell checkpoint %d, finished lines in out %d: resuming from %d\n", cp, in_file, job.start_cell);
    fprintf(stderr, "Resuming from cell %d/%d\n", job.start_cell, job.total);
    job.fout = boinc_fopen(out_path, job.start_cell == 0 ? "w" : "a");
    if (!job.fout) {
        fprintf(stderr, "can't open %s for writing\n", out_path);
        return 1;
    }
    if (job.start_cell == 0) {
        kampylos_write_header(job.fout, job.in, job.anchor, backend);
        fflush(job.fout);
    }
    return 0;
}

static void kampylos_boinc_after_cell(int done, int total) {
    boinc_fraction_done((double)done / (double)total);
    // Once per cell (minutes to hours apart): always, and not only when the client's timer says so -- the CPU app's
    // seed checkpoints used to consume that timer and starve this one (review 2026-10-10).
    boinc_begin_critical_section();
    kampylos_write_checkpoint(done);
    kampylos_remove_seed_checkpoint();
    boinc_end_critical_section();
    boinc_checkpoint_completed();
}

// Runs every remaining cell with `fit_cell` and finishes the WU.
template <class FitCell>
static int kampylos_boinc_run(KampylosBoincJob &job, VBMicrolensing &vbm, FitCell fit_cell) {
    kampylos_run_cells(job.fout, stderr, job.in, job.data, job.anchor, vbm, job.start_cell, job.total,
                       fit_cell, kampylos_boinc_after_cell);
    fclose(job.fout);
    kampylos_write_checkpoint(job.total);
    boinc_fraction_done(1.0);
    boinc_finish(0);
    return 0;
}
