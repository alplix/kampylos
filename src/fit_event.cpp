// Standalone CLI driver for one Kampylos work unit, outside BOINC -- runs exactly the same code
// path as the BOINC apps (kampylos_wu.h: input parsing incl. the grid-convention marker, the
// single-lens anchor, kampylos_make_seeds, per-cell host finalisation and the result format),
// with the backend picked at compile time:
//   (default)            CPU   -- fit_binary_seeds()
//   -DKAMPYLOS_CLI_CUDA   CUDA  -- gpu_fit_binary_seeds()     (link gpu_fit_binary.cu)
//   -DKAMPYLOS_CLI_OPENCL OpenCL-- opencl_fit_binary_seeds()  (link kampylos_opencl_fit.cpp + cl_dynload.c)
//
// Usage:
//   fit_event <photometry_file> <in_file> <out_file> [--cells FIRST:COUNT] [--device N] [--anchor-only]
// <in_file> holds one WU input line, e.g. "mag -1 1 10 -5 0 10 log10" (see kampylos_wu.h).
// --cells runs only cells [FIRST, FIRST+COUNT) of the row-major grid (the header is still
// written), so a grid can be split over several processes and the data lines concatenated.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include "kampylos_wu.h"
#if defined(KAMPYLOS_CLI_CUDA)
#include "gpu_fit_binary.h"
#define KAMPYLOS_CLI_BACKEND "cuda"
#elif defined(KAMPYLOS_CLI_OPENCL)
#include "kampylos_opencl_fit.h"
#define KAMPYLOS_CLI_BACKEND "opencl"
#else
#define KAMPYLOS_CLI_BACKEND "cpu"
#endif

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <photometry_file> <in_file> <out_file> [--cells FIRST:COUNT] [--device N] [--anchor-only]\n", argv[0]);
        return 1;
    }
    std::string photfile = argv[1], infile = argv[2], outfile = argv[3];
    int first = 0, count = -1, device = 0;
    bool anchor_only = false;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--cells") && i + 1 < argc) { sscanf(argv[++i], "%d:%d", &first, &count); }
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) { device = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--anchor-only")) { anchor_only = true; }
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
    }

    KampylosWuInput in;
    {
        FILE* f = fopen(infile.c_str(), "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", infile.c_str()); return 1; }
        std::string err;
        bool ok = kampylos_read_input_file(f, &in, &err);
        fclose(f);
        if (!ok) { fprintf(stderr, "malformed input file: %s\n", err.c_str()); return 1; }
    }
    std::vector<DataPoint> data;
    try {
        data = load_photometry(photfile, in.is_mag);
    } catch (const std::exception& e) {
        fprintf(stderr, "error loading %s: %s\n", photfile.c_str(), e.what());
        return 1;
    }
    if (data.size() < 20) {
        fprintf(stderr, "error: only %zu usable data points (need at least 20)\n", data.size());
        return 1;
    }
    fprintf(stderr, "Loaded %zu data points from %s; grid %dx%d (%s)\n", data.size(), photfile.c_str(),
            in.n_s, in.n_q, kampylos_grid_name(in.log10_grid));

    auto t_start = std::chrono::steady_clock::now();
    KampylosWuAnchor anchor = kampylos_compute_anchor(data);
    double t_anchor = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    kampylos_log_anchor(stderr, anchor);
    fprintf(stderr, "anchor stage: %.2f s\n", t_anchor);

    FILE* out = fopen(outfile.c_str(), "w");
    if (!out) { fprintf(stderr, "cannot open %s for writing\n", outfile.c_str()); return 1; }
    kampylos_write_header(out, in, anchor, KAMPYLOS_CLI_BACKEND);
    fflush(out);
    if (anchor_only) { fclose(out); return 0; }

    VBMicrolensing vbm;
    int total = in.n_s * in.n_q;
    int last = (count < 0) ? total : std::min(total, first + count);

#if defined(KAMPYLOS_CLI_CUDA)
    std::string err;
    if (!GpuFitContext::select_device(device, &err)) { fprintf(stderr, "cudaSetDevice failed: %s\n", err.c_str()); return 1; }
    GpuFitContext ctx;
    gpu_upload_light_curve(ctx, data);
    auto fit_cell = [&](double ln_s, double ln_q, const BinarySeedSet& seeds) { return gpu_fit_binary_seeds(ctx, data, ln_s, ln_q, seeds); };
#elif defined(KAMPYLOS_CLI_OPENCL)
    OpenCLFitContext ctx;
    std::string err;
    if (!opencl_fit_select_device(ctx, device, &err)) { fprintf(stderr, "opencl_fit_select_device failed: %s\n", err.c_str()); return 1; }
    opencl_fit_upload_light_curve(ctx, data);
    auto fit_cell = [&](double ln_s, double ln_q, const BinarySeedSet& seeds) { return opencl_fit_binary_seeds(ctx, data, ln_s, ln_q, seeds); };
#else
    (void)device;
    auto fit_cell = [&](double ln_s, double ln_q, const BinarySeedSet& seeds) { return fit_binary_seeds(vbm, data, ln_s, ln_q, seeds); };
#endif

    auto t_cells = std::chrono::steady_clock::now();
    auto t_prev = t_cells;
    kampylos_run_cells(out, stderr, in, data, anchor, vbm, first, last, fit_cell, [&](int done, int tot) {
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "  cell %d/%d took %.1f s\n", done, tot, std::chrono::duration<double>(now - t_prev).count());
        t_prev = now;
    });
    fclose(out);
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cells).count();
    fprintf(stderr, "Done: cells [%d,%d) written to %s in %.1f s (%.1f s/cell)\n", first, last, outfile.c_str(),
            dt, (last > first) ? dt / (last - first) : 0.0);
    return 0;
}
