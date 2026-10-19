// Kampylos, Metal (Apple GPU) variant -- same structure as main_kampylos_gpu_boinc.cpp (CUDA) /
// main_kampylos_opencl_boinc.cpp (OpenCL), calling metal_fit_binary_seeds() per grid cell
// instead. Plain C++ (not Objective-C++) -- kampylos_metal_fit.h hides every Metal/Obj-C detail
// behind an opaque MetalFitContext, same convention as keraunos_metal_search.h/main_jump_metal_boinc.cpp.
//
// No "--device N" parsing (unlike the CUDA/OpenCL wrappers): MTLCreateSystemDefaultDevice() is
// the right call on every real Mac this would run on (exactly one GPU on Apple Silicon; the
// right default on a discrete-GPU Intel Mac too) -- same convention as metal_search.mm's own
// gpu_search_init() for Potamos/Keraunos's Metal builds -- with MTLCopyAllDevices() as the
// fallback for a session without a window server (see metal_fit_select_device).
//
// A GPU that is unavailable right now (no device visible, or command buffers that keep being
// aborted because another process hogs the GPU) is NOT a reason to fail the work unit: those
// paths end in boinc_temporary_exit(), which leaves the task and its checkpoints intact and has
// the client start it again after a delay (it gives up only after 100 such exits with no
// checkpoint in between). Build and input problems still end in boinc_finish(1).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Same header-ordering discipline as the other Kampylos BOINC wrappers: Kampylos's own headers
// before boinc_api.h/filesys.h.
#include "../src/kampylos_wu.h"
#include "../src/kampylos_metal_fit.h"
#include "../src/kampylos_cpu_patch_pool.h"   // kampylos_set_patch_threads()

#include "kampylos_boinc_common.h"

// Build with -DKAMPYLOS_METAL_VERSION='"N"' so the printed version matches the server's version_num.
#ifndef KAMPYLOS_METAL_VERSION
#define KAMPYLOS_METAL_VERSION "2"
#endif

// Seconds the client waits before restarting the task after a boinc_temporary_exit() (BOINC's
// own default when an app gives none is 300).
#define KAMPYLOS_METAL_RETRY_DELAY_S 300

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    fprintf(stderr, "Kampylos Metal (gravitational microlensing binary-lens grid search, Apple GPU) v" KAMPYLOS_METAL_VERSION "\n");
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

    // The completion step's CPU threads (src/kampylos_cpu_patch_pool.h) follow the client's CPU
    // budget for this task, see kampylos_boinc_cpu_threads(). Until 2026-10-19 the pool took
    // four cores whatever the metal_apple_gpu plan class declared (0.1).
    APP_INIT_DATA aid;
    if (boinc_get_init_data(aid) == 0) {
        int n = kampylos_set_patch_threads(kampylos_boinc_cpu_threads(aid));
        fprintf(stderr, "completion step: %d CPU thread(s) (client budget %.2f CPUs)\n", n, aid.ncpus);
    }

    // Same ordering discipline as the other GPU wrappers in this project: parse and validate
    // every input file before touching the GPU at all, so a bad invocation fails before any
    // Metal device/pipeline gets created on this or any machine.
    KampylosBoincJob job;
    if (kampylos_boinc_load_inputs(job, true)) { boinc_finish(1); return 1; }   // seed-split WUs too, see below
    kampylos_boinc_compute_anchor(job);

    // Metal setup, now that every input has validated -- device/pipeline creation is the first
    // real Metal call (same spot every other GPU app in this project does its own device init),
    // then one light-curve upload reused for every grid cell below.
    MetalFitContext metal_ctx;
    std::string metal_err;
    if (!metal_fit_select_device(metal_ctx, &metal_err)) {
        fprintf(stderr, "metal_fit_select_device failed: %s\n", metal_err.c_str());
        if (metal_fit_setup_failure_is_transient(metal_ctx)) {
            fprintf(stderr, "no usable Metal device at the moment -- temporary exit, the client will retry in %d s\n", KAMPYLOS_METAL_RETRY_DELAY_S);
            boinc_temporary_exit(KAMPYLOS_METAL_RETRY_DELAY_S, "Metal device unavailable, will retry", false);
        }
        boinc_finish(1);
        return 1;
    }
    metal_fit_upload_light_curve(metal_ctx, job.data);

    if (kampylos_boinc_open_output(job, "metal")) { boinc_finish(1); return 1; }

    VBMicrolensing vbm;
    int cell = job.start_cell;   // cells are fitted in order, one call per cell
    return kampylos_boinc_run(job, vbm, [&](double ln_s, double ln_q, const BinarySeedSet &seeds) {
        // in-cell checkpoint and progress, see kampylos_boinc_batched_hooks(); a seed-split work
        // unit ("part i K") gets this part's seeds only, as in the CPU app (kampylos_seed_part)
        KampylosBatchedHooks hooks = kampylos_boinc_batched_hooks(cell, job.total);
        BinarySeedSet own = kampylos_seed_part(seeds, job.in.part, job.in.nparts);
        try {
            BinaryFitResult r = metal_fit_binary_seeds(metal_ctx, job.data, ln_s, ln_q, own, 2, &hooks, cell);
            cell++;
            return r;
        } catch (const std::exception &e) {
            // A command buffer kept failing even after re-submission (another process is
            // monopolising the GPU, or the device was lost): stop rather than write a cell
            // computed from stale results -- as a temporary exit, not an error: the finished
            // cells are in "out", the in-cell checkpoint is on disk, and the client restarts the
            // task from them after the delay. (Until 2026-10-12 this was boinc_finish(1), which
            // reported the work unit as failed and threw its progress away.)
            fprintf(stderr, "metal_fit_binary_seeds failed: %s -- temporary exit, the client will retry in %d s\n", e.what(), KAMPYLOS_METAL_RETRY_DELAY_S);
            boinc_temporary_exit(KAMPYLOS_METAL_RETRY_DELAY_S, "Metal command buffers keep failing, will retry", false);
            return BinaryFitResult{};
        }
    });
}
