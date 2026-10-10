// Kampylos, Metal (Apple GPU) variant -- same structure as main_kampylos_gpu_boinc.cpp (CUDA) /
// main_kampylos_opencl_boinc.cpp (OpenCL), calling metal_fit_binary_seeds() per grid cell
// instead. Plain C++ (not Objective-C++) -- kampylos_metal_fit.h hides every Metal/Obj-C detail
// behind an opaque MetalFitContext, same convention as keraunos_metal_search.h/main_jump_metal_boinc.cpp.
//
// No "--device N" parsing (unlike the CUDA/OpenCL wrappers): MTLCreateSystemDefaultDevice() is
// the right call on every real Mac this would run on (exactly one GPU on Apple Silicon; the
// right default on a discrete-GPU Intel Mac too) -- same convention as metal_search.mm's own
// gpu_search_init() for Potamos/Keraunos's Metal builds.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Same header-ordering discipline as the other Kampylos BOINC wrappers: Kampylos's own headers
// before boinc_api.h/filesys.h.
#include "../src/kampylos_wu.h"
#include "../src/kampylos_metal_fit.h"

#include "kampylos_boinc_common.h"

#define KAMPYLOS_METAL_VERSION "2"

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

    // Same ordering discipline as the other GPU wrappers in this project: parse and validate
    // every input file before touching the GPU at all, so a bad invocation fails before any
    // Metal device/pipeline gets created on this or any machine.
    KampylosBoincJob job;
    if (kampylos_boinc_load_inputs(job)) { boinc_finish(1); return 1; }
    kampylos_boinc_compute_anchor(job);

    // Metal setup, now that every input has validated -- device/pipeline creation is the first
    // real Metal call (same spot every other GPU app in this project does its own device init),
    // then one light-curve upload reused for every grid cell below.
    MetalFitContext metal_ctx;
    std::string metal_err;
    if (!metal_fit_select_device(metal_ctx, &metal_err)) {
        fprintf(stderr, "metal_fit_select_device failed: %s\n", metal_err.c_str());
        boinc_finish(1);
        return 1;
    }
    metal_fit_upload_light_curve(metal_ctx, job.data);

    if (kampylos_boinc_open_output(job, "metal")) { boinc_finish(1); return 1; }

    VBMicrolensing vbm;
    return kampylos_boinc_run(job, vbm, [&](double ln_s, double ln_q, const BinarySeedSet &seeds) {
        return metal_fit_binary_seeds(metal_ctx, job.data, ln_s, ln_q, seeds);
    });
}
