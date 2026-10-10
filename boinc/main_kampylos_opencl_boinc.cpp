// Kampylos, OpenCL variant -- same structure as main_kampylos_gpu_boinc.cpp (CUDA), same
// checkpoint granularity (one grid cell, see main_kampylos_boinc.cpp's own top comment for why),
// same input/output file conventions. The only real difference from the CUDA wrapper is which
// function does the per-cell fitting work: opencl_fit_binary_seeds()
// (vendor/kampylos/src/kampylos_opencl_fit.cpp) instead of gpu_fit_binary_seeds() -- same
// seed set, same restart logic, same batched-Nelder-Mead driver (kampylos_batched_fit.h); the
// per-cell chi2/fs/fb and diagnostics are recomputed on the host by the shared code. OpenCL covers AMD/Intel GPUs and any
// NVIDIA card a volunteer prefers not to run the CUDA build on, same role this project's other
// apps' OpenCL variants already play (opencl_kangaroo.c for Keraunos, potamos_opencl_search.c
// for Potamos).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Same header-ordering discipline as main_kampylos_boinc.cpp/main_kampylos_gpu_boinc.cpp:
// Kampylos's own headers before boinc_api.h/filesys.h, to avoid windows.h's macro pollution
// colliding with VBMicrolensingLibrary.h's local variable names.
#include "../src/kampylos_wu.h"
#include "../src/kampylos_opencl_fit.h"

#include "kampylos_boinc_common.h"

#define KAMPYLOS_OPENCL_VERSION "2"

// BOINC hands GPU app_versions the device index via "--device N" on the command line -- same
// convention as every GPU app in this project.
static int parse_device_arg(int argc, char **argv) {
    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--device")) {
            return atoi(argv[i + 1]);
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    fprintf(stderr, "Kampylos OpenCL (gravitational microlensing binary-lens grid search) v" KAMPYLOS_OPENCL_VERSION "\n");
    fprintf(stderr, "bitboinc -- https://bitboinc.athena.org.tr/\n");
    fprintf(stderr, "coded by Alperen Yavuz\n");

    int device_id = parse_device_arg(argc, argv);

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

    // Real bug, forum report 2026-10-05 (main BOINC forum, "Only one GPU being used but
    // BOINC says one task is on GPU0 and the other on GPU1"): the client's own
    // app_start.cpp marks the "--device N" command-line convention DEPRECATED in favor of
    // app_init_data, specifically because N (gpu_device_num, the vendor/ADL-level index
    // BOINC itself uses) is NOT always the same number OpenCL's own clGetDeviceIDs()
    // enumeration uses for that device -- app_init_data carries a SEPARATE field,
    // gpu_opencl_dev_index, computed for exactly this purpose. We only ever read --device,
    // so on hosts where the two indices diverge (seen live: an AMD discrete+integrated
    // pair) every OpenCL task picked the SAME (wrong) device regardless of which GPU BOINC
    // intended. Overrides device_id with the correct index when init_data has one;
    // --device remains as a fallback for standalone testing outside BOINC.
    APP_INIT_DATA aid;
    if (!boinc_get_init_data(aid) && aid.gpu_opencl_dev_index >= 0) {
        fprintf(stderr, "main_kampylos_opencl: overriding device index %d -> %d from app_init_data (gpu_opencl_dev_index)\n",
                device_id, aid.gpu_opencl_dev_index);
        device_id = aid.gpu_opencl_dev_index;
    }

    // Same ordering discipline as the other GPU wrappers in this project: parse and validate
    // every input file before touching the GPU at all, so a bad invocation fails before any
    // OpenCL context gets created on this or any machine.
    KampylosBoincJob job;
    if (kampylos_boinc_load_inputs(job)) { boinc_finish(1); return 1; }
    kampylos_boinc_compute_anchor(job);

    // OpenCL setup, now that every input has validated -- device/platform selection and the
    // kernel build are the first real OpenCL calls (same spot every other GPU app in this
    // project does its own device init), then one light-curve upload reused for every grid cell
    // below.
    OpenCLFitContext opencl_ctx;
    std::string cl_err;
    if (!opencl_fit_select_device(opencl_ctx, device_id, &cl_err)) {
        fprintf(stderr, "opencl_fit_select_device(%d) failed: %s\n", device_id, cl_err.c_str());
        boinc_finish(1);
        return 1;
    }
    opencl_fit_upload_light_curve(opencl_ctx, job.data);

    if (kampylos_boinc_open_output(job, "opencl")) { boinc_finish(1); return 1; }

    VBMicrolensing vbm;
    return kampylos_boinc_run(job, vbm, [&](double ln_s, double ln_q, const BinarySeedSet &seeds) {
        return opencl_fit_binary_seeds(opencl_ctx, job.data, ln_s, ln_q, seeds);
    });
}
