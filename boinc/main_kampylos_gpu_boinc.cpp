// Kampylos, GPU (CUDA) variant -- same structure, same checkpoint granularity (one grid cell,
// see main_kampylos_boinc.cpp's own top comment for why), same input/output file conventions
// (all shared through kampylos_boinc_common.h / vendor/kampylos/src/kampylos_wu.h). The only
// real difference is which function does the fitting work per cell: gpu_fit_binary_seeds()
// (vendor/kampylos/src/gpu_fit_binary.cu) instead of fit_binary_seeds() -- the same seed set
// (kampylos_make_seeds), the same restart logic, every seed's evaluations batched through the GPU
// instead of run one at a time on the CPU. Each cell's chi2/fs/fb and diagnostics are then
// recomputed exactly on the host by the same code the CPU app uses.
//
// Device selection and the light-curve upload happen ONCE here, before the per-cell loop --
// gpu_fit_binary_seeds() takes an already-set-up GpuFitContext rather than creating one itself,
// specifically so a WU's cells don't each redo a device-buffer setup/light-curve upload that's
// identical across all of them.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Same header-ordering discipline as main_kampylos_boinc.cpp: Kampylos's own headers (and the
// VBMicrolensing/CUDA headers they pull in) before boinc_api.h/filesys.h, to avoid windows.h's
// macro pollution colliding with VBMicrolensingLibrary.h's local variable names.
#include "../src/kampylos_wu.h"
#include "../src/gpu_fit_binary.h"

#include "kampylos_boinc_common.h"

#define KAMPYLOS_GPU_VERSION "2"

// BOINC hands GPU app_versions the device index via "--device N" on the command line -- same
// convention as main_jump_gpu_boinc.cpp/main_range_gpu_boinc.cpp.
static int parse_device_arg(int argc, char **argv) {
    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--device")) {
            return atoi(argv[i + 1]);
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    fprintf(stderr, "Kampylos GPU (gravitational microlensing binary-lens grid search, CUDA) v" KAMPYLOS_GPU_VERSION "\n");
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

    // Prefer app_init_data's gpu_device_num over "--device N", 2026-10-05: see the OpenCL
    // wrappers' fuller comment (forum report: two GPU tasks always running on the same
    // physical device because BOINC's own client marks --device deprecated in favor of
    // app_init_data). --device remains as a fallback for standalone testing outside BOINC.
    APP_INIT_DATA aid;
    if (!boinc_get_init_data(aid) && aid.gpu_device_num >= 0) {
        device_id = aid.gpu_device_num;
    }

    // Same ordering discipline as the other GPU wrappers in this project: parse and validate
    // every input file before touching the GPU at all, so a bad invocation fails before any CUDA
    // context gets created on this or any machine.
    KampylosBoincJob job;
    if (kampylos_boinc_load_inputs(job)) { boinc_finish(1); return 1; }
    kampylos_boinc_compute_anchor(job);

    // GPU setup, now that every input has validated -- device selection is the first real CUDA
    // call (same spot every other GPU app in this project does it), then one light-curve upload
    // reused for every grid cell below.
    std::string gpu_err;
    if (!GpuFitContext::select_device(device_id, &gpu_err)) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", device_id, gpu_err.c_str());
        boinc_finish(1);
        return 1;
    }
    GpuFitContext gpu_ctx;
    gpu_upload_light_curve(gpu_ctx, job.data);

    if (kampylos_boinc_open_output(job, "cuda")) { boinc_finish(1); return 1; }

    VBMicrolensing vbm;
    return kampylos_boinc_run(job, vbm, [&](double ln_s, double ln_q, const BinarySeedSet &seeds) {
        return gpu_fit_binary_seeds(gpu_ctx, job.data, ln_s, ln_q, seeds);
    });
}
