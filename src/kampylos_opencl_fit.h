#pragma once
// Plain declarations for kampylos_opencl_fit.cpp's public interface -- OpenCL equivalent of
// gpu_fit_binary.h/.cu, same public shape (OpenCLFitContext instead of GpuFitContext,
// opencl_fit_binary_multistart instead of gpu_fit_binary_multistart) so main_kampylos_opencl_boinc.cpp
// reads almost identically to main_kampylos_gpu_boinc.cpp (the CUDA wrapper). Same reasoning as
// that header for including the runtime's own types directly rather than hiding them behind an
// opaque handle: GpuFitContext includes cuda_runtime.h and uses real CUDA types plainly, so this
// does the same with cl_dynload.h/OpenCL types.
#include <string>
#include <vector>
#include "cl_dynload.h" // cl_api, cl_context/cl_command_queue/... (includes <CL/cl.h>)
#include "binary_fit.h" // BinaryFitResult
#include "lightcurve.h" // DataPoint

struct OpenCLFitContext {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_device_id device = nullptr;

    cl_mem d_data = nullptr;
    int n_points = 0;
    cl_mem d_cand = nullptr;
    cl_mem d_res = nullptr;
    int cand_capacity = 0;

    ~OpenCLFitContext();
};

// Same "--device N" convention as every GPU app in this project, resolved against a platform the
// same robust way opencl_search.c's gpu_search_init() already does (scan every platform for one
// that actually has a device, don't just trust index 0 -- real forum bug history, see that file's
// own comment). Returns false (with *err_out set) on any failure: no OpenCL ICD on this host, no
// GPU device, or a kernel build failure.
bool opencl_fit_select_device(OpenCLFitContext& ctx, int device_id, std::string* err_out);

void opencl_fit_upload_light_curve(OpenCLFitContext& ctx, const std::vector<DataPoint>& data);

// VBMicrolensing's full definition already arrived via binary_fit.h's own #include chain above --
// no separate pull-in or forward declare needed here (same note as gpu_fit_binary.h).

BinaryFitResult opencl_fit_binary_multistart(
    OpenCLFitContext& ctx,
    VBMicrolensing& vbm,
    const std::vector<DataPoint>& data,
    double log_s, double log_q,
    double t0_anchor, double u0_anchor, double tE_anchor,
    double rho_seed = 1e-3,
    int n_alpha_seeds = 8,
    int n_restarts = 2
);
