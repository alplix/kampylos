#pragma once
// Plain C++ interface to kampylos_metal_fit.mm (the Metal/Objective-C++ implementation) -- same
// shape as keraunos_metal_search.h: the BOINC wrapper (main_kampylos_metal_boinc.cpp) doesn't
// need to know anything about Metal/Objective-C++, just these functions and an opaque context.
// Same public shape as gpu_fit_binary.h (CUDA) / kampylos_opencl_fit.h (OpenCL) otherwise --
// MetalFitContext instead of GpuFitContext/OpenCLFitContext, metal_fit_binary_multistart instead
// of gpu_/opencl_fit_binary_multistart -- so all three BOINC wrappers read almost identically.
#include <string>
#include <vector>
#include "binary_fit.h" // BinaryFitResult
#include "lightcurve.h" // DataPoint

struct MetalFitContext {
    void* impl; // MetalFitImpl* (Objective-C++ types -- MTLDevice/MTLCommandQueue/... -- live
                 // only inside kampylos_metal_fit.mm, never in this plain-C++-consumable header)
    MetalFitContext();
    ~MetalFitContext();
};

// No device index parameter -- same convention as metal_search.mm's own gpu_search_init():
// MTLCreateSystemDefaultDevice() is the right call on every real Apple Silicon Mac (exactly one
// GPU), and a discrete-GPU Intel Mac's default is still the right one to pick for a volunteer
// host's main display GPU.
bool metal_fit_select_device(MetalFitContext& ctx, std::string* err_out);

void metal_fit_upload_light_curve(MetalFitContext& ctx, const std::vector<DataPoint>& data);

class VBMicrolensing;

// Metal twin of gpu_fit_binary_seeds() (see gpu_fit_binary.h): runs every seed of a
// kampylos_make_seeds() set at one grid cell, ln_s/ln_q natural logs; the work-unit driver
// recomputes chi2/fs/fb exactly on the host afterwards.
BinaryFitResult metal_fit_binary_seeds(
    MetalFitContext& ctx,
    const std::vector<DataPoint>& data,
    double ln_s, double ln_q,
    const BinarySeedSet& seeds,
    int n_restarts = 2
);

// Legacy entry point (unit tests): the 96-seed multistart around one anchor.
BinaryFitResult metal_fit_binary_multistart(
    MetalFitContext& ctx,
    VBMicrolensing& vbm,
    const std::vector<DataPoint>& data,
    double log_s, double log_q,
    double t0_anchor, double u0_anchor, double tE_anchor,
    double rho_seed = 1e-3,
    int n_alpha_seeds = 8,
    int n_restarts = 2
);
