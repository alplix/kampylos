#pragma once
// Plain declarations for gpu_fit_binary.cu's public interface -- included by
// main_kampylos_gpu_boinc.cpp, which must NOT itself be compiled as CUDA source (it has no
// kernel launches or __global__ code of its own). Same project convention as
// main_jump_gpu_boinc.cpp/keraunos_gpu_search.cu: the .cpp wrapper and the .cu kernel file are
// passed to nvcc as separate inputs on the same command line (each compiled according to its own
// extension, then linked), not one #including the other -- #include-ing gpu_fit_binary.cu
// directly from a .cpp file was tried first and failed exactly this way (nvcc only applies CUDA
// compilation to .cu-named translation units; a .cpp that pulls in __global__/<<<>>> syntax via
// #include gets routed to the plain host compiler, which doesn't understand that syntax at all).
//
// GpuFitContext's own methods (cudaMalloc/cudaMemcpy/cudaFree/cudaSetDevice) are genuine CUDA
// RUNTIME API calls, not CUDA LANGUAGE EXTENSIONS -- those are safe to call from plain,
// nvcc-independent host C++ as long as cuda_runtime.h is included and the binary links against
// the CUDA runtime, so the struct itself is declared here rather than needing its own .cu.
#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <vector>
#include "kampylos_gpu_types.h"
#include "binary_fit.h" // BinaryFitResult
#include "lightcurve.h" // DataPoint

struct GpuFitContext {
    LightCurvePoint* d_data = nullptr;
    int n_points = 0;
    int data_capacity = 0;
    Candidate* d_cand = nullptr;
    CandidateResult* d_res = nullptr;
    int cand_capacity = 0;

    static bool select_device(int device_id, std::string* err_out = nullptr) {
        cudaError_t cerr = cudaSetDevice(device_id);
        if (cerr != cudaSuccess) {
            if (err_out) *err_out = cudaGetErrorString(cerr);
            return false;
        }
        return true;
    }

    static void check(cudaError_t e, const char* what) {
        if (e != cudaSuccess) {
            fprintf(stderr, "GpuFitContext: %s failed: %s\n", what, cudaGetErrorString(e));
        }
    }

    void ensure_data_capacity(int n) {
        if (n <= data_capacity) return;
        if (d_data) cudaFree(d_data);
        check(cudaMalloc(&d_data, n * sizeof(LightCurvePoint)), "cudaMalloc(d_data)");
        data_capacity = n;
    }
    void ensure_cand_capacity(int n) {
        if (n <= cand_capacity) return;
        if (d_cand) cudaFree(d_cand);
        if (d_res) cudaFree(d_res);
        check(cudaMalloc(&d_cand, n * sizeof(Candidate)), "cudaMalloc(d_cand)");
        check(cudaMalloc(&d_res, n * sizeof(CandidateResult)), "cudaMalloc(d_res)");
        cand_capacity = n;
    }
    void upload_light_curve(const std::vector<LightCurvePoint>& data) {
        ensure_data_capacity((int)data.size());
        n_points = (int)data.size();
        check(cudaMemcpy(d_data, data.data(), n_points * sizeof(LightCurvePoint), cudaMemcpyHostToDevice), "cudaMemcpy(d_data)");
    }
    ~GpuFitContext() {
        if (d_data) cudaFree(d_data);
        if (d_cand) cudaFree(d_cand);
        if (d_res) cudaFree(d_res);
    }
};

// VBMicrolensing's full definition already arrived via binary_fit.h's own #include chain above --
// no separate pull-in or forward declare needed here.

void gpu_upload_light_curve(GpuFitContext& ctx, const std::vector<DataPoint>& data);

BinaryFitResult gpu_fit_binary_multistart(
    GpuFitContext& ctx,
    VBMicrolensing& vbm,
    const std::vector<DataPoint>& data,
    double log_s, double log_q,
    double t0_anchor, double u0_anchor, double tE_anchor,
    double rho_seed = 1e-3,
    int n_alpha_seeds = 8,
    int n_restarts = 2
);
