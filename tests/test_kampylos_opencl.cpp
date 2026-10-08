// Validates the OpenCL port (kampylos_kernel.cl) against the real CPU reference, same two checks
// as test_kampylos_gpu_kernel.cu's CUDA version: kernel mechanics (reduction/indexing match a
// host computation using the identical struct layouts) and needs_fallback correctness (cross-
// checked against the real VBMicrolensing BinaryMag2/BinaryMag0). Reuses kampylos_gpu_complete.h
// unchanged -- CandidateResult/Candidate/LightCurvePoint are the same structs on both the CUDA
// and OpenCL sides (kampylos_gpu_types.h), so the same CPU-side fallback-patching code works for
// either backend's kernel output.
//
// Build (Windows, using the CUDA toolkit's bundled Khronos headers + import lib for local
// iteration -- production code loads OpenCL at runtime via cl_dynload.c instead, see that file's
// own top comment for why):
//   cl /EHsc /std:c++17 /O2 ^
//      /I "..\src" /I "..\vendor\VBMicrolensing\VBMicrolensing\lib" ^
//      /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\include" ^
//      test_kampylos_opencl.cpp ..\vendor\VBMicrolensing\VBMicrolensing\lib\VBMicrolensingLibrary.cpp ^
//      /link "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\lib\x64\OpenCL.lib" ^
//      /out:test_kampylos_opencl.exe
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include "kampylos_gpu_types.h"
#include "kampylos_gpu_complete.h"
#include "VBMicrolensingLibrary.h"

static std::string read_file(const char* path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

#define CL_CHECK(call) do { \
    cl_int _e = (call); \
    if (_e != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL error %s:%d: code %d\n", __FILE__, __LINE__, (int)_e); \
        exit(1); \
    } \
} while (0)

int main() {
    // --- OpenCL setup: first platform/device with a GPU, same spirit as the project's own
    // opencl_search.c (minus the production dynamic-loading/fallback-scanning, not needed for a
    // local correctness test against a known GPU). ---
    cl_uint n_platforms = 0;
    CL_CHECK(clGetPlatformIDs(0, NULL, &n_platforms));
    if (n_platforms == 0) { fprintf(stderr, "no OpenCL platforms\n"); return 1; }
    std::vector<cl_platform_id> platforms(n_platforms);
    CL_CHECK(clGetPlatformIDs(n_platforms, platforms.data(), NULL));

    cl_device_id device = nullptr;
    cl_platform_id chosen_platform = nullptr;
    for (auto p : platforms) {
        cl_uint n_dev = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, NULL, &n_dev) == CL_SUCCESS && n_dev > 0) {
            std::vector<cl_device_id> devs(n_dev);
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, n_dev, devs.data(), NULL);
            device = devs[0];
            chosen_platform = p;
            break;
        }
    }
    if (!device) { fprintf(stderr, "no OpenCL GPU device found\n"); return 1; }

    char dev_name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("OpenCL device: %s\n", dev_name);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK(err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    CL_CHECK(err);

    std::string types_src = read_file("../src/kampylos_gpu_types.h");
    std::string kernel_src = read_file("../src/kampylos_kernel.cl");
    const char* sources[2] = { types_src.c_str(), kernel_src.c_str() };
    size_t lengths[2] = { types_src.size(), kernel_src.size() };
    cl_program program = clCreateProgramWithSource(ctx, 2, sources, lengths, &err);
    CL_CHECK(err);
    cl_int build_err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (build_err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        fprintf(stderr, "Build failed:\n%s\n", log.data());
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "kampylos_eval_candidates", &err);
    CL_CHECK(err);

    // --- Same synthetic light curve as the CUDA kernel test ---
    const double true_log_s = 0.05, true_log_q = -2.3;
    const double true_t0 = 750.0, true_u0 = 0.08, true_tE = 25.0, true_alpha = 1.2, true_rho = 0.01;
    const double true_fs = 1.0, true_fb = 0.2;

    VBMicrolensing vbm_ref;
    vbm_ref.Tol = 1.e-2;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> cadence(0.2, 1.5);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::vector<LightCurvePoint> h_data;
    double t = 0.0;
    while (t < 1500.0) {
        t += cadence(rng);
        double pr[7] = { true_log_s, true_log_q, true_u0, true_alpha, log(true_rho), log(true_tE), true_t0 };
        double mag = vbm_ref.BinaryLightCurve(pr, t);
        double sigma = 0.02;
        LightCurvePoint p; p.t = t; p.flux = true_fs * mag + true_fb + noise(rng) * sigma; p.sigma = sigma;
        h_data.push_back(p);
    }
    int n_points = (int)h_data.size();
    printf("Synthetic light curve: %d points\n", n_points);

    std::vector<Candidate> h_cand;
    { Candidate c; c.t0 = true_t0; c.u0 = true_u0; c.log_tE = log(true_tE); c.alpha = true_alpha; c.log_rho = log(true_rho); h_cand.push_back(c); }
    std::uniform_real_distribution<double> t0_dist(740, 760), u0_dist(-0.3, 0.3), alpha_dist(0, 6.28);
    std::uniform_real_distribution<double> logtE_dist(log(10.0), log(50.0)), logrho_dist(-4, -2.3);
    for (int i = 0; i < 2000; i++) {
        Candidate c; c.t0 = t0_dist(rng); c.u0 = u0_dist(rng); c.log_tE = logtE_dist(rng); c.alpha = alpha_dist(rng); c.log_rho = logrho_dist(rng);
        h_cand.push_back(c);
    }
    int n_cand = (int)h_cand.size();

    cl_mem d_data = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_points * sizeof(LightCurvePoint), h_data.data(), &err);
    CL_CHECK(err);
    cl_mem d_cand = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_cand * sizeof(Candidate), h_cand.data(), &err);
    CL_CHECK(err);
    cl_mem d_res = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n_cand * sizeof(CandidateResult), NULL, &err);
    CL_CHECK(err);

    int local_size = 128;
    size_t sh_bytes = 7 * local_size * sizeof(double);
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_data));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(int), &n_points));
    const double lin_s = exp(true_log_s), lin_q = exp(true_log_q); // kernel takes linear s, q
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(double), &lin_s));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(double), &lin_q));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_cand));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &n_cand));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_res));
    CL_CHECK(clSetKernelArg(kernel, 7, sh_bytes, NULL));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), NULL));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), NULL));

    size_t global_size = (size_t)n_cand * local_size;
    size_t local_size_t = (size_t)local_size;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size_t, 0, NULL, NULL));
    CL_CHECK(clFinish(queue));

    std::vector<CandidateResult> h_res(n_cand);
    CL_CHECK(clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0, n_cand * sizeof(CandidateResult), h_res.data(), 0, NULL, NULL));

    double s = exp(true_log_s), q = exp(true_log_q);
    int n_dead = 0, n_overflow = 0, n_with_bad = 0, mismatches = 0;
    long long total_bad = 0;
    int max_bad = 0;
    double worst_relerr = 0;

    for (int ci = 0; ci < n_cand; ci++) {
        CompletedFit cf = kampylos_complete_candidate(vbm_ref, h_res[ci], h_cand[ci], h_data.data(), s, q);
        if (h_res[ci].overflow) n_overflow++;
        if (cf.dead) { n_dead++; continue; }
        if (h_res[ci].n_bad > 0) n_with_bad++;
        total_bad += h_res[ci].n_bad;
        if (h_res[ci].n_bad > max_bad) max_bad = h_res[ci].n_bad;

        Candidate c = h_cand[ci];
        double tE = exp(c.log_tE), tE_inv = 1.0 / tE, rho = exp(c.log_rho);
        double salpha = sin(c.alpha), calpha = cos(c.alpha);
        double S_AA = 0, S_A1 = 0, S_11 = 0, S_Ay = 0, S_1y = 0, S_yy = 0;
        for (int i = 0; i < n_points; i++) {
            double tn = (h_data[i].t - c.t0) * tE_inv;
            double y1 = c.u0 * salpha - tn * calpha;
            double y2 = -c.u0 * calpha - tn * salpha;
            double A = vbm_ref.BinaryMag2(s, q, y1, y2, rho);
            double w = 1.0 / (h_data[i].sigma * h_data[i].sigma);
            double flux = h_data[i].flux;
            S_AA += w * A * A; S_A1 += w * A; S_11 += w; S_Ay += w * A * flux; S_1y += w * flux; S_yy += w * flux * flux;
        }
        double det = S_AA * S_11 - S_A1 * S_A1;
        double ref_fs = (S_11 * S_Ay - S_A1 * S_1y) / det;
        double ref_fb = (S_AA * S_1y - S_A1 * S_Ay) / det;
        double ref_chi2 = S_yy - ref_fs * S_Ay - ref_fb * S_1y;

        double relerr = fabs(cf.chi2 - ref_chi2) / fabs(ref_chi2);
        if (relerr > worst_relerr) worst_relerr = relerr;
        if (relerr > 1e-6) {
            mismatches++;
            if (mismatches <= 8) {
                printf("MISMATCH cand#%d (n_bad=%d): opencl_chi2=%.8f ref_chi2=%.8f relerr=%.3e\n",
                    ci, h_res[ci].n_bad, cf.chi2, ref_chi2, relerr);
            }
        }
    }

    printf("\n=== OpenCL kernel test: %d points, %d candidates ===\n", n_points, n_cand);
    printf("dead (poisoned/overflow): %d, overflow specifically: %d\n", n_dead, n_overflow);
    printf("candidates with >=1 bad point: %d / %d, total bad-point evals: %lld, max n_bad: %d\n",
        n_with_bad, n_cand - n_dead, total_bad, max_bad);
    printf("chi2 mismatches (relerr > 1e-6): %d, worst relerr %.3e\n", mismatches, worst_relerr);

    bool pass = (mismatches == 0) && (n_overflow == 0);
    printf("\n%s\n", pass ? "PASS" : "FAIL");

    clReleaseMemObject(d_data); clReleaseMemObject(d_cand); clReleaseMemObject(d_res);
    clReleaseKernel(kernel); clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(ctx);
    return pass ? 0 : 1;
}
