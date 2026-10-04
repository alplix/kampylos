// OpenCL equivalent of gpu_fit_binary.cu -- same batched Nelder-Mead driving logic (see that
// file's own top comment for the full design rationale: NelderMeadStepper instances advanced in
// lockstep rounds, every active stepper's pending points batched into one kernel launch per
// round, kampylos_gpu_complete.h patching in the few per-candidate points that need the real
// VBMicrolensing finite-source treatment), just dispatched through cl_api instead of the CUDA
// runtime. Embedded kernel source (kampylos_cl_embedded.h) is the SAME kampylos_kernel.cl
// validated directly against the CUDA version and the CPU reference
// (tests/test_kampylos_opencl.cpp) -- this file only adds the OpenCL plumbing around it.
#include <cstdio>
#include <cstring>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <algorithm>
#include "kampylos_gpu_types.h"
#include "kampylos_gpu_complete.h"
#include "kampylos_cpu_patch_pool.h"
#include "kampylos_opencl_fit.h"
#include "nelder_mead_stepper.h"
#include "kampylos_cl_embedded.h"

OpenCLFitContext::~OpenCLFitContext() {
    if (d_data) cl_api.ReleaseMemObject(d_data);
    if (d_cand) cl_api.ReleaseMemObject(d_cand);
    if (d_res) cl_api.ReleaseMemObject(d_res);
    if (kernel) cl_api.ReleaseKernel(kernel);
    if (program) cl_api.ReleaseProgram(program);
    if (queue) cl_api.ReleaseCommandQueue(queue);
    if (context) cl_api.ReleaseContext(context);
}

bool opencl_fit_select_device(OpenCLFitContext& ctx, int device_id, std::string* err_out) {
    if (!cl_dynload_init()) {
        if (err_out) *err_out = "could not load an OpenCL ICD on this host";
        return false;
    }

    cl_uint n_platforms = 0;
    cl_api.GetPlatformIDs(0, NULL, &n_platforms);
    if (n_platforms == 0) {
        if (err_out) *err_out = "no OpenCL platforms found";
        return false;
    }
    std::vector<cl_platform_id> platforms(n_platforms);
    cl_api.GetPlatformIDs(n_platforms, platforms.data(), NULL);

    // Same robust platform scan as opencl_search.c's gpu_search_init() (real forum bug history,
    // see that file's own comment): don't just trust platform index 0 -- scan for the first
    // platform that actually reports a device.
    cl_platform_id platform = 0;
    cl_uint n_devices = 0;
    for (cl_uint i = 0; i < n_platforms; i++) {
        cl_uint n = 0;
        if (cl_api.GetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &n) == CL_SUCCESS && n > 0) {
            platform = platforms[i];
            n_devices = n;
            break;
        }
    }
    if (!platform) {
        if (err_out) *err_out = "no OpenCL devices found on any platform";
        return false;
    }
    std::vector<cl_device_id> devices(n_devices);
    cl_api.GetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, n_devices, devices.data(), NULL);
    if (device_id < 0 || (cl_uint)device_id >= n_devices) device_id = 0;
    ctx.device = devices[device_id];

    char name[256] = {0};
    cl_api.GetDeviceInfo(ctx.device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    fprintf(stderr, "opencl_fit_select_device: using OpenCL device \"%s\"\n", name);

    cl_int err;
    ctx.context = cl_api.CreateContext(NULL, 1, &ctx.device, NULL, NULL, &err);
    if (!ctx.context) { if (err_out) *err_out = "clCreateContext failed"; return false; }
    ctx.queue = cl_api.CreateCommandQueue(ctx.context, ctx.device, 0, &err);
    if (!ctx.queue) { if (err_out) *err_out = "clCreateCommandQueue failed"; return false; }

    const char* sources[2] = { kKampylosClTypesSource, kKampylosClKernelSource };
    size_t lengths[2] = { strlen(kKampylosClTypesSource), strlen(kKampylosClKernelSource) };
    ctx.program = cl_api.CreateProgramWithSource(ctx.context, 2, sources, lengths, &err);
    if (!ctx.program) { if (err_out) *err_out = "clCreateProgramWithSource failed"; return false; }
    err = cl_api.BuildProgram(ctx.program, 1, &ctx.device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        cl_api.GetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size + 1);
        cl_api.GetProgramBuildInfo(ctx.program, ctx.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        log[log_size] = 0;
        fprintf(stderr, "opencl_fit_select_device: kernel build failed:\n%s\n", log.data());
        if (err_out) *err_out = "kernel build failed";
        return false;
    }
    ctx.kernel = cl_api.CreateKernel(ctx.program, "kampylos_eval_candidates", &err);
    if (!ctx.kernel) { if (err_out) *err_out = "clCreateKernel failed"; return false; }
    return true;
}

void opencl_fit_upload_light_curve(OpenCLFitContext& ctx, const std::vector<DataPoint>& data) {
    std::vector<LightCurvePoint> h_data(data.size());
    for (size_t i = 0; i < data.size(); i++) h_data[i] = { data[i].t, data[i].flux, data[i].sigma };
    ctx.n_points = (int)h_data.size();
    if (ctx.d_data) cl_api.ReleaseMemObject(ctx.d_data);
    cl_int err;
    ctx.d_data = cl_api.CreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        h_data.size() * sizeof(LightCurvePoint), h_data.data(), &err);
}

static void ensure_cand_capacity(OpenCLFitContext& ctx, int n) {
    if (n <= ctx.cand_capacity) return;
    if (ctx.d_cand) cl_api.ReleaseMemObject(ctx.d_cand);
    if (ctx.d_res) cl_api.ReleaseMemObject(ctx.d_res);
    cl_int err;
    ctx.d_cand = cl_api.CreateBuffer(ctx.context, CL_MEM_READ_ONLY, n * sizeof(Candidate), NULL, &err);
    ctx.d_res = cl_api.CreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, n * sizeof(CandidateResult), NULL, &err);
    ctx.cand_capacity = n;
}

static std::vector<double> opencl_eval_batch(
    OpenCLFitContext& ctx, VBMicrolensing& vbm,
    double log_s, double log_q, double s, double q,
    const std::vector<LightCurvePoint>& h_data,
    const std::vector<Candidate>& points
) {
    int n = (int)points.size();
    std::vector<double> chi2(n);
    if (n == 0) return chi2;

    ensure_cand_capacity(ctx, n);
    cl_api.EnqueueWriteBuffer(ctx.queue, ctx.d_cand, CL_TRUE, 0, n * sizeof(Candidate), points.data(), 0, NULL, NULL);

    int local_size = 128;
    size_t sh_bytes = 7 * local_size * sizeof(double);
    cl_api.SetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.d_data);
    cl_api.SetKernelArg(ctx.kernel, 1, sizeof(int), &ctx.n_points);
    cl_api.SetKernelArg(ctx.kernel, 2, sizeof(double), &log_s);
    cl_api.SetKernelArg(ctx.kernel, 3, sizeof(double), &log_q);
    cl_api.SetKernelArg(ctx.kernel, 4, sizeof(cl_mem), &ctx.d_cand);
    cl_api.SetKernelArg(ctx.kernel, 5, sizeof(int), &n);
    cl_api.SetKernelArg(ctx.kernel, 6, sizeof(cl_mem), &ctx.d_res);
    cl_api.SetKernelArg(ctx.kernel, 7, sh_bytes, NULL);
    cl_api.SetKernelArg(ctx.kernel, 8, sizeof(int), NULL);
    cl_api.SetKernelArg(ctx.kernel, 9, sizeof(int), NULL);

    size_t global_size = (size_t)n * local_size;
    size_t local_size_t = (size_t)local_size;
    cl_api.EnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, NULL, &global_size, &local_size_t, 0, NULL, NULL);
    cl_api.Finish(ctx.queue);

    std::vector<CandidateResult> h_res(n);
    cl_api.EnqueueReadBuffer(ctx.queue, ctx.d_res, CL_TRUE, 0, n * sizeof(CandidateResult), h_res.data(), 0, NULL, NULL);

    // Same fix as gpu_fit_binary.cu's own gpu_eval_batch() (see that file and
    // kampylos_cpu_patch_pool.h for the full story) -- this loop was the same single-threaded
    // bottleneck for a pathological anchor, now spread across a small worker-thread pool instead.
    (void)vbm;
    static KampylosCpuPatchPool patch_pool(4);
    patch_pool.complete_batch(h_res, points, h_data.data(), s, q, chi2);
    return chi2;
}

static Candidate to_candidate(const std::vector<double>& x) {
    Candidate c;
    c.t0 = x[0]; c.u0 = x[1]; c.log_tE = x[2]; c.alpha = x[3]; c.log_rho = x[4];
    return c;
}

static std::vector<SimplexResultLite> opencl_batched_nelder_mead(
    OpenCLFitContext& ctx, VBMicrolensing& vbm,
    double log_s, double log_q, double s, double q,
    const std::vector<LightCurvePoint>& h_data,
    const std::vector<std::vector<double>>& x0s,
    const std::vector<std::vector<double>>& steps,
    int max_iter, double ftol, double xtol
) {
    int n_seeds = (int)x0s.size();
    std::vector<NelderMeadStepper> steppers;
    steppers.reserve(n_seeds);
    for (int i = 0; i < n_seeds; i++) steppers.emplace_back(x0s[i], steps[i], max_iter, ftol, xtol);

    for (;;) {
        std::vector<Candidate> batch;
        std::vector<int> pending_count(n_seeds, 0);
        bool any_active = false;
        for (int s_i = 0; s_i < n_seeds; s_i++) {
            if (steppers[s_i].done()) continue;
            any_active = true;
            const auto& pts = steppers[s_i].pending_points();
            pending_count[s_i] = (int)pts.size();
            for (const auto& p : pts) batch.push_back(to_candidate(p));
        }
        if (!any_active) break;

        std::vector<double> chi2 = opencl_eval_batch(ctx, vbm, log_s, log_q, s, q, h_data, batch);

        int offset = 0;
        for (int s_i = 0; s_i < n_seeds; s_i++) {
            if (pending_count[s_i] == 0) continue;
            std::vector<double> sub(chi2.begin() + offset, chi2.begin() + offset + pending_count[s_i]);
            offset += pending_count[s_i];
            steppers[s_i].submit_results(sub);
        }
    }

    std::vector<SimplexResultLite> out;
    out.reserve(n_seeds);
    for (auto& st : steppers) out.push_back(st.result());
    return out;
}

BinaryFitResult opencl_fit_binary_multistart(
    OpenCLFitContext& ctx,
    VBMicrolensing& vbm,
    const std::vector<DataPoint>& data,
    double log_s, double log_q,
    double t0_anchor, double u0_anchor, double tE_anchor,
    double rho_seed,
    int n_alpha_seeds,
    int n_restarts
) {
    std::vector<LightCurvePoint> h_data(data.size());
    for (size_t i = 0; i < data.size(); i++) h_data[i] = { data[i].t, data[i].flux, data[i].sigma };
    double s = exp(log_s), q = exp(log_q);

    const double rho_seeds[] = { rho_seed, 1e-4, 1e-2 };
    const double u0_mag_seeds[] = { fabs(u0_anchor), 0.02 };

    std::vector<std::vector<double>> x0s, steps;
    for (int i = 0; i < n_alpha_seeds; i++) {
        double alpha_seed = 2.0 * M_PI * i / n_alpha_seeds;
        for (double u0_mag : u0_mag_seeds) {
            for (double u0_sign : { 1.0, -1.0 }) {
                for (double rs : rho_seeds) {
                    x0s.push_back({ t0_anchor, u0_sign * u0_mag, log(tE_anchor), alpha_seed, log(rs) });
                    steps.push_back({ std::max(0.5, tE_anchor * 0.05), 0.05, 0.2, 0.3, 0.5 });
                }
            }
        }
    }

    auto pass1 = opencl_batched_nelder_mead(ctx, vbm, log_s, log_q, s, q, h_data, x0s, steps, 400, 1e-9, 1e-8);

    std::vector<SimplexResultLite> current = pass1;
    for (int restart = 0; restart < n_restarts; restart++) {
        std::vector<std::vector<double>> rx0, rsteps;
        for (auto& r : current) {
            rx0.push_back(r.x);
            rsteps.push_back({ 0.3, 0.02, 0.08, 0.15, 0.2 });
        }
        auto next = opencl_batched_nelder_mead(ctx, vbm, log_s, log_q, s, q, h_data, rx0, rsteps, 400, 1e-10, 1e-9);
        for (size_t i = 0; i < current.size(); i++) {
            if (next[i].fval < current[i].fval) current[i] = next[i];
        }
    }

    int best = 0;
    for (size_t i = 1; i < current.size(); i++) if (current[i].fval < current[best].fval) best = (int)i;

    BinaryFitResult out;
    out.log_s = log_s; out.log_q = log_q;
    out.t0 = current[best].x[0]; out.u0 = current[best].x[1]; out.tE = exp(current[best].x[2]);
    out.alpha = current[best].x[3]; out.rho = exp(current[best].x[4]);
    out.chi2 = current[best].fval;

    std::vector<double> mag(data.size());
    double pr[7] = { log_s, log_q, out.u0, out.alpha, log(out.rho), log(out.tE), out.t0 };
    for (size_t i = 0; i < data.size(); i++) mag[i] = vbm.BinaryLightCurve(pr, data[i].t);
    FluxFit ff = linear_flux_fit(data, mag);
    out.fs = ff.fs; out.fb = ff.fb;
    return out;
}
