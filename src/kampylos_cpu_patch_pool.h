#pragma once
// Parallelizes gpu_fit_binary.cu's per-round CPU "completion" step -- see gpu_eval_batch() in
// that file for how this is wired in.
//
// FIRST ATTEMPT (2026-10-04, earlier the same night) crashed on real hardware: "malloc():
// unaligned tcache chunk detected" / SIGSEGV, classic heap-corruption data race, despite giving
// each worker thread its own VBMicrolensing instance (which looked like it should have been
// enough -- no shared mutable state between threads, or so it seemed). Root cause: grepping
// VBMicrolensingLibrary.cpp found its magnification functions (BinaryMag2 and the contour-
// integration machinery under it, e.g. around lines 718-913) are full of function-LOCAL static
// scratch variables (static double Mag, static complex coefs[24], static _curve* Prov, etc. --
// 243 of them across the file). A function-local static is shared by EVERY call to that
// function from EVERY thread, regardless of which object instance made the call -- separate
// VBMicrolensing instances per thread does NOT make these reentrant, since the statics live at
// the process level, not the instance level.
//
// FIX (same night, after finding the above): every one of those 243 function-local statics in
// vendor/kampylos/vendor/VBMicrolensing/VBMicrolensing/lib/VBMicrolensingLibrary.{cpp,h} is now
// `thread_local static` instead of plain `static` -- each thread gets its own persistent copy,
// which both eliminates the cross-thread race AND preserves the library's own intentional
// cross-call caching within a single thread (e.g. BinaryMag0's cached polynomial coefficients
// for a repeated (s,q) pair, which legitimately needs to persist call-to-call, just not across
// threads). Re-wired into gpu_fit_binary.cu after this fix, pending a real-hardware re-test on
// the same pathological case to confirm both no crash AND numerically identical results to the
// pre-parallelization sequential version before this gets anywhere near production.
//
// Original problem this exists to solve, for context:
// Parallelizes gpu_fit_binary.cu's per-round CPU "completion" step: for every candidate the GPU
// kernel flagged as having bad points (needing a real VBMicrolensing-exact magnification, not the
// kernel's own cheap approximation), kampylos_complete_candidate() does that patching on the CPU.
// Before this, gpu_eval_batch() did this in a single-threaded loop over every candidate in the
// batch -- fine when each candidate has 0-1 bad points, but for a pathological anchor (e.g. tE far
// beyond the data's own span) a candidate can need MANY points patched, and that loop runs on
// EVERY round of the batched Nelder-Mead (up to ~1200 rounds per grid cell). Real-hardware test:
// one such cell took 91+ minutes on GPU, almost entirely in this loop -- the GPU kernel launch
// itself is not the bottleneck for a pathological anchor, this CPU-side cleanup is.
//
// VBMicrolensing is NOT safe to call concurrently from a single shared instance (BinaryMag2 etc.
// mutate the instance's own internal scratch state), so this gives each worker thread its own
// instance rather than sharing the caller's -- confirmed cheap to construct (VBMicrolensingLibrary
// .cpp's constructor is plain field initialization, no file I/O, no shared global/static mutable
// state touched by BinaryMag2 either, so N independent instances really are independent).
//
// Capped at a small, fixed thread count (default 4) rather than std::thread::hardware_concurrency()
// -- this is a GPU app; volunteers running it alongside other CPU work haven't committed a whole
// core's worth of CPU time to it, and over-subscribing would be exactly the kind of "GPU app eating
// all my CPU too" complaint this project has already had to fix once (app.weight/credit tuning
// earlier). (The first version spawned/joined fresh std::threads each batch -- replaced by
// persistent workers on 2026-10-08, see below.) Falls back to inline (no threads) for small
// batches, where handing work to the workers would cost more than it saves.
//
// 2026-10-08, PERSISTENT WORKERS: the pool used to spawn and join fresh std::threads for every
// batch. Every thread exit runs the destructors of VBMicrolensing's thread_local scratch objects
// (see above), and with MinGW-w64 (winpthreads/emutls) one of those, an
// _augmented_priority_queue, frees memory it does not own -> heap corruption, process killed
// with 0xC0000374 within the first cell (reproduced with a MinGW build of the OpenCL path; MSVC
// and Linux/glibc builds happened not to trip it). The vendored library must not be modified, so
// the workers are now created ONCE, kept for the life of the process and never exit: no
// thread_local destructor ever runs. Callers create the pool with `new` and never delete it, so
// no static destructor tries to join them at exit either (process exit simply ends them).
// The calling thread does share 0 of every batch itself, so n_threads = 4 still means 4 CPU
// threads busy, as before.
#include <thread>
#include <vector>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include "kampylos_gpu_complete.h"

class KampylosCpuPatchPool {
public:
    explicit KampylosCpuPatchPool(int n_threads = 4)
        : vbms_(std::max(1, n_threads)) {
        for (int t = 1; t < (int)vbms_.size(); t++) workers_.emplace_back([this, t]() { worker_loop(t); });
    }
    KampylosCpuPatchPool(const KampylosCpuPatchPool&) = delete;
    KampylosCpuPatchPool& operator=(const KampylosCpuPatchPool&) = delete;
    // Never destroyed in practice (see above); detach rather than join if it ever is.
    ~KampylosCpuPatchPool() { for (auto& w : workers_) w.detach(); }

    // Completes `results[i]`/`points[i]` pairs (same data/s/q for every one -- one grid cell's
    // worth), writing chi2_out[i] for each, in the SAME order as the input. Safe to call
    // repeatedly on the same instance (that's the point -- one pool per process, reused every
    // round). Not reentrant: one batch at a time.
    void complete_batch(
        const std::vector<CandidateResult>& results,
        const std::vector<Candidate>& points,
        const LightCurvePoint* data,
        double s, double q,
        std::vector<double>& chi2_out,
        double fs_bound = 1e300, double fb_offset = 0.0
    ) {
        int n = (int)results.size();
        chi2_out.resize(n);
        int n_threads = (int)vbms_.size();

        if (n_threads <= 1 || n < n_threads) {
            for (int i = 0; i < n; i++) {
                CompletedFit cf = kampylos_complete_candidate(vbms_[0], results[i], points[i], data, s, q, fs_bound, fb_offset);
                chi2_out[i] = cf.chi2;
            }
            return;
        }

        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = Job{ &results, &points, data, s, q, &chi2_out, fs_bound, fb_offset, n, n_threads };
            pending_ = n_threads - 1;
            generation_++;
        }
        start_cv_.notify_all();
        run_share(0, job_);
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this]() { return pending_ == 0; });
    }

private:
    struct Job {
        const std::vector<CandidateResult>* results;
        const std::vector<Candidate>* points;
        const LightCurvePoint* data;
        double s, q;
        std::vector<double>* chi2_out;
        double fs_bound, fb_offset;
        int n, n_threads;
    };

    void run_share(int t, const Job& j) {
        for (int i = t; i < j.n; i += j.n_threads) {
            CompletedFit cf = kampylos_complete_candidate(vbms_[t], (*j.results)[i], (*j.points)[i], j.data, j.s, j.q,
                                                          j.fs_bound, j.fb_offset);
            (*j.chi2_out)[i] = cf.chi2;
        }
    }

    void worker_loop(int t) {
        unsigned long long seen = 0;
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu_);
                start_cv_.wait(lk, [this, seen]() { return generation_ != seen; });
                seen = generation_;
                j = job_;
            }
            run_share(t, j);
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--pending_ == 0) done_cv_.notify_one();
            }
        }
    }

    std::vector<VBMicrolensing> vbms_;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable start_cv_, done_cv_;
    unsigned long long generation_ = 0;
    int pending_ = 0;
    Job job_{};
};
