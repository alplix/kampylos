#pragma once
// The seeds of one cell on several threads, for a multi-threaded CPU app (2026-11-05, KAM-040).
//
// Why: a CPU task is often one cell, and on an event whose single-lens anchor has no baseline one cell takes
// days (docs/BENCHMARK.md, "Measurements of 2026-10-16" §4.2). The GPU apps already spread their completion
// step over the client's CPU budget (kampylos_cpu_patch_pool.h, KAM-032); this does the same for the CPU app's
// seed loop, so a task declared as N CPUs (an `mt` plan class, avg_ncpus = N) finishes about N times sooner for
// the same CPU time. One worker per CPU, each with its own VBMicrolensing instance, takes the next seed
// as soon as it is free (seeds differ in cost by a factor of a few, so fixed stripes would leave threads idle);
// the calling thread only merges.
//
// Same result as one thread: the fits come back to the caller in seed order (commit(j, fit) for j = 0, 1, ...),
// so the best-of merge, the progress and the seed checkpoint (the first seed not yet merged, and the best fit
// before it) are exactly those of the sequential loop in kampylos_cpu_cell.h. That makes the cell result
// identical only when a seed's fit does not depend on what its thread computed before -- which is
// -DKAMPYLOS_RESET_STATE=1 (kampylos_vbm_state.h, KAM-030); without it, a seed fitted on another thread after
// other seeds can end in another minimum. kampylos_cpu_seed_threads_allowed() says whether the build allows
// more than one thread: the reset, and the thread-safe library (thread_local statics: the fork, or --block).
//
// Threads are created once and never exit (VBMicrolensing's thread_local destructors corrupt the heap on
// MinGW-w64 when a thread exits, KAM-010); create the pool with `new` and never delete it, as the completion
// pool. They block SIGALRM, so BOINC's timer signal goes to the main thread (where the BOINC runtime handles
// suspend and quit), and between two evaluations they call kampylos_worker_pause() (below), which the BOINC
// wrapper points at a wait while the client has the task suspended (KAM-027).
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#if !defined(_WIN32)
#include <pthread.h>
#include <signal.h>
#endif
#include "binary_fit.h"

#if defined(VBM_THREAD_LOCAL_STATICS) || defined(VBM_THREAD_SAFE)
#define KAMPYLOS_SEED_POOL_THREAD_SAFE 1
#else
#define KAMPYLOS_SEED_POOL_THREAD_SAFE 0
#endif

// True if this build may fit the seeds of a cell on more than one thread with the result of one thread.
inline bool kampylos_cpu_seed_threads_allowed() { return KAMPYLOS_SEED_POOL_THREAD_SAFE && KAMPYLOS_RESET_STATE; }

// Blocks SIGALRM in the calling thread (POSIX): BOINC's timer signal must reach the main thread, where the
// runtime suspends and quits the task; a worker that received it would be the one put to sleep instead.
inline void kampylos_block_timer_signal() {
#if !defined(_WIN32)
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGALRM);
    pthread_sigmask(SIG_BLOCK, &set, nullptr);
#endif
}

class KampylosSeedPool {
public:
    explicit KampylosSeedPool(int n_threads) : n_(std::max(1, n_threads)), vbms_(std::max(1, n_threads)) {
        for (int t = 0; t < n_; t++) workers_.emplace_back([this, t]() { worker_loop(t); });
    }
    KampylosSeedPool(const KampylosSeedPool&) = delete;
    KampylosSeedPool& operator=(const KampylosSeedPool&) = delete;
    ~KampylosSeedPool() { for (auto& w : workers_) w.detach(); }   // never destroyed in practice (see above)
    int threads() const { return n_; }

    // Fits seed idx[j] of `seeds` for j = 0 .. idx.size()-1 with fit_binary_local_steps() and calls commit(j, fit)
    // on the calling thread in the order j = 0, 1, ..., each as soon as the fits up to it are done. Once commit
    // returns false no further seed is started; run() returns when the seeds in flight are done.
    void run(const std::vector<DataPoint>& data, double ln_s, double ln_q, const BinarySeedSet& seeds,
             const std::vector<size_t>& idx, const std::function<bool(size_t, const BinaryFitResult&)>& commit) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = Job{ &data, ln_s, ln_q, &seeds, &idx };
            res_.assign(idx.size(), BinaryFitResult{});
            done_.assign(idx.size(), 0);
            next_ = 0;
            stop_ = false;
            busy_ = n_;
            generation_++;
        }
        start_cv_.notify_all();
        for (size_t j = 0; j < idx.size(); j++) {
            BinaryFitResult r;
            {
                std::unique_lock<std::mutex> lk(mu_);
                done_cv_.wait(lk, [&]() { return done_[j] != 0; });
                r = res_[j];
            }
            if (!commit(j, r)) {
                std::lock_guard<std::mutex> lk(mu_);
                stop_ = true;
                break;
            }
        }
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [&]() { return busy_ == 0; });
    }

private:
    struct Job {
        const std::vector<DataPoint>* data;
        double ln_s, ln_q;
        const BinarySeedSet* seeds;
        const std::vector<size_t>* idx;
    };

    void worker_loop(int t) {
        kampylos_block_timer_signal();
        unsigned long long seen = 0;
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu_);
                start_cv_.wait(lk, [&]() { return generation_ != seen; });
                seen = generation_;
                j = job_;
            }
            for (;;) {
                size_t k;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    if (stop_ || next_ >= j.idx->size()) break;
                    k = next_++;
                }
                size_t seed = (*j.idx)[k];
                BinaryFitResult r = fit_binary_local_steps(vbms_[t], *j.data, j.ln_s, j.ln_q, j.seeds->x0[seed], j.seeds->step[seed]);
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    res_[k] = r;
                    done_[k] = 1;
                }
                done_cv_.notify_all();
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                busy_--;
            }
            done_cv_.notify_all();
        }
    }

    int n_;
    std::vector<VBMicrolensing> vbms_;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable start_cv_, done_cv_;
    unsigned long long generation_ = 0;
    Job job_{};
    std::vector<BinaryFitResult> res_;
    std::vector<char> done_;
    size_t next_ = 0;
    bool stop_ = false;
    int busy_ = 0;
};
