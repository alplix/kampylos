# Benchmarking Kampylos (`scripts/bench/`)

How to measure whether a change makes Kampylos faster and whether it keeps the results, the scripts that do it, and
the results so far. The method is the one to reuse for every later speed change, on any platform; the results sections
below it grow with each change. The scripts build the real BOINC wrappers against the fake BOINC runtime
(`tests/boinc_stub`) and run them on fixed work units. First used on an Apple M1 on 2026-11-04.

## Method

### Two kinds of speed change

* **Same results**: build flags, allocator, library layout, scheduling of GPU and CPU work. The result lines (`out`
  without its `#` header lines) must stay byte-identical to the old build's on the same toolchain. Such a change needs
  a measured gain, the output check and the correctness checks below, nothing more.
* **Search changes**: anything that changes which candidates are evaluated (stopping rules, restarts, seeds,
  tolerances) changes the numbers. Measure it the same way, on whole cells, and treat it as a science change that the
  maintainers decide on, because the server compares results of different app versions (KAM-025). Until then keep it
  behind a switch that is off by default (`scripts/bench/patches/nm-stopping.patch`).

### Rules for every measurement

1. **Build both sides the same way, from clean copies.** The baseline is a git ref (`REF=upstream/master`) built with
   the documented build line; the candidate is the working tree, or the baseline plus a patch. `build_variant.sh`
   copies the tracked files, so a patched submodule or a stale object file in the working tree cannot leak in.
2. **Find out where the time goes first.** Sample a run (`sample <pid> 10` on macOS, `perf record -g` on Linux) and,
   for a GPU app, split GPU time from CPU completion time. A change saves at most the share of time it touches: a
   change to 10 % of the time cannot gain more than 1.11x.
3. **Use a profile that exercises the change.** A change to large GPU rounds does nothing on `planet-gpu` (about 41
   candidates per round); a change to the anchor stage weighs more on a 10-seed profile than in a whole task. Confirm
   the final number on a whole cell (`planet-cell`).
4. **Interleave and repeat.** `run_bench.sh` runs A B C A B C ... in fresh directories; use at least 3 rounds (with
   fewer the table says "too few runs" instead of a verdict). The load
   from other programs changes over minutes, and interleaving makes it hit every build alike. Compare medians.
5. **Accept by the rule, not by eye.** A candidate is faster when its median is at least 5 % below the baseline's and
   its slowest run beats the baseline's fastest, or when its median is at least 10 % below. Anything else is "no clear
   gain" and is reported as that.
6. **Check the output of every run.** Each run's result lines are hashed; `same output` means every run of a label
   matches the baseline. Compare like with like: different compilers, backends and thread counts differ at rounding
   level (KAM-012), so a Linux GCC build is compared with a Linux GCC build.
7. **Measure on a quiet machine.** Pause the BOINC client (CPU and GPU tasks) and other heavy programs, keep a laptop on
   power, and look at `uptime` (the scripts record the load). On a loaded machine an interleaved comparison is still
   fair, but its absolute times and spreads are not worth quoting.
8. **Write it down.** Date, machine, OS, compiler, load, profile, rounds, medians and spread, verdict and output go into
   the results below. A number in a commit message, a README or a code comment comes from there, and a platform that
   was not measured is called unmeasured.

### Steps for one change

1. Sample the current build on the profile closest to production and note where the time goes.
2. Make the change in the working tree, or as a patch (`build_variant.sh <name> <patch>`); build the baseline with
   `REF=`.
3. Run one round (`run_bench.sh planet 1 ...`) to see that the output stays the same.
4. Run 3 or more rounds on a quiet machine, on each profile the change touches, and a whole cell for the final number.
5. Run the correctness checks: `check_change.sh` (library thread test, Metal objective test, Metal kill-and-resume
   test, same output with 4 completion threads) and the tests that AGENTS.md routes the touched files to.
6. Measure other platforms where possible: `linux_ab.sh` (Linux, natively or in a container on a Mac), and
   `OPT="-O2 -femulated-tls"` for the emulated thread_local of MinGW-w64 GCC and older Android. Name the platforms that
   were not measured.
7. Record the results here, and update `boinc/README.md` when the way to build changes.

### Traps seen here

* Other programs: on the M1 the same build varied by up to 40 % between rounds at load 5 to 30, and a BOINC Metal task
  on the same GPU doubled a Metal run (222 s instead of 112 s).
* Efficiency cores: `taskpolicy -b` (background QoS) puts a process on the efficiency cores of Apple Silicon, `nice`
  does not, and the BOINC client's way of lowering priority decides which a volunteer's Mac does. `RUN_PREFIX`
  compares both.
* Seed history (KAM-030): a seed's fit depends on the VBMicrolensing state the previous seeds left. A replay of recorded
  evaluations predicted unchanged results for a restart rule; the real run moved the 10-seed OGLE profile by +21.6
  chi2. Measure search changes by real runs on whole cells, never by replay or on seed subsets alone.
* Fixed costs: the anchor stage takes 1.7-4.3 s once per task, which is a large share of a 10-seed profile.
* Containers on a Mac run on the same cores and share their load: good for ratios between Linux builds (thread_local,
  glibc allocator, GCC against clang), not for comparing Linux with macOS.

## Testing this branch on a Mac

On a quiet machine (BOINC client paused, laptop on power), from the repository root:

```
scripts/bench/check_change.sh        # correctness of the faster GPU app: PASS or FAIL, about 20 minutes
scripts/bench/mac_ab.sh              # speed and output: 3 rounds of planet, ogle, far and planet-gpu, about 40 minutes
scripts/bench/mac_ab.sh 5 planet     # 5 rounds of one profile
scripts/bench/linux_ab.sh            # optional: Linux builds in a container (docker), about 15 minutes
```

They need the Xcode command line tools, `python3`, the submodule (`git submodule update --init`) and Homebrew's
jemalloc (`brew install jemalloc`); `linux_ab.sh` on a Mac also needs docker or podman. Everything goes to
`~/kampylos-work` (`KAMPYLOS_WORK=` moves it). `mac_ab.sh` was checked end to end on the M1 on 2026-11-04 (`mac_ab.sh 1`,
13 minutes). It builds the baseline `upstream/master` (or `origin/master`; `BASE_REF=` picks another ref) the way
`boinc/README.md` describes, and this working tree the fast way, then runs each profile. Each profile ends with a table
like this one (`~/kampylos-work/bench-runs/<date>-<profile>/summary.txt`):

```
label        runs  median s    min s    max s    cpu s  RSS MB  GPU%   vs 1st  verdict / output
doc             3     45.89    45.64    47.24     45.2       2     1   1.000x  baseline; same output
strpj           3     24.31    23.77    25.35     24.1      10     3   1.888x  ACCEPT (+47.0 %); same output
```

`verdict` applies the rule used for every claim in this file: a candidate is faster when its median is at least 5 %
below the baseline's and its slowest run beats the baseline's fastest, or when its median is at least 10 % below.
`same output` means every run of that label wrote the same result lines (`out` without the `#` header lines) as the
baseline, `same output as X` the same as an earlier label (a Metal build after CPU builds: backends differ at rounding
level); otherwise the column shows the change of the first cell's chi2. `GPU%` is a rough sample of the whole GPU
(`ioreg`, twice a second); do not base decisions on it.

## The pieces

| file | what it does |
|---|---|
| `build_variant.sh <name> [patch ...]` | copies the tracked files (or `REF=<git ref>`), applies patches, prepares VBMicrolensing (`VBM=fork`, `strip` or `block`), links an allocator (`ALLOC=system`, `jemalloc` or `mimalloc`), builds `bins/<name>/cpu` and, on macOS, `bins/<name>/metal`. `OPT=` and `CXX=` set compiler and flags (`OPT="-O2 -femulated-tls"` imitates the thread_local of MinGW-w64 GCC). Linux: CPU only |
| `run_bench.sh <profile> <rounds> label=binary ...` | interleaved runs in fresh directories; wall, user and system time, peak memory, a hash of the result lines, chi2 of the first cell. `RUN_PREFIX` runs every binary through a command (`taskpolicy -b`, `nice -n 19`) |
| `summarize_bench.py <results.tsv>` | the table above |
| `profiles/*.wu` | `photometry`, `in` line and CPU budget (`KAMPYLOS_STUB_NCPUS`, the completion threads of a GPU app) of one work unit |
| `mac_ab.sh` | the documented builds of `BASE_REF` against this tree's fast builds, on all profiles |
| `check_change.sh` | the correctness checks of the faster GPU app: library thread test, Metal objective test, Metal kill-and-resume test, same result lines with 4 completion threads |
| `linux_ab.sh` | the library forms and allocators with GCC (and clang with `CLANG=1`) on Linux, natively or in a Debian 12 container on a Mac |

| profile | work unit | why |
|---|---|---|
| `planet` | synthetic planet (401 points), its own cell (log10 s = 0.2, log10 q = -3), `part 0 24` = 10 of 240 seeds | the source crosses a caustic: about 7 % of the points of a candidate need VBMicrolensing's finite-source contour integration, which then takes most of the time on every backend |
| `planet-gpu` | the same cell, `part 0 4` = 60 seeds | bigger GPU rounds than `planet` (about 41 candidates on average) |
| `ogle` | `tests/sample_ogle_blg1900.dat` (448 points, magnitudes), cell (0.3, -2.5), 10 seeds | a real light curve |
| `far` | synthetic planet, cell (0.2, -5), 10 seeds | far from the planet's mass ratio; the large source of this event (anchor rho = 0.10) still sends the points near the peak to the finite-source integration |
| `planet-cell` | the planet cell with all 240 seeds | one whole production task; 8-18 minutes per run |

The anchor stage (single-lens fits, before the first cell) is part of every run: 4.3 s on the planet light curve and
1.7 s on the OGLE one with the fast CPU build. It is paid once per task, so it weighs more in these 10-seed profiles
than in a production task.

## Results on an Apple M1 (2026-11-04)

Apple M1 (4 performance + 4 efficiency cores, 8-core GPU, 16 GB), macOS 15.7.4, Apple clang 14, `-O2`. Other jobs ran on
the machine during most measurements (load average 5 to 30); the runs are interleaved, so drift hits every build alike.

### Where the time goes

* **CPU app**: VBMicrolensing. With the fast build (plain statics, jemalloc), sampled on the planet cell: the
  polynomial root finders (`cmplx_laguerre2newton`, `cmplx_newton_spec`, `cmplx_roots_gen`) about 55 %, the contour
  integration (`NewImages`, `OrderImages`, skip lists) about 25 %, allocation about 10 %. Nelder-Mead, the flux fit and
  the seeding are invisible.
* **Metal app**: the CPU completion step. On 60 seeds of the planet cell (2757 rounds, 113720 candidates, 3.39 million
  finite-source points), with the build of `boinc/README.md`: GPU 13.2 s, completion of the finite-source points on the
  CPU 153.8 s, of a 172 s run. The whole cell took 1085 s (909 s of CPU) with that Metal build; the fast CPU app took
  536 s in the 2026-10-16 report. On a caustic-crossing cell the GPU app is as fast as its CPU side, and that CPU side
  was the slow one (thread_local library, system allocator).
* **Nelder-Mead**: 71 % of the objective evaluations are in the two restart passes, and 52 % in restarts that improve
  the seed's chi2 by no more than 1e-3. A restart from a converged point nearly always finds something a rounding step
  lower, and the old rule restarted again after any gain: 475 of 480 seeds (planet and OGLE cells, all seeds traced) ran
  the second restart. 606 of the 1440 passes stopped at the 400-iteration cap.

### Same results, faster

| change | where | measured | output |
|---|---|---|---|
| plain statics + jemalloc (`boinc/README.md` "Faster CPU build", 2026-10-16) | CPU app | `planet` 45.9 s -> 24.3 s (1.89x) | same |
| VBMicrolensing `--block` (one thread_local block per function) | GPU apps | CPU app with the block library, `planet`: 45.9 s -> 38.2 s (1.20x; plain statics 34.7 s) | same |
| `--block` + jemalloc | Metal app | `planet-gpu` 189.5 s -> 112.0 s (1.69x), see below | same |
| GPU/CPU overlap inside a round | Metal app | whole planet cell 1.12-1.14x, see below | same |

The `--block` rewrite (`vendor/patch_vbm_thread_local.py --block`) keeps every static thread_local, as the GPU apps'
completion threads need, but gathers each function's statics into one struct: a call pays one TLS lookup instead of one
per access. In `cmplx_laguerre2newton` that is 1 lookup per call instead of 44 places in the code; over the library 158
TLS references remain of 1208. It also makes the `complex` constructors `constexpr` and gives the struct members the zero
a static starts with, so the struct needs no initialization guard, and turns the two statics of `_point::operator-`
(written before every read) into locals. Same variables, same per-thread state, same initial values: the result lines
are byte-identical, as the table shows.

Metal app, profile `planet-gpu` (60 seeds, one completion thread as under today's plan classes), 3 interleaved runs:

| build | median s | min-max s | CPU s | vs `boinc/README.md` |
|---|---|---|---|---|
| `boinc/README.md` build line (fork, system allocator) | 189.5 | 173.8-254.9 | 160.8 | 1.00x |
| jemalloc | 138.4 | 138.2-155.7 | 124.5 | 1.37x |
| `--block` | 145.2 | 142.2-148.7 | 130.8 | 1.30x |
| `--block` + jemalloc | 112.0 | 110.6-116.9 | 96.8 | 1.69x |

Every run of every build wrote the same result line (chi2 411.2246). The slowest baseline run (254.9 s) coincided with
another process using the GPU.

**GPU and CPU overlap in the Metal app** (`src/kampylos_metal_fit.mm`, with `complete_batch(first, count)` in
`src/kampylos_cpu_patch_pool.h`): the app used to complete a round on the CPU only after the GPU had finished all of it,
and the GPU idled while the CPU worked. A round is now cut into at least 4 chunks of at least 64 candidates, and chunk
k+1 runs on the GPU while the CPU completes chunk k. Every thread of the pool completes the same candidates in the same
order as before, so the results are bit-identical. It acts only on rounds of more than 64 candidates, so `planet-gpu`
(about 41 per round) did not change (113.1 s against 112.0 s). On the whole planet cell (`planet-cell`, both with
`--block` and jemalloc, two pairs run back to back in alternating order while the machine's load changed):

| pair | without overlap | with overlap | output |
|---|---|---|---|
| 1 | 853.4 s | 751.0 s (1.14x) | same line (chi2 411.2230) |
| 2 | 603.9 s | 537.1 s (1.12x) | same line |

The fast CPU app took 536 s for this cell in the 2026-10-16 report: on a caustic-crossing cell the Metal app with one
completion thread is about as fast as the CPU app, because both spend their time in the same finite-source integration.
CUDA and OpenCL keep one launch per round; the same split would apply there (not done, not testable here).

### Search changes: fewer evaluations, other minima

`scripts/bench/patches/nm-stopping.patch` adds two switches to both drivers (CPU `binary_fit.h`/`simplex.h`, GPU
`kampylos_batched_fit.h`/`nelder_mead_stepper.h`), off unless set with `-D`; `EXPERIMENTS=1 scripts/bench/mac_ab.sh`
builds and runs them. Neither is in the apps.

* **Restart only after a real gain** (`KAMPYLOS_RESTART_MIN_GAIN=1e-3`): a restart earns the next one only when it
  lowered chi2 by more than 1e-3. Replaying the recorded evaluations gave 1.27x fewer evaluations and the same cell
  chi2 on both traced cells. Run for real it is faster as predicted, but the result is not always the same:

  | work unit (fast CPU build) | old rule | gain > 1e-3 |
  |---|---|---|
  | planet cell, all 240 seeds | chi2 411.2228 | same line, 1.15x less CPU time |
  | OGLE cell, all 240 seeds | chi2 2811.6732 | same line, 1.24x less CPU time |
  | profile `planet` (10 seeds), 3 runs each | 24.6 s, chi2 411.3205 | 19.9 s (1.24x), same line |
  | profile `ogle` (10 seeds), 3 runs each | 10.8 s, chi2 2833.4503 | 7.8 s (1.38x), chi2 2855.0742 (+21.6) |

  (The whole-cell times were measured while other runs shared the machine.) In the OGLE profile the rule did not touch
  seed 96, the best one, yet its first pass ended at 2855.31 instead of 2833.61: skipping the second restarts of seeds
  0-72 changed the VBMicrolensing state that seed 96 started from (the root solver's warm start), and Nelder-Mead
  followed another path from the first last-bit difference. Fitted alone, seed 96 also ends at 2855.31. That is
  KAM-030: a seed's result depends on what the process computed before it, so a resumed task or a seed-split part can
  end a seed in another minimum. Any change to the search moves individual seeds like this; the whole-cell results
  above held.

* **Stop a pass that stalls** (`KAMPYLOS_NM_STALL_ITERS=n`, `KAMPYLOS_NM_STALL_TOL=tol`: the best value has not
  improved by more than `tol` in the last `n` iterations; the GPU checkpoint carries the stall state): the replay
  estimates 2.7x (`n` ~ 50, `tol` 1e-4) to 3.1x fewer evaluations. In full runs (fast CPU build plus the restart rule
  above):

  | rule | planet cell | OGLE cell |
  |---|---|---|
  | none (restart rule only) | 577 CPU s, chi2 411.2228 | 281 CPU s, chi2 2811.6732 |
  | 100 iterations, 1e-4 | 366 CPU s, same line | 157 CPU s, chi2 2810.7380 (-0.94) |
  | 50 iterations, 1e-4 | 243 CPU s, same line | 118 CPU s, chi2 2815.6617 (+3.99) |

  (Measured while other runs shared the machine: compare the columns, not with the tables above.) The search then ends in
  other local minima on the OGLE cell, better or worse by a few chi2. That is the size of the difference between seeds
  that end in neighbouring minima, small next to the delta-chi2 of a detection, but the result of a cell is no longer
  the one the current apps report, which matters if the server compares results of different app versions (KAM-025).
  Worth it only together with a server-side decision, and easier to judge once KAM-030 is fixed (then each seed's
  change is its own).

### Tried, not proposed

* **Compiler flags**: `-O3`, `-mcpu=apple-m1` and LTO gave 0-8 %, within noise (2026-10-16 report; not repeated).
* **Hoisting `exp`, `sin`, `cos` out of `BinaryLightCurve`**, early exit of a trial evaluation, point order: below 1 %
  or changes results (2026-10-16 report).

### Checks of this branch (2026-11-04, M1)

* `--block`: `tests/test_vbm_threads.cpp` passes (4 threads x 3000 `BinaryMag2`, no mismatch); the script refuses a
  library that is not the thread_local v5.6 and `--strip` on a block-form library; the CPU and Metal apps write the same
  result lines as with the fork on every profile above.
* Overlap: `tests/test_metal_fit.mm` passes; `KAMPYLOS_RESUME_BACKEND=metal sh tests/test_boinc_resume.sh` passes with
  `--block` and the overlap (23 kills, byte-identical); with 4 completion threads (`KAMPYLOS_STUB_NCPUS=4`,
  `planet-gpu`, first round 360 candidates in 4 chunks) the result is the same with and without the overlap.
* `scripts/bench/patches/nm-stopping.patch` with every switch at 0 gives the same result lines as without it.
* `--block` with GCC 12 and clang 14 on Linux: builds, same result lines as the fork (below).
* Not run: CUDA, OpenCL, Windows builds, the regression suite (it needs the server's `kampylos_analyze.py`).

## Other platforms

### Measured on 2026-11-04 (the same M1, under load)

The CPU app on profile `planet`, 3 interleaved rounds each, while other jobs kept the load at 5 to 16, so the spreads are
wide. The CPU app runs the same VBMicrolensing code as the GPU apps' completion step, so the library forms compare as
they would there.

| setup | thread_local library (fork) | `--block` | plain statics | allocator | output |
|---|---|---|---|---|---|
| macOS, Apple clang 14, native thread_local (results above) | 45.9 s | 38.2 s (1.20x) | 34.7 s (1.32x) | plain statics + jemalloc 24.3 s (1.89x) | same |
| macOS, Apple clang 14 with `-femulated-tls`: the thread_local of MinGW-w64 GCC and of Android below API 29 | 74.3 s, 0.78x of the native fork (57.6 s in the same runs) | 59.1 s (**1.26x**; every run faster than every fork run) | 45.1 s (1.65x) | not measured | same |
| Linux arm64 in a Debian 12 container (`linux_ab.sh`), GCC 12.2, glibc 2.36 | 37.4 s | 39.0 s (0.96x) | 38.7 s (0.97x) | jemalloc 34.1 s (1.09x), with `--block` 35.3 s (1.06x), mimalloc 35.5 s (1.05x): no clear gain | same |

Clang 14 on the same Linux wrote other last digits than GCC 12 (another compiler; compare like with like) and was
slower in 9 of 10 interleaved pairs (1.2x to 1.9x), but at load 10 to 15: repeat that on a quiet machine before
choosing a compiler for the Linux apps.

### What carries over

* **thread_local costs differ by platform**, and so does the gain of `--block`. macOS: every access is a call
  (`_tlv_get_addr`); `--block` gained 1.20x in the CPU app and 1.30x in the Metal app. Emulated TLS (MinGW-w64 GCC,
  Android NDK below API 29): a call to `__emutls_get_address` per access, the most expensive form; `--block` gained 1.26x.
  Linux, thread_local in the executable: a load relative to the thread pointer; neither `--block` nor plain statics
  gained anything. Windows with MSVC, and llvm-mingw (native Windows TLS): not measured; a few loads through the
  thread's TLS array, so expect something between Linux and macOS. `--block` keeps the same objects with the same
  destructors, so the rule that completion threads never exit (KAM-010) applies unchanged.
* **`--block` compiles** with Apple clang 14, GCC 12 and clang 14, and gives the fork's result lines with each. MSVC
  and the host compilers of the CUDA builds were not tried.
* **The allocator** matters most on macOS (its system allocator took about a third of the CPU app). With glibc,
  jemalloc and mimalloc gained 5 to 9 %, within noise. The Windows heap was not measured.
* **GPU apps on caustic-crossing cells are CPU-bound** on every backend: CUDA and OpenCL use the same completion step
  (`kampylos_gpu_complete.h`, `kampylos_cpu_patch_pool.h`). A faster GPU does not help such a cell; the completion
  threads do, and their number is the plan class's `avg_ncpus` (0.1-0.25 today, so one thread). The overlap of GPU
  and CPU inside a round is only in the Metal app (`kampylos_metal_fit.mm`); CUDA (`gpu_fit_binary.cu`) and OpenCL
  (`kampylos_opencl_fit.cpp`) launch once per round and wait (`cudaMemcpy`, `clFinish`) before completing, and could
  take the same `complete_batch(first, count)` split. It should gain more where the GPU's share of a round is larger:
  a slower GPU, or more completion threads.
* **A GPU shared with another GPU task** slows the GPU part: one `planet-gpu` run of the new Metal build took 222 s
  (132 s of CPU) instead of 112 s while the BOINC client ran another Metal app on the same GPU. The overlap
  inside a round hides part of such waits; nothing in the app can remove them.
* **Short tasks pay the anchor stage** (1.7-4.3 s here) once per task; seed-split work units with many parts pay it once
  per part.
* **BOINC priority**: the client runs apps at low priority. On Apple Silicon a process with background QoS runs on the
  efficiency cores only, a niced one need not; which one a volunteer's Mac gives the app depends on how the client
  lowers the priority. Compare `RUN_PREFIX="taskpolicy -b"` and `RUN_PREFIX="nice -n 19"` before trusting a gain.
