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
  glibc allocator, GCC against clang), not for comparing Linux with macOS. An x86_64 container on Apple Silicon runs
  under Rosetta: use it to see that a build runs and what it writes, never for speed.
* The compiler version changes results: Apple clang 21 fuses multiply-adds where Apple clang 14 did not (Results on an
  Apple M4), so "same output" holds only between builds of one compiler with one `-ffp-contract`. Compare like with like,
  and with contraction on even PGO changed the Metal app's result.
* A gain measured on one machine need not carry over to another of the same family: `--block` gave 1.30x on the M1 and
  nothing on the M4 Pro, the overlap 1.13x and 1.03x. Say which machine a number is from.

## Testing this branch on a Mac

On a quiet machine (BOINC client paused, laptop on power), from the repository root:

```
scripts/bench/check_change.sh        # correctness of the faster GPU app: PASS or FAIL (M1: 20 minutes, M4 Pro: 5)
scripts/bench/mac_ab.sh              # speed and output: 3 rounds of planet, ogle, far and planet-gpu (M1: 40 min, M4 Pro: 12)
scripts/bench/mac_ab.sh 5 planet     # 5 rounds of one profile
scripts/bench/linux_ab.sh            # optional: Linux arm64 builds in a container (docker), about 10 minutes
scripts/bench/cross_check.sh         # optional: Windows (MinGW-w64, Wine) and Linux x86_64 builds, emulated, about 10 minutes
```

They need the Xcode command line tools (Xcode itself is not needed: on 2026-11-04 the M4 ran everything with the
Command Line Tools alone), `python3`, the submodule (`git submodule update --init`) and Homebrew's jemalloc
(`brew install jemalloc`); `linux_ab.sh` and `cross_check.sh` on a Mac also need docker or podman, and `cross_check.sh`
MinGW-w64 (`brew install mingw-w64`). Everything goes to
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
| `build_variant.sh <name> [patch ...]` | copies the tracked files (or `REF=<git ref>`), applies patches, prepares VBMicrolensing (`VBM=fork`, `strip` or `block`; `VBMX="reset pool"` adds the patch script's `--reset` and `--pool`), links an allocator (`ALLOC=system`, `jemalloc` or `mimalloc`), builds `bins/<name>/cpu` and, on macOS, `bins/<name>/metal`. `OPT=` and `CXX=` set compiler and flags (`OPT="-O2 -femulated-tls"` imitates the thread_local of MinGW-w64 GCC). Linux: CPU only |
| `run_bench.sh <profile> <rounds> label=binary ...` | interleaved runs in fresh directories; wall, user and system time, peak memory, a hash of the result lines, chi2 of the first cell. `RUN_PREFIX` runs every binary through a command (`taskpolicy -b`, `nice -n 19`) |
| `summarize_bench.py <results.tsv>` | the table above |
| `patches/censor-count.patch` | instrumentation only (2026-11-05): counts the objective evaluations by Nelder-Mead step and what a lower bound of chi2 would have let them skip; printed at exit |
| `profiles/*.wu` | `photometry`, `in` line and CPU budget (`KAMPYLOS_STUB_NCPUS`, the completion threads of a GPU app) of one work unit |
| `mac_ab.sh` | the documented builds of `BASE_REF` against this tree's fast builds, on all profiles |
| `check_change.sh` | the correctness checks of the faster GPU app: library thread test, Metal objective test, Metal kill-and-resume test, same result lines with 4 completion threads |
| `linux_ab.sh` | the library forms and allocators with GCC (and clang with `CLANG=1`) on Linux, natively or in a Debian 12 container on a Mac |
| `cross_check.sh` | the CPU app for Windows (MinGW-w64 GCC, static, run by Wine) and Linux x86_64 (GCC) in each library form, in an x86_64 Debian 12 container: compiles, runs, which result lines. On Apple Silicon it runs under Rosetta, so its times are no speed result |

| profile | work unit | why |
|---|---|---|
| `planet` | synthetic planet (401 points), its own cell (log10 s = 0.2, log10 q = -3), `part 0 24` = 10 of 240 seeds | the source crosses a caustic: about 7 % of the points of a candidate need VBMicrolensing's finite-source contour integration, which then takes most of the time on every backend |
| `planet-gpu` | the same cell, `part 0 4` = 60 seeds | bigger GPU rounds than `planet` (about 41 candidates on average) |
| `ogle` | `tests/sample_ogle_blg1900.dat` (448 points, magnitudes), cell (0.3, -2.5), 10 seeds | a real light curve |
| `far` | synthetic planet, cell (0.2, -5), 10 seeds | far from the planet's mass ratio; the large source of this event (anchor rho = 0.10) still sends the points near the peak to the finite-source integration |
| `planet-cell` | the planet cell with all 240 seeds | one whole production task; 8-18 minutes per run on the M1, 5 on the M4 Pro |
| `ogle-cell`, `far-cell` | the OGLE and the far cell with all 240 seeds (2026-11-05) | whole cells for search changes; 3 and 4 minutes with the fast CPU build on the M4 Pro |

The anchor stage (single-lens fits, before the first cell) is part of every run: 4.3 s on the planet light curve and
1.7 s on the OGLE one with the fast CPU build. It is paid once per task, so it weighs more in these 10-seed profiles
than in a production task.

## Measurements of 2026-10-16 (before this suite)

The repository used to carry these as an HTML report (`kampylos-report-2026-10-16.html`, removed on 2026-11-04); the
numbers that issues and comments cite are kept here, with that report's section numbers. Apple M1, macOS 15.7.4, BOINC
client 8.2.10, Apple clang 14, master 12cb505, a BOINC task running on another core; repeats differed by up to 10 %.

* **§2.1 Where the CPU app's time went** (`sample`, one grid cell): the shipped `kampylos_v12_macos_arm64` spent 18 %
  in thread-local lookups and 36 % in malloc/free on the synthetic planet (15 % and 31 % on the real event ev25055); a
  local build with the `boinc/README.md` line 46 % and 23 %. Nelder-Mead, the flux fit and seeding: under 0.2 %.
* **§2.2 Same results, faster** (per-seed fits printed at `%.17g`, same md5 in every row; A = synthetic planet, seeds
  0-3; B = ev25055, 6 shortened seeds):

  | variant | A | B |
  |---|---|---|
  | fork (thread_local), system malloc | 16.15 s | 41.16 s |
  | plain statics (`--strip`) | 1.36x | 1.35x |
  | plain statics + jemalloc | 2.04x | 1.83x |
  | plain statics + a pool allocator (upper bound, measurement only) | 2.20x | 1.88x |
  | thread_local + jemalloc (the GPU apps' host side) | 1.33x | 1.25x |
  | plain statics with `-O3`, `-mcpu=apple-m1` or LTO | as plain statics | as plain statics |
  | `-femulated-tls` (MinGW-w64 GCC, Android NDK below API 29) | 0.79x | 0.82x |

  Whole work unit (synthetic planet cell, 240 seeds): 1066 s with the documented build, 536 s with plain statics and
  jemalloc (byte-identical `out`), 839 s with the shipped binary, whose line differed in the last digits (t0 ...101
  against ...102, rho 1.155625e-06 against 1.155637e-06): it was built with a newer toolchain (macOS SDK 27), which
  fuses multiply-adds (see Results on an Apple M4). OGLE cell (0.3, -2.5): chi2 2811.6732 from both local builds.
* **§2.4 Tried in the algorithm, nothing material that keeps the result:** an early exit inside a trial evaluation
  (a lower bound of chi2 from the points summed so far): byte-identical, 0.9 % fewer magnification calls; hoisting the
  `exp`, `sin`, `cos` of `BinaryLightCurve`: 0.2-0.9 % fewer instructions; another point order: no gain, and not
  bit-identical, because VBMicrolensing's root solver starts from the previous call's roots (`zr` in `NewImages`, a
  function-local static), so a magnification can differ in its last bit depending on the point before it. Seed order
  did not matter (seeds 0-3, 100, 200 forwards and backwards: bit-identical per-seed fits). Repeated uninterrupted CPU
  runs were byte-identical. Faster but result-changing, not proposed: the far-field single-lens shortcut (up to ~3x per
  evaluation, changes A at 1e-5), dropping the mirror-image half of the seeds, looser tolerances, fewer iterations.
* **§3 The shipped macOS binaries were older than master** (as of 2026-10-16): `kampylos_v12_macos_arm64` printed "v4"
  and had the old resume path; `kampylos_metal_v2_macos_arm64` printed "v1", had only the cell checkpoint, left fast math
  on (with fast math the synthetic planet cell converged to chi2 692 instead of 411) and never checked a command buffer.
* **§4.2 Cost estimates:** an event without a baseline (ev25055: FSPL anchor with u0 = 1.2e-12, tE = 1207 d,
  n_peak = 1191 of 1191 points) took 24-50 minutes per seed, 96-199 CPU-hours per task, against a median of 6 minutes
  for 152 other Kampylos tasks of the same host; `rsc_fpops_est` = 9e13 put it at 4.3 hours (22-46x too low). A 10-cell
  row (ev23010_row4) took 34 hours.
* **§4.3 Branches of 2026-10-16** on the contributor's fork: `build/faster-cpu-app` (`--strip`, merged since),
  `fix/metal-device-unavailable`, `fix/resume-trust-output` (KAM-007), `fix/opencl-work-group-size` (KAM-006, only
  syntax-checked), `docs/agents-md`.

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
  536 s in the 2026-10-16 measurements (above). On a caustic-crossing cell the GPU app is as fast as its CPU side, and that CPU side
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

The fast CPU app took 536 s for this cell on 2026-10-16: on a caustic-crossing cell the Metal app with one
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

* **Compiler flags**: `-O3`, `-mcpu=apple-m1` and LTO gave 0-8 %, within noise (2026-10-16, §2.2 above; repeated on the M4 below).
* **Hoisting `exp`, `sin`, `cos` out of `BinaryLightCurve`**, early exit of a trial evaluation, point order: below 1 %
  or changes results (2026-10-16, §2.4 above).

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

## Results on an Apple M4 (2026-11-04)

Apple M4 Pro (8 performance + 4 efficiency cores, 16-core GPU, 24 GB), macOS 27.0 (26A428), Apple clang 21.0.0
(clang-2100.3.34.2, Command Line Tools only), Homebrew jemalloc 5.4.0, `-O2` unless a row says otherwise. The BOINC client's
tasks were suspended; the load average was 2.3 to 4.2, of which one core is the benchmark itself and about 0.7 the window
server and the BOINC Manager window, so 7 of the 8 performance cores stayed free. Spreads were below 2 % in every table.
Every table is 3 interleaved rounds, medians, the verdict by the rule of this file.

### The documented A/B (`scripts/bench/mac_ab.sh 3`, about 12 minutes here)

| profile | build | median s | min-max s | vs baseline | output |
|---|---|---|---|---|---|
| `planet` | CPU, `boinc/README.md` build line | 19.37 | 19.30-19.46 | 1.00x | |
| | CPU, fast build (`--strip`, jemalloc), base and this tree | 15.95 | 15.94-16.13 | 1.21x, ACCEPT | same |
| `ogle` | CPU, build line | 8.34 | 8.33-8.35 | 1.00x | |
| | CPU, fast build | 6.82 | 6.81-6.85 | 1.22x, ACCEPT | same |
| `far` | CPU, build line / fast | 15.04 / 12.51 | 15.01-15.06 / 12.50-12.55 | 1.20x, ACCEPT | same |
| | Metal, build line / this tree (`--block`, jemalloc, overlap) | 18.28 / 16.51 | 18.19-18.39 / 16.45-16.66 | 1.11x, ACCEPT | same |
| `planet-gpu` | Metal, build line | 68.65 | 68.11-68.70 | 1.00x | chi2 411.2240 |
| | Metal, this tree | 62.78 | 62.68-62.84 | 1.09x, ACCEPT | same |

`scripts/bench/check_change.sh`: PASS in 5 minutes (library thread test, Metal objective test, Metal kill-and-resume
test in 217 s, same result lines with 4 completion threads).

**The same changes gain much less here than on the M1** (CPU app 1.21x instead of 1.89x, Metal app 1.09x instead of
1.69x), and the split shows why:

| CPU app, `planet` | system allocator | jemalloc |
|---|---|---|
| thread_local library (fork) | 19.28 s | 17.13 s (1.13x) |
| `--block` | 19.42 s (0.99x) | 17.49 s (1.10x) |
| plain statics (`--strip`) | 17.99 s (1.07x) | 15.92 s (1.21x) |

On the M1 (macOS 15.7, Apple clang 14) `--block` alone gave 1.20x and plain statics 1.32x. Here `--block` gains nothing
and plain statics 7 %: a thread_local is cheap on this machine. Apple clang 21 still makes every access a call through
the TLS descriptor (1088 indirect calls in `VBMicrolensingLibrary.o` of the fork, 63 with `--block`, 2 with
`--strip`), so it is the call that became cheap (macOS 27's `_tlv_get_addr`, the M4, or both: one machine cannot
separate them), not the code that changed. Volunteers' Macs run older macOS versions and older chips, where the M1
numbers apply, and the block library costs nothing here (the Metal app without it: 60.65 s against 61.55 s, no clear
difference, same output). Keep `--block` in the GPU builds. jemalloc remains the larger macOS gain (1.13x alone).

### More completion threads: the large lever of the GPU apps

The Metal app of this tree (`--block`, jemalloc, overlap) on `planet-gpu` with the stub's CPU budget set to 1, 2, 4 and 8
(`KAMPYLOS_STUB_NCPUS`, the client's `ncpus`, which the plan class's `avg_ncpus` sets):

| completion threads | median s | min-max s | CPU s | vs 1 thread |
|---|---|---|---|---|
| 1 (today's plan classes) | 63.02 | 62.83-63.27 | 55.6 | 1.00x |
| 2 | 37.96 | 37.85-38.20 | 57.3 | 1.66x |
| 4 | 25.03 | 25.02-25.07 | 60.4 | 2.52x |
| 8 | 18.66 | 18.56-18.69 | 60.6 | 3.38x |

The CPU time grows by 9 % from 1 to 8 threads, so the same work finishes 3.4 times sooner for almost the same CPU. With
4 threads this tree's build is 1.19x faster than the build line (29.73 s, 29.66-29.81, against 25.03 s, same result
lines), as on one thread. The result lines differ between thread counts at rounding level (chi2 +0.0001 at 2 threads,
KAM-012), which is why this is a plan-class decision for the server and not a build change: a GPU task that runs on a
caustic-crossing cell is CPU-bound, and every completion thread it is given shortens it nearly in proportion.

### Compiler flags, PGO and floating-point contraction

| build (CPU app, `--strip` + jemalloc) | `planet` median s | `ogle` median s | output |
|---|---|---|---|
| `-O2` | 15.80 | 6.71 | |
| `-O3` | 15.68 (+0.8 %) | 6.72 (-0.1 %) | same |
| `-O2 -flto` | 15.18 (+3.9 %) | 6.50 (+3.1 %) | same |
| `-O2` with PGO (`-fprofile-instr-use`, trained on seeds the profiles do not use) | 15.64 (+1.0 %) | 6.66 (+0.7 %) | same |
| `-O2 -ffp-contract=off` | 15.11 (+3.1 %) | 6.85 (-2.4 %) | **other**: see below |

None passes the rule; flags stay as they are. PGO of the Metal app (+1.5 %, 60.73 s against 61.68 s) changed its result
(chi2 -0.0008): with contraction on, which multiply-adds the backend fuses depends on its optimisation decisions.

**Apple clang 21 fuses multiply-adds; Apple clang 14 did not.** Clang 21 defaults to `-ffp-contract=on` and emits 2433
`fmadd`-family instructions in `VBMicrolensingLibrary.o` (fast CPU build), where the 2026-10-27 count for Apple clang 14
was none (PAT-018). The results move in the last digits: the CPU app writes chi2 411.3206 on `planet` with clang 21's
default and 411.3205 with `-ffp-contract=off`, which is the M1's (clang 14) value; on `ogle` the result lines differ as
well. The build line and the fast build agree with each other with either setting. So the Mac apps' result lines depend
on the Xcode version of the release machine, and `-ffp-contract=off` (no speed cost measured: +3.1 % and -2.4 %) pins them
to what older Xcode versions and GCC with `-std=c99` produce. Whether that matters is KAM-025's question; the facts are in
KAM-025's log. The Metal app's result also depends on the GPU (411.2240 here with either setting, 411.2246 on the M1).

### Whole planet cell: the GPU/CPU overlap alone

`scripts/bench/run_bench.sh planet-cell 3`, this tree's Metal app (`--block`, jemalloc) with and without the overlap
(`no-overlap.patch` = this tree's `src/kampylos_metal_fit.mm` and `src/kampylos_cpu_patch_pool.h` set back to
`upstream/master`), one completion thread:

| build | median s | min-max s | CPU s | output |
|---|---|---|---|---|
| without the overlap | 302.91 | 302.89-304.21 | 282.8 | |
| with the overlap | 293.21 | 292.73-294.72 | 283.5 | same (chi2 411.2240) |

1.03x, "no clear gain" by the rule, where the M1 gained 1.12-1.14x. The overlap can hide at most the GPU's share of a
round, and the M4's 16-core GPU leaves little to hide: about 10 s of the 293 s are not CPU time. It costs nothing and helps
GPUs that are slower relative to their CPU (the M1's 8-core GPU, older Intel Macs), so it stays. A whole cell took 293 s
here against 537-751 s on the M1 (with load there).

### Search changes (`EXPERIMENTS=1 scripts/bench/mac_ab.sh 3 "planet ogle"`)

Labels as in "Search changes" above (fast CPU build plus the switch):

| profile | fast build | restart only after a gain > 1e-3 | also stop a stalled pass (100 iterations, 1e-4) |
|---|---|---|---|
| `planet` | 15.73 s | 13.41 s (1.17x), same line | 9.66 s (1.63x), other line (chi2 411.3206 at 4 decimals) |
| `ogle` | 6.75 s | 5.24 s (1.29x), chi2 +21.62 | 3.33 s (2.03x), chi2 +21.62 |

The same picture as on the M1: the restart rule keeps the planet cell's line and moves the 10-seed OGLE profile by
+21.6 through seed 96 (KAM-030), and the stall rule changes the lines. They stay behind their switches, a decision for
the maintainers.

### Next steps by platform (plan of 2026-11-04)

Ordered by expected gain for volunteers; "decide" items need the maintainers.

| platform | app | next step | expected | evidence |
|---|---|---|---|---|
| all GPU apps | Metal, CUDA, OpenCL | give the completion step more CPU (plan class `avg_ncpus`), decide | 1.7-3.4x on caustic-crossing cells at 2-8 threads | KAM-032, measured on the M4 (Metal) |
| macOS | CPU, Metal | release this tree's builds (fast CPU build; Metal with `--block`, jemalloc, overlap) as new app versions | 1.2x (M4, macOS 27) to 1.9x (M1, macOS 15) CPU app; 1.1x-1.7x Metal | above; the shipped binaries were older than master (§3 of 2026-10-16) |
| macOS, Linux arm64 | all | `-ffp-contract=off` in every build line if results are compared across builds, decide | no speed change; stable results across compilers | KAM-031, KAM-025 |
| Windows (MinGW-w64) | CPU | build with `--strip` | about 1.3x (emulated-TLS model; unmeasured on Windows) | cross_check.sh: compiles, same lines |
| Windows (MinGW-w64) | OpenCL | build with `--block` | about 1.2x on the completion step (model) | cross_check.sh, thread test under Wine |
| Windows | all | measure the Windows heap and mimalloc on a real Windows host | unknown (jemalloc gave 1.13x on macOS 27, 1.1x on glibc) | not measured |
| Linux | CPU | link jemalloc or mimalloc; keep GCC (clang 14 was 1.33x slower) | about 1.1x (arm64 container) | Other platforms (M4) |
| Linux, Windows | CUDA, OpenCL | split each round with `complete_batch(first, count)` as the Metal app does | up to the GPU's share of a round (Metal: 1.03x on the M4, 1.13x on the M1) | needs a GPU host to test |
| Android | CPU | `--strip` when built with an NDK below API 29 (emulated TLS) | about 1.3x (model) | not measured |
| all | all | the search changes behind their switches, decide | 1.2-2x, other minima | Search changes above |

## Results on an Apple M4 (2026-11-05)

The same Apple M4 Pro as the day before (macOS 27.0, Apple clang 21.0.0 from the Command Line Tools, Homebrew jemalloc
5.4.0, `-O2`). The BOINC client (8.2.11) was running with its tasks suspended (0 % CPU); the load average was 2.5 to 6,
of which about 1 is the window server and the BOINC Manager window and the rest the benchmark itself (four whole cells
at once in the whole-cell tables, one run otherwise). Spreads were below 2 % in every interleaved table. "Fast build"
means `--strip` and jemalloc for the CPU app and `--block` and jemalloc for the Metal app, as on 2026-11-04; the
baseline is the branch tip of 2026-11-04 (fa03bd1) built the same way. Tables of 3 rounds are interleaved, medians,
verdict by the rule of this file; whole cells are one run per build.

### History-free evaluations (`-DKAMPYLOS_RESET_STATE=1`, off by default)

VBMicrolensing's root solver starts every solve from the previous solve's roots (`zr` in `NewImages`), which is why a
seed's fit depends on the seeds before it (KAM-030) and a GPU app's result on its number of completion threads
(KAM-012). `vendor/patch_vbm_thread_local.py --reset` adds `ResetRootWarmStart()` to the library, and with the switch
every candidate evaluation, every final recomputation of a fit and every cell finalization starts from a fresh
thread's state (`src/kampylos_vbm_state.h`). `tests/test_vbm_state_reset.cpp` checks it: 300 candidates of the planet
cell in either order give the same bits through the CPU objective and through the GPU completion step, while without
the reset 153 of the 300 differ; seed 96 of the OGLE cell gives 2833.4503 alone and after seeds 0-72. It passes with the
`--strip` and the `--block` library.

| what | switch off (this tree) against the branch tip | switch on |
|---|---|---|
| CPU app, `planet` / `ogle` / `far` (3 rounds) | 15.92 / 6.92 / 12.56 s against 15.91 / 6.92 / 12.63 s, same output | same line on `planet` and `ogle`; `far` chi2 -0.1119 (another, better minimum) |
| Metal app, `planet-gpu` (3 rounds, 1 thread) | 64.01 s against 63.72 s, same output | chi2 +0.0001 |
| whole planet, OGLE and far cells (CPU app) | | the same result lines as the branch tip (chi2 411.2228, 2810.4678, 714.1706) |
| fixed work: 300 candidates, objective / completion step | | 0.160-0.165 s / 0.100-0.103 s with and without the reset |

The reset costs nothing: on a fixed set of candidates the time is the same. The wall time of a profile still moves (CPU
app `planet` 1.13x faster, `far` 1.16x slower, Metal `planet-gpu` 1.07x slower) because the seeds take other paths: the
instrumented build below counts 16956 evaluations instead of 20184 on `planet` and 22338 instead of 20404 on `far`.
That is the one-time change of results that the switch makes, and the reason it needs a new app version.

The completion threads no longer change the result (Metal app, `planet-gpu`, one run each):

| completion threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| switch on: seconds, result | 67.83 | 40.47 | 26.30 | 19.51, the same line at every count |
| branch tip | 64.47 | | 25.28 (chi2 -0.0006 against 1 thread) | |

### Where the time goes now

* **CPU app** (fast build with the reset, `sample` of the planet cell): the polynomial root finders 61 %
  (`cmplx_laguerre2newton` 39 %, `cmplx_newton_spec` 16 %, `cmplx_roots_gen` 4 %, `solve_quadratic_eq` 2 %),
  `NewImages` 13 %, ordering the images and the skip lists 8 %, allocation 5.5 % plus 1.5 % in jemalloc's thread lookup
  (`pthread_getspecific`).
* **Metal app** (`planet-gpu`, one completion thread): the completion thread has the same mix (root finders about 60 % of
  its samples); the GPU waits most of the time. A GPU version of the finite-source integration would therefore be a
  kernel of root solves at fixed contour samples, with the image ordering on the host. Not started: it changes results
  at the accuracy of the integration and needs an fp64 GPU for CUDA and OpenCL.
* **Evaluations by kind** (`scripts/bench/patches/censor-count.patch`, profile `planet`, branch tip): 20184 evaluations,
  47 % reflections, 6 % expansions, 25 % contractions, 20 % shrink vertices, 1 % initial vertices. The finite-source
  points are 67 % of the cost on `planet`, 20 % on `ogle` and 55 % on `far`.
* **The anchor stage** takes 2.44 s on the planet light curve and 1.08 s on the OGLE one (fast build): 15 % of a 10-seed
  profile, under 1 % of a whole cell. It matters only for work units split into many short parts.
* **BOINC priority on a Mac.** `nice -n 19` runs at full speed (13.85 s against 13.75 s on `planet`); background QoS
  (`taskpolicy -b`) puts the process on the efficiency cores and takes 3.76x as long (51.69 s). The BOINC client 8.2.11
  binary imports `setpriority()` and no QoS call, and its one running task had nice 10, so the client lowers the
  priority with nice: volunteers' tasks keep the performance cores, and the numbers in this file apply to them (KAM-038).

### Censored evaluations: measured, not worth doing

Nelder-Mead compares a trial point's chi2 with a value it knows beforehand and never stores a rejected one (a
reflection is rejected at or above the second-worst vertex, an expansion at or above the reflection, a contraction at or
above the worst vertex). An evaluation could stop as soon as a lower bound of its chi2 (the least-squares chi2 of the
points computed so far, with their own best fs and fb) reaches that value. `scripts/bench/patches/censor-count.patch`
counts how much finite-source work that would save, with the points in time order and with the point-source points
first:

| profile | rejected trial points | cost saved, point-source points first | cost saved, time order | bound above the true chi2 |
|---|---|---|---|---|
| `planet` | 6561 of 15869 | 0.6 % | 0.5 % | never |
| `ogle` | 5744 of 13987 | 0.0 % | 0.0 % | never |
| `far` | 6376 of 14884 | 0.2 % | 0.4 % | never |

A rejected trial point's chi2 is close above its threshold, and the bound from the point-source points alone is far
below it, because the finite-source points carry the fit. Dropped; nothing in the apps changed.

### Free lists for VBMicrolensing's small objects (`--pool`, same results, KAM-036)

`vendor/patch_vbm_thread_local.py --pool` gives the seven small classes the magnification allocates per point and per
contour sample (`_point`, `_curve`, `_skiplist_curve`, `_theta`, `_thetas`, `annulus`, `_sols_for_skiplist_curve`) an
`operator new`/`delete` with a free list (plain statics in the `--strip` form, `thread_local` in the thread-safe forms).
Every pair below wrote the same result lines; `tests/test_vbm_threads.cpp` passes with the `--block` library and its
thread_local free lists.

| build (switch on) | `planet` median s | `ogle` median s | output |
|---|---|---|---|
| CPU app, jemalloc | 14.10 | 6.95 | |
| + free lists | 12.85 (1.10x, ACCEPT) | 6.31 (1.10x, ACCEPT) | same |
| CPU app, macOS system allocator | 15.82 | 7.89 | same |
| + free lists | 12.94 (1.22x, ACCEPT) | 6.38 (1.24x, ACCEPT) | same |
| CPU app, `--block` with `-femulated-tls` (the MinGW-w64 model), system allocator | 17.43 | 8.59 | same |
| + free lists | 14.79 (1.18x, ACCEPT) | 7.26 (1.18x, ACCEPT) | same |
| Metal app, `planet-gpu`, jemalloc | 66.85 | | |
| + free lists | 61.45 (1.09x, ACCEPT) | | same |

With the free lists the system allocator is as fast as jemalloc (12.94 s against 12.85 s): builds that cannot link
jemalloc (Windows, where it was never measured, and Android) get the allocator gain without it. On Linux arm64 in the
container (`linux_ab.sh`, GCC 12.2, glibc 2.36, `planet`, 3 rounds, switch off) the free lists beat jemalloc: plain
statics 18.88 s, with jemalloc 17.10 s (1.10x), with the free lists 16.27 s (1.16x), same output (with the switch on as
well, 14.56 s, the same line; the difference to 16.27 s is the reset's other search paths on this profile, as on the
Mac, not a speed-up). The Windows heap was not measured.

### Threaded CPU app (seeds on N threads, needs the switch and the thread-safe library)

`src/kampylos_cpu_seed_pool.h`: with a CPU budget of N (the client's `ncpus`, an `mt` plan class's `avg_ncpus`) the CPU
app fits the seeds of a cell on N threads, each taking the next seed when it is free, and merges them in seed order, so
progress, seed checkpoints and the result are those of one thread. It needs the thread-safe library (`--block` here)
and history-free evaluations, and otherwise keeps one thread. The whole planet cell (240 seeds), one run each:

| build | wall s | CPU s | against 1 thread | output |
|---|---|---|---|---|
| `--strip`, 1 thread (today's CPU app, with the switch) | 312.45 | 312.0 | | |
| `--block`, 1 thread | 350.22 | 349.3 | 1.00x | same |
| 2 threads | 183.38 | 359.3 | 1.91x | same |
| 4 threads | 95.93 | 361.9 | 3.65x | same |
| 8 threads | 54.15 | 383.9 | 6.47x | same |

Every thread count wrote the branch tip's line for this cell, byte for byte. The thread-safe library costs 12 % on one
thread (`--strip` is the faster single-thread form); with 8 threads the task finishes 5.8x sooner than today's CPU app
for 23 % more CPU time. Whether to offer it is a plan-class decision (KAM-040). While the stub reported the task as
suspended, the 4-thread app used 0.05 s of CPU in 10 s (KAM-027).

### Search changes on whole cells, with history-free evaluations

All builds but the first have the switch on; one run each, four at a time (wall seconds, the CPU time is within 0.5 % of
it; speed-ups against the switch-on build; "same" = the branch tip's result line, byte for byte):

| build | planet cell | OGLE cell | far cell |
|---|---|---|---|
| branch tip (switch off) | 326 s, chi2 411.2228 | 174 s, chi2 2810.4678 | 246 s, chi2 714.1706 |
| switch on | 326 s, same | 169 s, same | 247 s, same |
| half the seeds (`KAMPYLOS_SEEDS_HALF=1`) | 156 s (2.09x), chi2 411.2228 (the mirror solution) | 82 s (2.06x), **chi2 2833.4503 (+22.98)** | 131 s (1.88x), **chi2 748.6989 (+34.53)** |
| `KAMPYLOS_VBM_RELTOL=1e-3` | 325 s, same | 169 s, same | 243 s, same |
| `KAMPYLOS_VBM_RELTOL=1e-4` | 324 s, same | 169 s, same | 246 s, same |
| restart only after a gain above 1e-3 (`nm-stopping.patch`) | 271 s (1.20x), same | 138 s (1.22x), same | 213 s (1.16x), same |
| that and stop a pass stalled for 100 iterations at 1e-4 | 168 s (1.94x), same | 76 s (2.23x), same | 122 s (2.02x), chi2 714.1706, other digits |

The same builds on the 10-seed profiles (3 rounds, against the switch-on build):

| build | `planet` | `ogle` | `far` |
|---|---|---|---|
| switch on | 14.30 s | 6.97 s | 14.62 s |
| `KAMPYLOS_VBM_RELTOL=1e-3` | 14.73 s, same | 6.98 s, same | 13.18 s (1.11x), other line, same chi2 to 4 decimals |
| `KAMPYLOS_VBM_RELTOL=1e-4` | 14.30 s, same | 6.97 s, same | 14.66 s, same |
| restart rule | 13.15 s (1.09x), same | 6.24 s (1.12x), same | 13.98 s (1.05x, no clear gain), same |
| restart and stall rules | 9.68 s (1.48x), same chi2 | 3.85 s (1.81x), same chi2 | 7.90 s (1.85x), same chi2 |

* **Half the seeds: not proposed.** Each seed set is 120 mirror pairs, (u0, alpha) and (-u0, -alpha), which give the same
  light curve (`src/binary_fit.h`, "MIRROR PAIRS"; `tests/test_mirror_seeds.cpp` checks the pairing and that the two
  starts of a pair have the same chi2 to 1e-9). Dropping one of each pair halves the work, and on the planet cell the
  kept half found the same minimum in its mirror form (u0 +0.100075, alpha 0.135638 instead of -0.100075, 6.147547). On
  the OGLE and far cells it lost the best minimum by 23 and 35 in chi2: the two runs of a pair start with different
  simplex steps and are independent attempts, and both are needed. A server that compares results should treat
  (u0, alpha) and (-u0, 2 pi - alpha) as the same solution (KAM-039).
* **A relative tolerance for the finite-source integration: no effect here.** VBMicrolensing's `RelTol` loosens the
  integration only where it is coarser than the absolute `Tol` of 1e-2, that is above a magnification of 10 (1e-3) or 100
  (1e-4); the finite-source points of these three cells are below that. It may help a high-magnification event; not
  measured.
* **The stopping rules: the strongest search change measured so far.** With history-free evaluations, restarting only
  after a gain above 1e-3 and stopping a pass after 100 iterations without a gain of 1e-4 takes 1.94-2.23x less CPU on
  the three whole cells and gives the same cell chi2 to four decimals on all three (the same bytes on the planet and
  OGLE cells). On the M1 without the reset the same rules moved the OGLE cell by -0.94 (2026-11-04): each seed now ends
  where its own rule takes it, not where its predecessor's state does. A decision for the maintainers, with more cells
  (the regression suite's real events) before it goes into an app (KAM-037). The GPU driver has the same switches: the
  Metal app with both rules took 27.20 s on `planet-gpu` against 67.10 s (2.47x, 3 rounds, ACCEPT), chi2 0.0005 lower.

### CUDA and OpenCL: written, not measured (KAM-035, KAM-011)

The Metal app's split of a round (chunk k+1 on the GPU while the host completes chunk k) is now in the CUDA and OpenCL
hosts as well, behind `-DKAMPYLOS_CUDA_PIPELINE=1` and `-DKAMPYLOS_OPENCL_PIPELINE=1`, off by default; the chunk rule is
shared (`kampylos_gpu_round_chunk()`), and the Metal app uses it with the same constants as before. Not run: the CUDA
code was not compiled (no toolkit here), the OpenCL code passes a syntax check against Apple's OpenCL headers and a
stand-in for `cl_dynload.h`, which must provide `clFlush`.

### GPU app against CPU app by task size (from the measurements above, one completion thread)

| seeds per task | CPU app | Metal app | faster |
|---|---|---|---|
| 10 (`far`, 2026-11-04) | 12.5 s | 16.5 s | CPU app, 1.32x |
| 60 (`planet-gpu`; the CPU app's time estimated from its whole planet cell: 1.35 s per seed plus the anchor) | about 83 s | 63.7 s | Metal app, about 1.3x |
| 240 (the whole planet cell; Metal 2026-11-04) | 326 s | 293 s | Metal app, 1.11x |

On a caustic-crossing cell a GPU task with one completion thread is only as fast as a CPU task, and on a few seeds it
is slower. GPU tasks want whole cells and more completion threads (KAM-032); CPU tasks can be split finer.

### Checks of this tree (2026-11-05, M4)

* `tests/test_vbm_state_reset.cpp` passes with the `--strip --reset` and the `--block --reset` library; `test_cpu_resume`,
  `test_batched_resume`, `test_nelder_mead_stepper` and `test_mirror_seeds` (also with `-DKAMPYLOS_SEEDS_HALF=1`) pass
  with the switches off and on; `tests/test_vbm_threads.cpp` passes with the `--block --pool` library.
* `scripts/bench/check_change.sh`: PASS (library thread test, Metal objective test, Metal kill-and-resume test in 214 s,
  the same result lines with 4 completion threads as `upstream/master`'s Metal build).
* `tests/test_boinc_resume.sh` with the threaded CPU app (4 threads, switch on): the killed and restarted run and the
  stale-checkpoint run are byte-identical to the uninterrupted one.
* With the switches off this tree writes the branch tip's result lines on every profile above (CPU and Metal apps),
  checked again after the last change (`planet`, `ogle`, `far`, `planet-gpu`, within 1.6 % in time).
* `scripts/bench/cross_check.sh planet 1 "fork block strip strip+reset+pool block+reset+pool"` (forms with `+` are new):
  the CPU app for Windows (MinGW-w64 GCC 16.2, static, run by Wine) and Linux x86_64 (GCC 12.2), both under Rosetta,
  compiles, links and runs to the end in all five forms, including the threaded CPU app's library form and the new
  switches; within each platform every form writes the same lines (chi2 411.3205 on Windows, 411.3207 on Linux x86_64,
  as on 2026-11-04). `scripts/bench/linux_ab.sh 3` (Linux arm64, GCC 12.2): every build the same lines.
* Not run: CUDA and OpenCL (no fp64 GPU here), the regression suite (it needs the server's `kampylos_analyze.py`).

### Next steps (2026-11-05)

| platform | app | step | expected | evidence |
|---|---|---|---|---|
| all | all | decide: history-free evaluations (`KAMPYLOS_RESET_STATE=1` with `--reset`) in the next app versions | no speed change; exact resumes and seed-split parts; GPU results independent of the thread count; needed for the threaded CPU app | above, KAM-030, KAM-012 |
| all | all | release builds with `--pool` | 1.09-1.10x with jemalloc, 1.18-1.24x without it | above |
| all | CPU | decide: an `mt` plan class for the CPU app (seed threads, KAM-040) | 1.9x, 3.7x, 6.5x shorter tasks at 2, 4, 8 threads, 3-10 % more CPU | above |
| all | all | decide: the stopping rules, after the regression suite's real events | about 2x on whole cells, same chi2 on three cells | above |
| Linux, Windows | CUDA, OpenCL | build with `KAMPYLOS_CUDA_PIPELINE` / `KAMPYLOS_OPENCL_PIPELINE` on a GPU host, compare and time | up to the GPU's share of a round | KAM-035 |
| all GPU apps | Metal, CUDA, OpenCL | decide: `avg_ncpus` of the GPU plan classes (KAM-032); with the switch on the result no longer depends on it | 1.7-3.5x on caustic cells | above |

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

### Measured on 2026-11-04 (Apple M4 Pro, quiet machine)

The CPU app on profile `planet`. Linux arm64 runs in a Debian 12 container of Docker Desktop at native speed;
x86_64 Linux and Windows run under Rosetta in an x86_64 container (`scripts/bench/cross_check.sh`), so their times are
not speed results: they show that each build compiles, links, runs to the end and which result lines it writes.

| setup | thread_local library (fork) | `--block` | plain statics | allocator | output |
|---|---|---|---|---|---|
| macOS 27, Apple clang 21 (results above) | 19.28 s | 19.42 s (0.99x) | 17.99 s (1.07x) | jemalloc: fork 1.13x, plain statics 1.21x | same |
| macOS 27, Apple clang 21 with `-femulated-tls` (the MinGW-w64 GCC model) | 23.22 s, 0.82x of the native fork (19.00 s in the same runs) | 19.40 s (**1.20x** against the emulated fork) | 17.64 s (1.32x) | not measured | same |
| Linux arm64 container (`linux_ab.sh`), GCC 12.2, glibc 2.36, 3 rounds | 18.63 s | 19.49 s (0.96x) | 18.66 s (1.00x) | plain statics + jemalloc 16.80 s (**1.11x**, ACCEPT), + mimalloc 16.90 s (1.10x, ACCEPT); `--block` + jemalloc 17.63 s (no clear gain) | same |
| Linux arm64 container, Debian clang 14.0.6 | 24.81 s (**0.75x** of GCC) | | | plain statics + jemalloc 23.02 s | same as each other, not as GCC |
| x86_64 Linux container under Rosetta, GCC 12.2, 1 round | 33.3 s | 34.0 s | 32.9 s | | chi2 411.3207, same for all three |
| Windows x86_64: MinGW-w64 GCC 16.2, `-static`, under Wine 8.0 under Rosetta, 1 round | 136.9 s | 44.4 s | 42.1 s | | chi2 411.3205, same for all three |

* **Linux: keep GCC.** On a quiet machine clang 14 was 1.33x slower than GCC 12 in every run (the M1's noisy hint
  holds). With glibc the allocator gains about 10 % for the CPU app (jemalloc or mimalloc), a gain the loaded M1 could not
  show; the thread_local forms change nothing.
* **Windows builds compile and run** in all three library forms and write identical result lines; `--block` and
  `--strip` need nothing special from MinGW-w64. MinGW-w64 GCC 16.2 still implements `thread_local` with emulated TLS
  (`__emutls_get_address`; checked in the assembly on 2026-11-04). Under Wine the fork form took 3.1x as long as
  `--block`; that is Wine's `TlsGetValue` under Rosetta, not a Windows figure, but it points the same way as the
  `-femulated-tls` model (1.20x). The library thread test (`tests/test_vbm_threads.cpp`, 4 threads) passes as a MinGW
  build under Wine with the fork and with `--block` (Wine's heap may not catch the corruption KAM-010 saw on Windows).
* **Result lines by platform** on `planet` (one cell, 10 seeds): macOS clang 21 default 411.3206, macOS clang 21 with
  `-ffp-contract=off` and the M1's clang 14 411.3205, Windows MinGW GCC 16 411.3205 (no FMA on x86_64 without
  `-mfma`, but another libm), Linux x86_64 GCC 12 411.3207, Linux arm64 GCC 12 and clang 14 411.3205 with two other hashes (other digits of the line). Every
  platform agrees with itself across library forms, allocators and flags (except PGO of the Metal app with contraction on,
  above). Facts for KAM-025.
* **Not measured:** Windows on real hardware (the heap, emulated TLS at native speed), MSVC and llvm-mingw builds, Linux
  x86_64 at native speed, Android, the CUDA and OpenCL apps.

### What carries over

* **thread_local costs differ by platform and by OS version**, and so does the gain of `--block`. macOS: every access
  is a call (`_tlv_get_addr`); `--block` gained 1.20x in the CPU app and 1.30x in the Metal app on the M1 (macOS 15),
  nothing on the M4 Pro (macOS 27), where the call has become cheap. Keep `--block`: most volunteers' Macs are older. Emulated TLS (MinGW-w64 GCC,
  Android NDK below API 29): a call to `__emutls_get_address` per access, the most expensive form; `--block` gained 1.26x
  (M1) and 1.20x (M4) in the `-femulated-tls` model. MinGW-w64 GCC 16.2 (2026) still emulates TLS.
  Linux, thread_local in the executable: a load relative to the thread pointer; neither `--block` nor plain statics
  gained anything. Windows with MSVC, and llvm-mingw (native Windows TLS): not measured; a few loads through the
  thread's TLS array, so expect something between Linux and macOS. `--block` keeps the same objects with the same
  destructors, so the rule that completion threads never exit (KAM-010) applies unchanged.
* **`--block` compiles** with Apple clang 14 and 21, GCC 12 (arm64 and x86_64), clang 14 and MinGW-w64 GCC 16.2, and
  gives the fork's result lines with each. MSVC and the host compilers of the CUDA builds were not tried.
* **The allocator** matters most on macOS (its system allocator took about a third of the CPU app). With glibc,
  jemalloc and mimalloc gained 5 to 9 % on the loaded M1, within noise, and 10-11 % on the quiet M4 (ACCEPT). The
  Windows heap was not measured.
* **GPU apps on caustic-crossing cells are CPU-bound** on every backend: CUDA and OpenCL use the same completion step
  (`kampylos_gpu_complete.h`, `kampylos_cpu_patch_pool.h`). A faster GPU does not help such a cell; the completion
  threads do, and their number is the plan class's `avg_ncpus` (0.1-0.25 today, so one thread). On the M4 Pro 2, 4
  and 8 threads made `planet-gpu` 1.66x, 2.52x and 3.38x faster for 3-9 % more CPU time (Results on an Apple M4). The overlap of GPU
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
