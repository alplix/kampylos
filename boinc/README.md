# BOINC wrappers

The BOINC application front ends of Kampylos. The search itself lives in `../src`; these files only handle the
work-unit files, progress, checkpointing and result upload.

| file | backend |
|---|---|
| `main_kampylos_boinc.cpp` | CPU (all platforms) |
| `main_kampylos_gpu_boinc.cpp` | CUDA |
| `main_kampylos_opencl_boinc.cpp` | OpenCL |
| `main_kampylos_metal_boinc.cpp` | Apple Metal |

All four accept seed-split work units (`part i K` at the end of the input line: this task fits every K-th seed of each
cell starting at seed i, the server keeps the lowest chi2 per cell over the K parts). The CPU app has since v12; the GPU
wrappers since 2026-10-19 (`kampylos_seed_part()` in `src/binary_fit.h`) -- GPU binaries built before that refuse such a
work unit. Splitting is how a cell that needs the finite-source integration for most of its points (single-lens anchor
with `n_peak` close to `npts`) is cut into tasks of a few hours instead of days, on GPU hosts as on CPU hosts.
| `kampylos_boinc_common.h` | shared input/output/checkpoint handling |

## Build (CPU, Linux example)

You need the BOINC client libraries (`libboinc_api`, `libboinc`) built for your target and the VBMicrolensing sources
(`git submodule update --init`, pinned to upstream v5.6).

For the GPU apps (CUDA, OpenCL, Metal) also run `python3 vendor/patch_vbm_thread_local.py` once after the submodule
checkout: it makes the library's function-local statics `thread_local`, which the completion step's worker threads
need (`src/kampylos_cpu_patch_pool.h`). Without the patch the library header lacks `VBM_THREAD_LOCAL_STATICS` and the
pool falls back to one thread automatically; `tests/test_vbm_threads.cpp` shows the difference.

```
VBM=vendor/VBMicrolensing/VBMicrolensing/lib
g++ -O2 -std=c++17 -I src -I $VBM -I <boinc>/api -I <boinc>/lib -c boinc/main_kampylos_boinc.cpp -o main.o
g++ -O2 -std=c++17 -I $VBM -c $VBM/VBMicrolensingLibrary.cpp -o vbm.o
g++ main.o vbm.o <boinc>/api/libboinc_api.a <boinc>/lib/libboinc.a -lpthread -o kampylos
```

Add `-DKAMPYLOS_VERSION='"12"'` (CPU; `KAMPYLOS_GPU_VERSION`, `KAMPYLOS_OPENCL_VERSION`, `KAMPYLOS_METAL_VERSION` for the
GPU apps) so the version the app prints to stderr is the `version_num` the server assigns to that binary.

Run it in a directory with the two input files the app expects, `in` (one line, see `src/kampylos_wu.h`) and
`photometry` (the event's light curve); it writes `out`. Outside a BOINC client it runs standalone.

**The compiler version changes the last digits.** No build line sets `-ffp-contract`, so each compiler fuses
multiply-adds as its default says: GCC on arm64 does (thousands of FMAs in VBMicrolensing), GCC on x86_64 without `-mfma`
cannot, Apple clang 14 did not, and Apple clang 21 (Command Line Tools for macOS 27) does. Measured on 2026-11-04 on an
Apple M4: the same source gives chi2 411.3206 on the planet profile with clang 21's default and 411.3205 with
`-ffp-contract=off`, the value of the clang 14 builds, at no measured speed cost (`docs/BENCHMARK.md`, "Results on an
Apple M4"). If results of different app versions or platforms are compared (KAM-025), add `-ffp-contract=off` to every
compile line so a new Xcode or GCC cannot move them.

## Build (Metal, macOS)

```
VBM=vendor/VBMicrolensing/VBMicrolensing/lib
clang++ -O2 -std=c++17 -fobjc-arc -I src -I $VBM -I <boinc>/api -I <boinc>/lib \
    boinc/main_kampylos_metal_boinc.cpp src/kampylos_metal_fit.mm $VBM/VBMicrolensingLibrary.cpp \
    <boinc>/api/libboinc_api.a <boinc>/lib/libboinc.a \
    -framework Metal -framework Foundation -framework CoreGraphics -o kampylos_metal
```

`-fobjc-arc` is required: `kampylos_metal_fit.mm` relies on ARC to release Metal buffers it replaces. The kernel is
compiled at start-up with fast math off and a self-test refuses to run if the compiler broke the double-single
arithmetic; every command buffer is checked and re-submitted on failure (`src/kampylos_metal_fit.mm`). Validate a
build with `tests/test_metal_fit.mm` and `src/fit_event.cpp -DKAMPYLOS_CLI_METAL` under `tests/regression/run_regression.py`.

Metal device access: `MTLCreateSystemDefaultDevice()` resolves the display's GPU and, per Apple, needs CoreGraphics
linked (hence the framework above); from a process with no window-server session -- a BOINC task runs as
`boinc_project` under the client daemon, possibly with nobody logged in -- it can return nil although the GPU is there.
The app then takes the device from `MTLCopyAllDevices()`. If there is still no device, or the device will not run the
self-test, or command buffers keep being aborted (another process hogging the GPU), the app calls
`boinc_temporary_exit()` instead of failing: the task and its checkpoints stay, and the client retries after 300 s.

## CPU threads of the GPU apps

The GPU apps finish every candidate on the CPU (`src/kampylos_cpu_patch_pool.h`: the points the kernel's point-source
test rejects go through VBMicrolensing's finite-source integration). Since 2026-10-19 that pool has as many threads as the
client budgeted for the task -- `init_data`'s `ncpus`, i.e. the app version's `avg_ncpus` from the plan class -- rounded
down and at least one, and the app prints the count (`completion step: N CPU thread(s) (client budget ...)`). The plan
classes currently declare 0.1-0.25 CPUs, so a task runs one completion thread; before this the pool always took four
cores, which the client had meanwhile given to other tasks. To let the completion step use more cores, raise `avg_ncpus`
in the plan class (`plan_class_spec.xml`, or the `--avg_ncpus` of the app version); the app follows. Outside BOINC
(`fit_event`, tests) the pool keeps the compile-time default, `KAMPYLOS_PATCH_THREADS` (4 with the thread-safe
VBMicrolensing fork, 1 otherwise). The stub takes the budget from `KAMPYLOS_STUB_NCPUS`.

## Points the GPU kernel leaves to the CPU

The kernels evaluate every candidate with the point-source magnification and VBMicrolensing's own quadrupole test; the
points that fail the test (finite-source effects matter there) are listed per candidate and finished on the CPU with
`BinaryMag2`, exactly as the CPU app computes them (`src/kampylos_gpu_complete.h`). A candidate can list up to 512 such
points. Beyond that the kernel only reports an overflow, and since 2026-10-19 the completion step then computes that
candidate's objective entirely on the CPU, the way the CPU app does; before, such a candidate was declared dead
(chi2 = 1e18). On an event whose single-lens anchor shows no baseline (`n_peak` close to `npts`, e.g. ev25055) that
killed 1100 of a cell's 1440 starting vertices, so the GPU result for the cell came from the few surviving seeds. In
that regime the GPU app now costs about what the CPU app costs; the seed-split work units keep the task length down.

## Checkpoints and resume

Two files in the slot directory, both written atomically (temporary file + rename):

| file | contents |
|---|---|
| `kampylos_checkpoint` | number of grid cells completed and written to `out`; written after every cell |
| `kampylos_seed_checkpoint` | CPU app only: cell index, next seed, best fit so far (`src/kampylos_cpu_cell.h`); written whenever BOINC asks for a checkpoint, removed when the cell completes |
| `kampylos_batched_checkpoint` | GPU apps only: the batched Nelder-Mead driver's full state between two rounds, or inside a long round together with the chi2 already computed for it (`src/kampylos_batched_fit.h`), a few hundred KB of text; same timing as the seed checkpoint |

On a restart the app resumes from `min(kampylos_checkpoint, complete lines in out)`; the result file is the
authoritative record of finished cells, so a missing or stale checkpoint file never costs a finished cell. Inside a
cell both the CPU and the GPU apps resume bit-identically, and report progress after every seed / every round.

## Tests

- `tests/test_cpu_resume.cpp` -- the resumable seed loop and the result-file helpers (seconds).
- `tests/test_batched_resume.cpp` -- the GPU apps' driver, with a CPU evaluator: state round trip, kills at random
  rounds across all passes, bit-identical resume (a few minutes).
- `tests/test_boinc_resume.sh` -- builds the real CPU wrapper against the fake BOINC runtime in `tests/boinc_stub/`,
  kills it with SIGKILL at random moments and checks the resumed `out` is byte-identical to an uninterrupted run (minutes).

## VBMicrolensing

`vendor/VBMicrolensing` is a fork of valboz/VBMicrolensing v5.6 (`alplix/VBMicrolensing`, branch `thread-safe-statics`) in
which every function-local `static` is `thread_local` and `VBM_THREAD_SAFE` is defined. The GPU backends run the exact
finite-source completion on several CPU threads (`src/kampylos_cpu_patch_pool.h`); with the unpatched upstream library the pool
falls back to a single thread because upstream's statics are shared between threads. To build against plain upstream v5.6
instead, run `python3 vendor/patch_vbm_thread_local.py` (it recognises the fork as already patched).

## Faster CPU build (same results)

Two build-level changes make the CPU app about 1.5-2x faster with a byte-identical result file. Neither touches the search
code. Measured on an Apple M1 (macOS 15.7, Apple clang 14, `-O2`) with the synthetic planet work unit of the regression
suite (`flux 0.2 0.2 1 -3 -3 1 log10`, one cell, 240 seeds): 1066 s with the build lines above, 536 s with both changes;
the shipped `kampylos_v12_macos_arm64` took 839 s on the same input. The two result files from the two local builds are
byte-identical (`cmp`), and so are the per-seed fits printed at `%.17g`.

1. **Build VBMicrolensing without the `thread_local` patch.** The CPU app is single-threaded; the patch exists for the GPU
   apps' completion threads. On macOS every access to one of the library's 577 `thread_local` variables is a call to
   `_tlv_get_addr`, 46 % of the CPU app's samples with Apple clang 14 (18 % in the shipped binary, built with a newer
   compiler); with emulated TLS (MinGW-w64 GCC, Android NDK below API 29) a thread_local costs more still. Plain statics
   alone gave 1.35x.
   ```
   python3 vendor/patch_vbm_thread_local.py --strip      # fork -> plain upstream v5.6, VBM_THREAD_SAFE removed
   # ... build the CPU app as above ...
   git -C vendor/VBMicrolensing checkout -- .           # restore the fork before building a GPU app
   ```
   `--strip` also removes `VBM_THREAD_SAFE`, so a GPU app built from a stripped checkout by mistake runs its completion
   step on one thread instead of corrupting the heap.
2. **Link a faster allocator.** VBMicrolensing creates and deletes many small objects per magnification (`_theta`, `_point`,
   `_curve`, image lists, far more inside the finite-source integration); about a third of the CPU app's time went to the
   macOS system allocator. Measured with jemalloc (`libjemalloc.a` on the link line, no source change): 2.0x together with
   step 1. mimalloc is the portable equivalent (Linux, Windows, macOS): `#include <mimalloc-new-delete.h>` in one
   translation unit and link its library. The library allocates with `new`, so overriding `new`/`delete` covers the hot path.

Check a new build before shipping it: build `src/fit_event.cpp` the old way and the new way, run the same work unit through
both (e.g. the synthetic planet one above) and `cmp` the two result files. On one toolchain they must be byte-identical.
Builds from different compilers can differ in the last printed digits, so compare like with like.

These gains are M1 figures (macOS 15). On an Apple M4 Pro with macOS 27 and Apple clang 21 (2026-11-04) the same
build gained 1.21x: a thread_local costs little there (plain statics alone 1.07x) and jemalloc alone gave 1.13x. On
Linux (arm64, GCC 12, glibc 2.36, in a container) plain statics change nothing, because a thread_local costs a single
load there; jemalloc or mimalloc gained 1.10-1.11x on the quiet M4 (within noise on the loaded M1). With emulated TLS
(MinGW-w64 GCC, which still emulates it in version 16.2) plain statics gained 1.65x on the M1 and 1.32x on the M4
(`docs/BENCHMARK.md`, "Other platforms"). The Windows build compiles and gives the same result lines in every library
form (MinGW-w64 GCC 16.2, run under Wine, `scripts/bench/cross_check.sh`).

## Faster GPU app build (same results)

The GPU apps need the thread_local library (their completion threads), and on a cell where the source crosses a caustic
they spend most of their time in it: on 60 seeds of the synthetic planet's cell the Metal app spent 13 s on the GPU and
154 s completing the finite-source points on the CPU (2026-11-04, `docs/BENCHMARK.md`). Two build-level changes, no
change to the search, byte-identical result files:

1. **One thread_local block per function.** `python3 vendor/patch_vbm_thread_local.py --block` keeps every static
   thread_local but gathers each function's statics into one struct, so a call pays one TLS lookup instead of one per
   access. Restore the fork with `git -C vendor/VBMicrolensing checkout -- .` afterwards.
2. **The faster allocator**, as for the CPU app.

Measured on an Apple M1 with `scripts/bench` (profile `planet-gpu`, 3 interleaved runs each, median): Metal app 189.5 s
with the build line above, 145.2 s with `--block`, 138.4 s with jemalloc, 112.0 s with both (1.69x); every run wrote the
same result lines. `scripts/bench/mac_ab.sh` repeats the comparison on any Mac, and `scripts/bench/check_change.sh`
runs the correctness checks. The same two changes apply to the CUDA and OpenCL apps, whose completion step is the same
code; they were not measured there. What they gain depends on the platform (`docs/BENCHMARK.md`, "Other platforms"):
`--block` gained 1.26x with emulated TLS (as MinGW-w64 GCC builds) and nothing with GCC 12 on Linux, and jemalloc gained
nothing clear with glibc. On the M4 Pro (macOS 27) the Metal app gained 1.09x on `planet-gpu` (62.8 s against 68.7 s):
`--block` costs nothing there and gains nothing, jemalloc gives the rest; keep both for older Macs.

The largest lever of the GPU apps is not a build flag but the number of completion threads, which the client's CPU
budget (the plan class's `avg_ncpus`) sets: on the M4 Pro 2, 4 and 8 threads ran `planet-gpu` 1.66x, 2.52x and 3.38x
faster than one, for 3-9 % more CPU time in total. The result lines then differ at rounding level (KAM-012), so it is a
server-side decision; built with history-free evaluations (below) they do not.

The Metal app also overlaps GPU and CPU work inside a round (`src/kampylos_metal_fit.mm`, since 2026-11-04): 1.12-1.14x
on a whole cell on the M1, 1.03x on the M4 Pro, whose faster GPU leaves less to hide; same result line. That is a source change, so it comes with the next Metal app version, whatever the
build line.

## Switches for the next app versions (2026-11-05)

Measured on an Apple M4 Pro (`docs/BENCHMARK.md`, "Results on an Apple M4 (2026-11-05)"). All are off by default, so
the build lines above build what they built before (the same result lines, checked).

* **Free lists for the library's small objects, same results:** `python3 vendor/patch_vbm_thread_local.py --strip --pool`
  (CPU app) or `--block --pool` (GPU apps). The CPU app 1.10x faster with jemalloc and 1.22-1.24x with the system
  allocator, which then is as fast as jemalloc; the Metal app 1.09x; byte-identical result lines. For builds that cannot
  link jemalloc (Windows, Android) this is the allocator gain without it; the Windows heap itself was not measured.
* **History-free evaluations, a one-time change of results:** add `--reset` to the patch script's options and
  `-DKAMPYLOS_RESET_STATE=1` to every compile line of the app. Every candidate evaluation then starts from the same
  library state (`src/kampylos_vbm_state.h`): resumed tasks and seed-split parts give each seed exactly the fit of an
  uninterrupted run (KAM-030), and the GPU apps' result no longer depends on the number of completion threads
  (KAM-012). It costs nothing measurable. It changes results that depended on the order of calls (a 10-seed test moved
  by -0.11 in chi2; three whole cells kept their lines), so it needs a new app version. For example:
  ```
  python3 vendor/patch_vbm_thread_local.py --strip --reset --pool
  g++ -O2 -std=c++17 -DKAMPYLOS_RESET_STATE=1 -I src -I $VBM -I <boinc>/api -I <boinc>/lib -c boinc/main_kampylos_boinc.cpp -o main.o
  g++ -O2 -std=c++17 -I $VBM -c $VBM/VBMicrolensingLibrary.cpp -o vbm.o
  ```
* **A threaded CPU app:** built from the thread-safe library (the fork, or `--block`) with `--reset` and
  `-DKAMPYLOS_RESET_STATE=1`, the CPU app fits the seeds of a cell on as many threads as the client budgets CPUs for the
  task (`init_data` ncpus, the plan class's `avg_ncpus`; `src/kampylos_cpu_seed_pool.h`), with the result of one
  thread. Today's CPU plan class gives one CPU, and the app then runs one thread as before; an `mt` plan class with
  `avg_ncpus` N would finish a task about N times sooner for the same CPU time, which matters most for the day-long
  tasks of events without a baseline. The app prints `seed threads: N` at the start.
* **Helper threads pause while the task is suspended (KAM-027):** every wrapper now points `kampylos_worker_pause()` at a
  wait on `boinc_get_status()`, and the completion and seed threads block SIGALRM. No build change.
* **CUDA and OpenCL round split (KAM-035):** `-DKAMPYLOS_CUDA_PIPELINE=1` / `-DKAMPYLOS_OPENCL_PIPELINE=1` overlap the GPU
  and the host completion as the Metal app does. Not measured, and the CUDA code not even compiled: try it on a GPU host
  before using it (the OpenCL one needs `cl_api.Flush` in cl_dynload).
* **Search changes, for the maintainers:** `scripts/bench/patches/nm-stopping.patch` (restart only after a gain above
  1e-3, stop a pass stalled for 100 iterations at 1e-4) takes about 2x less CPU on three whole cells with the same cell
  chi2, with history-free evaluations on. `-DKAMPYLOS_SEEDS_HALF=1` (half of each mirror pair of seeds) lost the best
  minimum on two of three cells: not proposed. `-DKAMPYLOS_VBM_RELTOL` changed nothing on the measured cells.
