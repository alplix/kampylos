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

## Checkpoints and resume

Two files in the slot directory, both written atomically (temporary file + rename):

| file | contents |
|---|---|
| `kampylos_checkpoint` | number of grid cells completed and written to `out`; written after every cell |
| `kampylos_seed_checkpoint` | CPU app only: cell index, next seed, best fit so far (`src/kampylos_cpu_cell.h`); written whenever BOINC asks for a checkpoint, removed when the cell completes |
| `kampylos_batched_checkpoint` | GPU apps only: the batched Nelder-Mead driver's full state between two rounds (`src/kampylos_batched_fit.h`), a few hundred KB of text; same timing as the seed checkpoint |

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
