# BOINC wrappers

The BOINC application front ends of Kampylos. The search itself lives in `../src`; these files only handle the
work-unit files, progress, checkpointing and result upload.

| file | backend |
|---|---|
| `main_kampylos_boinc.cpp` | CPU (all platforms); supports seed-split work units (`part i K` in the input line) |
| `main_kampylos_gpu_boinc.cpp` | CUDA |
| `main_kampylos_opencl_boinc.cpp` | OpenCL |
| `main_kampylos_metal_boinc.cpp` | Apple Metal |
| `kampylos_boinc_common.h` | shared input/output/checkpoint handling |

## Build (CPU, Linux example)

You need the BOINC client libraries (`libboinc_api`, `libboinc`) built for your target and the VBMicrolensing sources
(`git submodule update --init`).

```
VBM=vendor/VBMicrolensing/VBMicrolensing/lib
g++ -O2 -std=c++17 -I src -I $VBM -I <boinc>/api -I <boinc>/lib -c boinc/main_kampylos_boinc.cpp -o main.o
g++ -O2 -std=c++17 -I $VBM -c $VBM/VBMicrolensingLibrary.cpp -o vbm.o
g++ main.o vbm.o <boinc>/api/libboinc_api.a <boinc>/lib/libboinc.a -lpthread -o kampylos
```

Run it in a directory with the two input files the app expects, `in` (one line, see `src/kampylos_wu.h`) and
`photometry` (the event's light curve); it writes `out`. Outside a BOINC client it runs standalone.

## Build (Metal, macOS)

```
VBM=vendor/VBMicrolensing/VBMicrolensing/lib
clang++ -O2 -std=c++17 -fobjc-arc -I src -I $VBM -I <boinc>/api -I <boinc>/lib \
    boinc/main_kampylos_metal_boinc.cpp src/kampylos_metal_fit.mm $VBM/VBMicrolensingLibrary.cpp \
    <boinc>/api/libboinc_api.a <boinc>/lib/libboinc.a -framework Metal -framework Foundation -o kampylos_metal
```

`-fobjc-arc` is required: `kampylos_metal_fit.mm` relies on ARC to release Metal buffers it replaces. The kernel is
compiled at start-up with fast math off and a self-test refuses to run if the compiler broke the double-single
arithmetic; every command buffer is checked and re-submitted on failure (`src/kampylos_metal_fit.mm`). Validate a
build with `tests/test_metal_fit.mm` and `src/fit_event.cpp -DKAMPYLOS_CLI_METAL` under `tests/regression/run_regression.py`.
