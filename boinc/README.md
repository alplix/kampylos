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

Add `-DKAMPYLOS_VERSION='"12"'` (CPU; `KAMPYLOS_GPU_VERSION`, `KAMPYLOS_OPENCL_VERSION`, `KAMPYLOS_METAL_VERSION` for the
GPU apps) so the version the app prints to stderr is the `version_num` the server assigns to that binary.

Run it in a directory with the two input files the app expects, `in` (one line, see `src/kampylos_wu.h`) and
`photometry` (the event's light curve); it writes `out`. Outside a BOINC client it runs standalone.

## Checkpoints and resume

Two files in the slot directory, both written atomically (temporary file + rename):

| file | contents |
|---|---|
| `kampylos_checkpoint` | number of grid cells completed and written to `out`; written after every cell |
| `kampylos_seed_checkpoint` | CPU app only: cell index, next seed, best fit so far (`src/kampylos_cpu_cell.h`); written whenever BOINC asks for a checkpoint, removed when the cell completes |

On a restart the app resumes from `min(kampylos_checkpoint, complete lines in out)`; the result file is the
authoritative record of finished cells, so a missing or stale checkpoint file never costs a finished cell.

## Tests

- `tests/test_cpu_resume.cpp` -- the resumable seed loop and the result-file helpers (seconds).
- `tests/test_boinc_resume.sh` -- builds the real CPU wrapper against the fake BOINC runtime in `tests/boinc_stub/`,
  kills it with SIGKILL at random moments and checks the resumed `out` is byte-identical to an uninterrupted run (minutes).
