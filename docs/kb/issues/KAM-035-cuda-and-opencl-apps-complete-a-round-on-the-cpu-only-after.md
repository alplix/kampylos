---
id: KAM-035
title: CUDA and OpenCL apps complete a round on the CPU only after the GPU has finished all of it
type: bug
status: fix-proposed
severity: low
platforms: [cuda, opencl]
area: performance
action:
next: On an fp64 GPU host: build with -DKAMPYLOS_CUDA_PIPELINE=1 / -DKAMPYLOS_OPENCL_PIPELINE=1, compare lines and time with the plain build; check cl_dynload has clFlush
blocked_by: []
pattern: PAT-033
aliases: []
reported: 2026-11-04
source: read 2026-11-04; the Metal app's overlap measured on the M1 and the M4 (docs/BENCHMARK.md)
where: src/gpu_fit_binary.cu:43-51, src/kampylos_opencl_fit.cpp:168-191 @ faba573
fixed_in:
---

## Symptom

Wasted time, no wrong result: in the CUDA and OpenCL apps the GPU idles while the CPU completes a round's
finite-source points, and the CPU idles while the GPU evaluates the next round.

## Cause

Both backends launch every candidate of a round (or of a 15 s chunk) at once and wait (`cudaMemcpy` in
`src/gpu_fit_binary.cu:43-51`, `clFinish` and blocking reads in `src/kampylos_opencl_fit.cpp:168-191` at faba573)
before `kampylos_complete_candidate()` runs on the host (PAT-033). The Metal app had the same loop until this branch.

## Fix

Proposed, not done: cut each round into chunks as `src/kampylos_metal_fit.mm` does and complete chunk k with
`complete_batch(first, count)` while chunk k+1 runs on the GPU; each candidate stays on the same completion thread in
the same order, so the results stay bit-identical. Measured for Metal: 1.12-1.14x on a whole cell on an M1, 1.03x on an
M4 Pro (docs/BENCHMARK.md). It should gain more where the GPU's share of a round is larger. Needs a CUDA or OpenCL host
with fp64 to test; not done blind (backend symmetry is kept by this record, AGENTS.md "Conventions").

Written on 2026-11-05, off by default until it is measured on a GPU host: `-DKAMPYLOS_CUDA_PIPELINE=1`
(`src/gpu_fit_binary.cu`) and `-DKAMPYLOS_OPENCL_PIPELINE=1` (`src/kampylos_opencl_fit.cpp`) cut a round with the
Metal app's rule (`kampylos_gpu_round_chunk()` in `src/kampylos_batched_fit.h`, now shared by all three) and complete
chunk k with `complete_batch(first, count)` while chunk k+1 runs. Each chunk's results are read back (blocking, one
in-order stream or queue) before the next chunk is launched, so no events or second streams are needed; the OpenCL
kernel indexes candidates by work-group id, so every chunk goes to the start of the buffers, and `clFlush` after each
launch makes the overlap real (this needs `cl_api.Flush` in cl_dynload, which is not in this repository). The CUDA code
was not compiled (no toolkit here); the OpenCL code passes a `-Wall` syntax check with both settings against Apple's
OpenCL headers and a stand-in `cl_dynload.h`, and was never run.

## Test

On a GPU host: `planet-cell` with and without the split, same result line; the batched resume test.

## Log

- 2026-11-04: recorded with pattern PAT-033.
- 2026-11-05: 2026-11-05: opt-in round split written for CUDA and OpenCL (off by default); OpenCL syntax-checked only, CUDA not compiled
