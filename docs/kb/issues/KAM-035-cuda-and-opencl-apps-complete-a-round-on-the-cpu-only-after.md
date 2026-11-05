---
id: KAM-035
title: CUDA and OpenCL apps complete a round on the CPU only after the GPU has finished all of it
type: bug
status: confirmed
severity: low
platforms: [cuda, opencl]
area: performance
action:
next: Split each round with complete_batch(first, count) in gpu_fit_binary.cu and kampylos_opencl_fit.cpp as kampylos_metal_fit.mm does; measure on a GPU host
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

## Test

On a GPU host: `planet-cell` with and without the split, same result line; the batched resume test.

## Log

- 2026-11-04: recorded with pattern PAT-033.
