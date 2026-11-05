---
id: PAT-033
title: A GPU app waits for each launch before its host work, so GPU and CPU take turns instead of overlapping
status: active
severity: low
platforms: [cuda, opencl, metal]
area: performance
applies_to: GPU apps that launch a kernel, wait for it (clFinish, cudaDeviceSynchronize, a blocking read, waitUntilCompleted) and then do host work before the next launch
check: 'clFinish|clWaitForEvents|CL_TRUE|cudaDeviceSynchronize|cudaMemcpy\(|cudaStreamSynchronize|waitUntilCompleted' -- src
origin: kampylos docs/BENCHMARK.md (Metal app, 2026-11-04)
reference_fix: kampylos src/kampylos_metal_fit.mm: a round cut into chunks, chunk k+1 on the GPU while the CPU completes chunk k (complete_batch(first, count))
added: 2026-11-04
---

## The defect

Wasted time, no wrong result. A GPU task whose host side does real work between launches (preparing the next batch,
finishing points the kernel left to the CPU, drawing random numbers, reducing results) leaves the GPU idle while the
CPU works and the CPU idle while the GPU works. The task takes the sum of both instead of the larger.

## Why it happens

The simple loop is "write inputs, launch, wait, read, process on the host, repeat": every step waits for the one before.
In kampylos the Metal app completed each round on the CPU only after the GPU had finished all of it; overlapping the two
gained 1.12-1.14x on a whole cell on an M1 and 1.03x on an M4 Pro, whose faster GPU left less to hide. The gain is at
most the smaller of the two shares, so it depends on the GPU, the CPU and the number of host threads.

## How to check a repository

```
git grep -nE 'clFinish|clWaitForEvents|CL_TRUE|cudaDeviceSynchronize|cudaMemcpy\(|cudaStreamSynchronize|waitUntilCompleted' -- src
```

No GPU app: `n/a`. For each launch loop, find what the host does between the wait and the next launch. Only buffer
reads and a few microseconds of bookkeeping: `not-affected`. Host work of a visible share of the loop (measure the two
times, or estimate them from the code and say so): `affected`, with the share in the issue.

## How to fix

Split the work into chunks and keep one chunk on the GPU while the host processes the previous one (double buffering,
two command queues or streams, events instead of blocking calls), or move the host work to a thread. Keep the order in
which results are combined, so the output stays byte-identical (the kampylos split keeps each candidate on the same
completion thread in the same order).

## How to test

Byte-identical output against the old build on the same device, the app's kill-and-resume test if the loop holds
checkpointed state, and an interleaved A/B on a whole work unit; report the GPU's and the host's share before and after.
