---
id: PAT-032
title: Small allocations in a hot loop are slow with the macOS system allocator
status: active
severity: low
platforms: [macos]
area: performance
applies_to: code that allocates and frees small objects (new/delete, malloc/free, growing containers) inside a hot loop
check: 'new [A-Za-z_]|malloc\(|calloc\(|std::vector<|std::string' -- src vendor
origin: kampylos docs/BENCHMARK.md (VBMicrolensing, 2026-10-16 and 2026-11-04)
reference_fix: kampylos boinc/README.md "Faster CPU build": a static jemalloc (or mimalloc) on the link line, no source change
added: 2026-11-04
---

## The defect

Wasted time, no wrong result. The macOS system allocator is slow for many small, short-lived blocks: in kampylos,
whose magnification library creates and deletes small objects for every point, it took about a third of the CPU app's
time on an M1 (macOS 15), and linking jemalloc made the app 1.13-1.5x faster depending on the machine (M1: 2.0x together
with plain statics; M4 Pro, macOS 27: 1.13x alone). With glibc on Linux the same change gained about 10 %.

## Why it happens

The default allocator of macOS is tuned for general use and takes locks and zone lookups that a fast thread-caching
allocator (jemalloc, mimalloc) avoids. The effect grows with the share of time a program spends allocating.

## How to check a repository

The grep above finds allocation sites, but only a profile says whether they matter: sample a production-like run on
macOS (`sample <pid> 10`) and add up `malloc`, `free`, `szone_*`, `tiny_*`, `_malloc_zone_*` and `operator new/delete`.
Under 2 % of the samples: `not-affected` (name the profile). Above: `affected`. No allocation in the hot code (one
buffer per task): `not-affected` with the place. A repository with no macOS build: `n/a`.

## How to fix

Link a static allocator that replaces malloc and `operator new`: Homebrew's `libjemalloc.a` on the link line (kampylos),
or mimalloc (`#include <mimalloc-new-delete.h>` in one translation unit). No source change; the results stay
byte-identical because allocation does not change arithmetic. Reducing the allocations in the hot code is the deeper
fix when the code is the repository's own.

## How to test

Byte-identical output, and an interleaved A/B of the builds by the rule of the repository's docs/BENCHMARK.md, on
macOS and, if the allocator is also used there, on Linux and Windows.
