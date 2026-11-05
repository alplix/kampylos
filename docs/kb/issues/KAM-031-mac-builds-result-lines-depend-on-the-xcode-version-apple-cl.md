---
id: KAM-031
title: Mac builds' result lines depend on the Xcode version: Apple clang 21 fuses multiply-adds, Apple clang 14 did not
type: gotcha
status: confirmed
severity: low
platforms: [macos, clang]
area: determinism
action:
next: Add -ffp-contract=off to the macOS build lines if KAM-025's answer is that results of different builds are compared
blocked_by: []
pattern: PAT-018
aliases: []
reported: 2026-11-04
source: M4 benchmark 2026-11-04 (docs/BENCHMARK.md, Results on an Apple M4)
where: boinc/README.md:32-33, 47-50 @ faba573
fixed_in:
---

## Symptom

The same source, built with the documented lines on two Macs, writes result lines that differ in the last printed
digits. Measured on 2026-11-04 with the CPU app on the planet profile (one cell, 10 seeds): chi2 411.3206 from an Apple
M4 build with Apple clang 21, 411.3205 from the M1 builds with Apple clang 14. The 2026-10-16 report saw the same
between the shipped `kampylos_v12_macos_arm64` (built with macOS SDK 27) and a clang 14 build of the same source
(t0 ...101 against ...102, rho 1.155625e-06 against 1.155637e-06; docs/BENCHMARK.md "Measurements of 2026-10-16" §2.2).

## Cause

No build line sets `-ffp-contract` (boinc/README.md:32-33 and 47-50, scripts/bench/build_variant.sh:58-65 at faba573),
so each compiler's default decides whether `a * b + c` becomes one fused multiply-add. Apple clang 21.0.0
(clang-2100.3.34.2, Command Line Tools for macOS 27) defaults to `-ffp-contract=on` and emits 2433 `fmadd`-family
instructions in `VBMicrolensingLibrary.o` at `-O2` (fast CPU build; 2330 in the block form); the 2026-10-27 count for
Apple clang 14 was none (PAT-018 ledger row). Verified by measurement, not only by counting: with `-ffp-contract=off`
the clang 21 build writes 411.3205, the clang 14 value, and the documented and fast builds agree with each other under
either setting. With contraction on, the result also depends on optimisation decisions: a PGO build of the Metal app
changed chi2 by -0.0008 on `planet-gpu`, and with `-ffp-contract=off` the PGO build wrote the same lines as the plain one.
Other platforms (2026-11-04): Linux arm64 GCC 12 and clang 14, Linux x86_64 GCC 12 and Windows MinGW-w64 GCC 16 each
write their own lines (docs/BENCHMARK.md, "Other platforms").

## Fix

Proposed, not applied: add `-ffp-contract=off` to every compile line of the Mac apps (and, for the same reason, of the
GCC builds for arm64), so a new Xcode or GCC cannot move the results. Measured cost on the M4: none (CPU app +3.1 % on
`planet`, -2.4 % on `ogle`, Metal app +2.2 % on `planet-gpu`; all within noise). It would itself change the lines of
any build whose compiler contracts today, once. Whether that matters depends on KAM-025 (are results of different
builds compared?), so it waits for that answer; boinc/README.md describes the switch.

## Test

`scripts/bench/run_bench.sh planet 3 default=<build> nofma=<build with OPT="-O2 -ffp-contract=off">`: on Apple clang 21
the two hashes differ and the nofma chi2 is 411.3205. Count the instructions with
`otool -tv vbm.o | grep -cE 'fn?m(add|sub)\s+d'`.

## Log

- 2026-11-04: created.
- 2026-11-04: measured on an Apple M4 Pro, macOS 27.0, Apple clang 21.0.0 (docs/BENCHMARK.md, Results on an Apple M4).
