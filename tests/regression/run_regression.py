#!/usr/bin/env python3
"""Kampylos end-to-end regression test (2026-10-08 grid-convention / anchor / diagnostics fix).

Runs work units through the same code path as the BOINC apps (the fit_event CLI, built for one or
more backends -- see src/fit_event.cpp), assembles the results exactly like the server's
kampylos_assimilate.sh does for a format-2 result, scores them with the server's own
kampylos_analyze.py --file, and checks the outcome against expected_results.json.

Suites:
  synthetic  tests/regression/synthetic_planet.dat: a planet injected at log10 s = 0.2,
             log10 q = -3 (tests/gen_synthetic_planet.cpp). The WU "flux 0.2 0.2 1 -5 -1 3 log10"
             must put its best cell at log10 q = -3 with a large delta-chi2 and no flags, and the
             SAME numbers without the marker (natural logs: s = 1.22, q = 0.05) must not fit.
             If several backends are given, every cell must agree across them.
  real       the archival test events (--real-data DIR, see expected_results.json): selected grid
             cells around the published planets plus the two vetted controls (MOA-2002-BLG-014,
             a dwarf nova -> poor_fit_any_model; MOA-2003-BLG-004, three bad points ->
             few_points_signal, not a candidate). Slow: tens of GPU-minutes per cell.

Usage (from vendor/kampylos):
  python tests/regression/run_regression.py --cli cpu=PATH [--cli cuda=PATH] [--cli opencl=PATH]
      [--suite synthetic|real|all] [--real-data DIR] [--analyze ../../server/kampylos_analyze.py]
      [--workdir DIR] [--jobs N] [--reuse]
--jobs runs that many CLI processes at once (keep it <= 4 on a shared machine; each GPU process
also uses 4 CPU threads for the finite-source patching). --reuse skips WUs whose output exists.
Exit code 0 = every check passed.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def run_wu(cli, phot, in_line, out_path, reuse):
    if reuse and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path) as f:
            if sum(1 for l in f if l.strip() and not l.startswith("#")) > 0:
                return 0
    in_path = out_path + ".in"
    with open(in_path, "w") as f:
        f.write(in_line + "\n")
    with open(out_path + ".log", "w") as log:
        return subprocess.run([cli, phot, in_path, out_path], stderr=log, stdout=log).returncode


def assemble(outputs, dest):
    """Same as kampylos_assimilate.sh for format-2 results: header once, anchors tagged."""
    header = None
    body = []
    for i, p in enumerate(outputs):
        with open(p) as f:
            lines = [l.rstrip("\r\n") for l in f]
        if not lines:
            continue
        if header is None:
            header = lines[0]
        elif lines[0] != header:
            raise RuntimeError(f"{p}: header differs from the first result's -- would not be merged")
        for l in lines[1:]:
            if l.startswith("# anchor "):
                body.append(l.replace("# anchor ", f"# anchor result={i} ", 1))
            elif l.strip() and not l.startswith("#"):
                body.append(l)
    with open(dest, "w") as f:
        f.write(header + "\n" + "\n".join(body) + "\n")


def analyze(analyze_py, dat):
    out = subprocess.run([sys.executable, analyze_py, "--file", dat], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def cells_of(path):
    cells = {}
    with open(path) as f:
        for l in f:
            if l.startswith("#") or not l.strip():
                continue
            v = [float(x) for x in l.split()]
            cells[(round(v[0], 4), round(v[1], 4))] = v
    return cells


class Checker:
    def __init__(self):
        self.fails = 0

    def check(self, ok, msg):
        print(("  ok    " if ok else "  FAIL  ") + msg)
        if not ok:
            self.fails += 1


def compare_backends(chk, outs_by_backend, tol):
    names = list(outs_by_backend)
    ref = cells_of(outs_by_backend[names[0]])
    for other in names[1:]:
        cmp = cells_of(outs_by_backend[other])
        for k, v in sorted(ref.items()):
            if k not in cmp:
                chk.check(False, f"{other}: cell {k} missing")
                continue
            w = cmp[k]
            dchi2 = abs(v[2] - w[2])
            ok = dchi2 <= max(tol["chi2_abs"], tol["chi2_rel"] * v[2])
            chk.check(ok, f"{names[0]} vs {other} cell {k}: chi2 {v[2]:.3f} vs {w[2]:.3f} (|diff| {dchi2:.3f})")


def run_suite(name, cases, args, clis, chk):
    print(f"== suite {name}")
    tol = cases["tolerances"]
    for case in cases["cases"]:
        print(f"-- {case['name']}: {case.get('what', '')}")
        phot = case["photometry"]
        phot = os.path.join(HERE, phot) if name == "synthetic" else os.path.join(args.real_data, phot)
        outs_by_backend = {}
        for bname, cli in clis.items():
            jobs = []
            for i, line in enumerate(case["wus"]):
                out = os.path.join(args.workdir, f"{case['name']}.{bname}.wu{i}.out")
                jobs.append((cli, phot, line, out))
            with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                rcs = list(ex.map(lambda j: run_wu(*j, args.reuse), jobs))
            chk.check(all(rc == 0 for rc in rcs), f"{bname}: all {len(jobs)} WU(s) ran (exit codes {rcs})")
            dat = os.path.join(args.workdir, f"{case['name']}.{bname}.dat")
            assemble([j[3] for j in jobs], dat)
            outs_by_backend[bname] = dat
            res = analyze(args.analyze, dat)
            exp = case["expect"]
            with open(dat) as f:
                head = f.readline().split()[1]
            chk.check(head == "grid=" + exp["grid"], f"{bname}: results header says {head}, expected grid={exp['grid']}")
            if res is None:
                chk.check(False, f"{bname}: analyze returned nothing")
                continue
            summary = (f"best ({res['best_log_s']:.3f}, {res['best_log_q']:.3f}) chi2={res['best_chi2']:.2f} "
                       f"single={res['single_chi2']:.2f} [{res['single_model']}] pspl={res['pspl_chi2']:.2f} "
                       f"dchi2={res['delta_chi2']:.2f} robust={res['dchi2_robust']:.2f} n_pts50={res['n_pts50']} "
                       f"n_nights50={res['n_nights50']} poor_fit={res['flag_poor_fit']} few_points={res['flag_few_points']}")
            print(f"  [{bname}] {summary}")
            case.setdefault("_results", {})[bname] = res
            if "best_log_q" in exp:
                chk.check(abs(res["best_log_q"] - exp["best_log_q"]) < 1e-3, f"{bname}: best grid q {res['best_log_q']:.3f} == {exp['best_log_q']}")
            if "best_log_s" in exp:
                chk.check(abs(res["best_log_s"] - exp["best_log_s"]) < 1e-3, f"{bname}: best grid s {res['best_log_s']:.3f} == {exp['best_log_s']}")
            for key, lo in exp.get("min", {}).items():
                chk.check(res[key] is not None and res[key] >= lo, f"{bname}: {key} = {res[key]} >= {lo}")
            for key, hi in exp.get("max", {}).items():
                chk.check(res[key] is not None and res[key] <= hi, f"{bname}: {key} = {res[key]} <= {hi}")
            for key, val in exp.get("equal", {}).items():
                chk.check(res[key] == val, f"{bname}: {key} = {res[key]} == {val}")
            for key, (ref, rtol) in exp.get("approx", {}).items():
                chk.check(abs(res[key] - ref) <= rtol * max(1.0, abs(ref)),
                          f"{bname}: {key} = {res[key]:.3f} ~ {ref} (rel tol {rtol})")
            if "cell_chi2_worse_than" in exp:
                other, margin = exp["cell_chi2_worse_than"]
                ref_case = next(c for c in cases["cases"] if c["name"] == other)
                ref_res = ref_case.get("_results", {}).get(bname)
                if ref_res is not None:
                    chk.check(res["best_chi2"] > ref_res["best_chi2"] + margin,
                              f"{bname}: chi2 {res['best_chi2']:.2f} > {other}'s {ref_res['best_chi2']:.2f} + {margin}")
        if len(outs_by_backend) > 1:
            compare_backends(chk, outs_by_backend, tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", action="append", required=True, help="backend=path/to/fit_event")
    ap.add_argument("--suite", default="synthetic", choices=["synthetic", "real", "all"])
    ap.add_argument("--real-data", default=None)
    ap.add_argument("--analyze", default=os.path.join(REPO, "..", "..", "server", "kampylos_analyze.py"))
    ap.add_argument("--workdir", default=os.path.join(HERE, "work"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated case names")
    args = ap.parse_args()
    clis = dict(c.split("=", 1) for c in args.cli)
    os.makedirs(args.workdir, exist_ok=True)
    with open(os.path.join(HERE, "expected_results.json")) as f:
        expected = json.load(f)
    chk = Checker()
    suites = ["synthetic", "real"] if args.suite == "all" else [args.suite]
    for s in suites:
        cases = expected[s]
        if args.only:
            keep = set(args.only.split(","))
            cases = dict(cases, cases=[c for c in cases["cases"] if c["name"] in keep])
        if s == "real" and not args.real_data:
            print("== suite real skipped (no --real-data)")
            continue
        run_suite(s, cases, args, clis, chk)
    print("PASS" if chk.fails == 0 else f"FAILED ({chk.fails} check(s))")
    sys.exit(1 if chk.fails else 0)


if __name__ == "__main__":
    main()
