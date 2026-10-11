#!/usr/bin/env python3
"""Make VBMicrolensing's function-local statics thread_local, in place.

VBMicrolensingLibrary.cpp keeps its scratch state in function-local `static` variables (242 of
them in v5.6, plus one in the header). A function-local static is shared by every call from every
thread regardless of which VBMicrolensing instance made the call, so BinaryMag2 & co. are not
reentrant even across separate instances -- the CPU patch pool (src/kampylos_cpu_patch_pool.h)
crashed with heap corruption on real hardware because of it. `thread_local static` gives each
thread its own copy and keeps the library's intentional cross-call caching within a thread.

The script refuses to run if the library does not look exactly like the version it was written
for (the statics count must match), so an upstream update is noticed rather than half-patched.
It also adds `#define VBM_THREAD_LOCAL_STATICS 1` to the header, which kampylos_cpu_patch_pool.h
uses to pick its default thread count (4 when patched, 1 otherwise).

Usage (from the repo root, after `git submodule update --init`):
    python3 vendor/patch_vbm_thread_local.py            # patch
    python3 vendor/patch_vbm_thread_local.py --check    # exit 0 if patched, 1 if not
"""
import re
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "VBMicrolensing", "VBMicrolensing", "lib")
CPP = os.path.join(LIB, "VBMicrolensingLibrary.cpp")
HDR = os.path.join(LIB, "VBMicrolensingLibrary.h")
MARK = "#define VBM_THREAD_LOCAL_STATICS 1"
EXPECTED_CPP = 242   # indented `static` declarations in v5.6: all of them function-local variables
EXPECTED_HDR = 1     # `static double dx1, dx2;` inside _point::operator-

# In the .cpp every indented `static` is a function-local variable (the file's only static
# function is at column 0); in the header the class members (ESPLtablefile & co.) must stay as
# they are, so only the one known function-local declaration is touched.
CPP_STATIC = re.compile(r"^([ \t]+)static([ \t])", re.M)
HDR_STATIC = re.compile(r"^([ \t]+)static([ \t]+double dx1, dx2;)", re.M)


def main():
    check_only = "--check" in sys.argv
    if not os.path.isfile(CPP) or not os.path.isfile(HDR):
        print("VBMicrolensing sources not found under %s (run: git submodule update --init)" % LIB)
        return 2
    hdr = open(HDR).read()
    cpp = open(CPP).read()
    patched = MARK in hdr and "thread_local static" in cpp
    if check_only:
        print("patched" if patched else "NOT patched")
        return 0 if patched else 1
    if patched:
        print("already patched")
        return 0
    n_cpp = len(CPP_STATIC.findall(cpp))
    n_hdr = len(HDR_STATIC.findall(hdr))
    if n_cpp != EXPECTED_CPP or n_hdr != EXPECTED_HDR:
        print("unexpected library version: found %d/%d local statics in .cpp/.h, expected %d/%d -- not patching"
              % (n_cpp, n_hdr, EXPECTED_CPP, EXPECTED_HDR))
        return 1
    cpp2 = CPP_STATIC.sub(r"\1thread_local static\2", cpp)
    hdr2 = HDR_STATIC.sub(r"\1thread_local static\2", hdr)
    hdr2 = hdr2.replace("#pragma once", "#pragma once\n" + MARK, 1) if "#pragma once" in hdr2 else MARK + "\n" + hdr2
    open(CPP, "w").write(cpp2)
    open(HDR, "w").write(hdr2)
    print("patched %d statics in %s and %d in %s" % (n_cpp, os.path.relpath(CPP), n_hdr, os.path.relpath(HDR)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
