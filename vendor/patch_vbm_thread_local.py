#!/usr/bin/env python3
"""Make VBMicrolensing's function-local statics thread_local, in place -- or undo it.

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

--strip is the reverse, for the CPU app (boinc/main_kampylos_boinc.cpp): it is single-threaded
and does not need the patch, but pays for it on every access to one of the 577 thread_local
variables -- on macOS each goes through a call to _tlv_get_addr (46 % of the CPU app's samples
with Apple clang 14), with emulated TLS (MinGW-w64, Android below API 29) more. Plain statics
make the CPU app about 1.35x faster with a byte-identical result file (see boinc/README.md).
--strip also removes the VBM_THREAD_SAFE / VBM_THREAD_LOCAL_STATICS marker, so a GPU app built
from the same checkout by mistake runs its completion step single-threaded instead of
corrupting the heap. The result is plain upstream v5.6 again; restore the fork's files with
`git -C vendor/VBMicrolensing checkout -- .` before building a GPU app.

Usage (from the repo root, after `git submodule update --init`):
    python3 vendor/patch_vbm_thread_local.py            # patch
    python3 vendor/patch_vbm_thread_local.py --strip    # undo (CPU app build)
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
CPP_TLS = re.compile(r"^([ \t]+)thread_local static([ \t])", re.M)
HDR_TLS = re.compile(r"^([ \t]+)thread_local static([ \t]+double dx1, dx2;)", re.M)
# The markers to drop with --strip: ours, and the alplix/VBMicrolensing fork's define together
# with the comment lines it carries ("// bitboinc patch: ..."). Line endings may be CRLF.
MARKER_LINES = re.compile(r"^(?://[^\n]*bitboinc patch[^\n]*\r?\n(?://[^\n]*\r?\n)*)?#define VBM_THREAD_(?:SAFE|LOCAL_STATICS) 1[ \t]*\r?\n", re.M)


def main():
    check_only = "--check" in sys.argv
    strip = "--strip" in sys.argv
    if not os.path.isfile(CPP) or not os.path.isfile(HDR):
        print("VBMicrolensing sources not found under %s (run: git submodule update --init)" % LIB)
        return 2
    hdr = open(HDR).read()
    cpp = open(CPP).read()
    # the alplix/VBMicrolensing fork (the submodule) defines VBM_THREAD_SAFE instead of our marker
    patched = (MARK in hdr or "VBM_THREAD_SAFE" in hdr) and "thread_local static" in cpp
    if check_only:
        print("patched" if patched else "NOT patched")
        return 0 if patched else 1
    if strip:
        if not patched and "thread_local static" not in cpp:
            print("already plain (no thread_local statics)")
            return 0
        n_cpp = len(CPP_TLS.findall(cpp))
        n_hdr = len(HDR_TLS.findall(hdr))
        if n_cpp != EXPECTED_CPP or n_hdr != EXPECTED_HDR:
            print("unexpected library version: found %d/%d thread_local statics in .cpp/.h, expected %d/%d -- not stripping"
                  % (n_cpp, n_hdr, EXPECTED_CPP, EXPECTED_HDR))
            return 1
        cpp2 = CPP_TLS.sub(r"\1static\2", cpp)
        hdr2 = MARKER_LINES.sub("", HDR_TLS.sub(r"\1static\2", hdr))
        open(CPP, "w").write(cpp2)
        open(HDR, "w").write(hdr2)
        print("stripped %d thread_local statics in %s and %d in %s (marker removed)"
              % (n_cpp, os.path.relpath(CPP), n_hdr, os.path.relpath(HDR)))
        return 0
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
