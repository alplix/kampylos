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

--block (2026-11-04) is for the GPU apps, which need the thread_local statics for their
completion threads: it keeps them thread_local but gathers each function's statics into one
thread_local struct, declared at the top of the function, and turns every original declaration
into references to its members at the same place (`auto& Mag = _vbm_tls.Mag;`). A call then pays
one TLS lookup instead of one per access of each variable. Members without an initializer get
`{}` (the zero a static starts with) and the header's `complex` constructors become constexpr, so
a struct of plain members needs no initialization guard; the two statics of `_point::operator-`,
written before every read, become locals. Same variables, same per-thread lifetime and initial
values, so the results are bit-identical. Apple M1, scripts/bench profile planet-gpu: the Metal
app 1.30x faster, 1.69x together with jemalloc (docs/BENCHMARK.md). It works on the fork (or a
patched upstream) and keeps the VBM_THREAD_SAFE marker; restore the plain fork with
`git -C vendor/VBMicrolensing checkout -- .`.

Usage (from the repo root, after `git submodule update --init`):
    python3 vendor/patch_vbm_thread_local.py            # patch
    python3 vendor/patch_vbm_thread_local.py --strip    # undo (CPU app build)
    python3 vendor/patch_vbm_thread_local.py --block    # one thread_local block per function (GPU app build)
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


def blank_code(text):
    """text with comments, string and character literals replaced by spaces (newlines kept)."""
    out, i, n, quote = [], 0, len(text), None
    while i < n:
        c = text[i]
        if quote:
            if c == "\\":
                out.append("  ")
                i += 2
                continue
            if c == quote:
                quote = None
            out.append("\n" if c == "\n" else " ")
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", text[i:j]))
            i = j
        else:
            if c in "\"'":
                quote = c
                c = " "
            out.append(c)
            i += 1
    return "".join(out)


def split_top_level(text):
    """text split at the commas that are outside (), [], {} and <>."""
    parts, depth, cur = [], 0, ""
    for c in text:
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        if c == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    parts.append(cur)
    return parts


def to_blocks(cpp):
    """The --block rewrite (see the docstring). Returns (new text, functions, variables)."""
    lines = cpp.split("\n")
    code = blank_code(cpp).split("\n")
    # the function each declaration belongs to = the line of the '{' that opened brace depth 1
    depth, opener, owner = 0, None, {}
    for li, cl in enumerate(code):
        if "thread_local" in cl and depth >= 1:
            owner[li] = opener
        for c in cl:
            if c == "{":
                depth += 1
                if depth == 1:
                    opener = li
            elif c == "}":
                depth -= 1
    members, replaced, n_vars = {}, {}, 0
    # a local `const int N = 10;` used as an array bound must be spelled out: the struct comes before it
    consts = dict(re.findall(r"^[ \t]*const int (\w+) = (\d+);", cpp, re.M))
    li = 0
    while li < len(lines):
        if li not in owner:
            li += 1
            continue
        m = re.match(r"^([ \t]*)thread_local static (.*)$", lines[li], re.S)
        if not m:
            raise ValueError("unexpected declaration at line %d: %s" % (li + 1, lines[li]))
        indent, text, last = m.group(1), m.group(2), li
        while ";" not in blank_code(text):          # declarations with a brace list may span lines
            last += 1
            text += "\n" + lines[last]
        cut = blank_code(text).index(";")
        decl, rest = text[:cut], text[cut + 1:]
        t = re.match(r"\s*((?:const\s+)?(?:std::)?[A-Za-z_]\w*)\s*(.*)$", decl, re.S)
        typ, refs = t.group(1), []
        for d in split_top_level(t.group(2)):
            v = re.match(r"\s*(\**)\s*([A-Za-z_]\w*)\s*(.*?)\s*$", d, re.S)
            stars, name, init = v.group(1), v.group(2), " ".join(v.group(3).split())
            init = re.sub(r"\[(\w+)\]", lambda b: "[%s]" % consts.get(b.group(1), b.group(1)), init)
            if "=" not in init and "{" not in init:
                init += "{}"     # the zero a static starts with; lets the struct be constant-initialized
            members.setdefault(owner[li], []).append("%s%s %s%s%s;" % (typ, stars, name, "" if init[:1] in ("[", "{") else " ", init))
            refs.append("auto& %s = _vbm_tls.%s;" % (name, name))
            n_vars += 1
        replaced[li] = (last, indent + " ".join(refs) + rest)
        li = last + 1
    out, li = [], 0
    while li < len(lines):
        if li in replaced:
            last, text = replaced[li]
            out.append(text)
            li = last + 1
            continue
        out.append(lines[li])
        if li in members:
            if not code[li].rstrip().endswith("{"):
                raise ValueError("function body does not open at the end of line %d" % (li + 1))
            cr = "\r" if lines[li].endswith("\r") else ""
            out.append("\tstruct _vbm_tls_t { %s }; thread_local static _vbm_tls_t _vbm_tls_; _vbm_tls_t& _vbm_tls = _vbm_tls_;%s" % (" ".join(members[li]), cr))
        li += 1
    return "\n".join(out), len(members), n_vars


# --block also edits the header: constexpr constructors for the library's complex class (same
# values), so that a block of complex/double/int members is constant-initialized and needs no
# init guard on every call; and _point::operator-'s two statics, which are written before they are
# read on every call, become plain locals.
HDR_BLOCK = [
    ("complex(double a, double b) { re = a; im = b; }", "constexpr complex(double a, double b) : re(a), im(b) {}"),
    ("complex(double a) { re = a; im = 0; }", "constexpr complex(double a) : re(a), im(0) {}"),
    ("complex(void) { re = 0; im = 0; }", "constexpr complex(void) : re(0), im(0) {}"),
    ("thread_local static double dx1, dx2;", "double dx1, dx2;"),
]


def main():
    check_only = "--check" in sys.argv
    strip = "--strip" in sys.argv
    block = "--block" in sys.argv
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
    if "_vbm_tls_t" in cpp:
        if strip:
            print("the library is in --block form: restore the fork first (git -C vendor/VBMicrolensing checkout -- .)")
            return 1
        print("already patched (one thread_local block per function)")
        return 0
    if block:
        n_cpp = len(CPP_TLS.findall(cpp))
        if not patched or n_cpp != EXPECTED_CPP:
            print("--block needs the thread_local library (the fork, or run this script without options first): "
                  "found %d thread_local statics in %s, expected %d" % (n_cpp, os.path.relpath(CPP), EXPECTED_CPP))
            return 1
        if any(hdr.count(a) != 1 for a, _ in HDR_BLOCK):
            print("unexpected library version: the complex constructors or _point::operator- in %s differ from v5.6 -- not changing"
                  % os.path.relpath(HDR))
            return 1
        cpp2, n_funcs, n_vars = to_blocks(cpp)
        for a, b in HDR_BLOCK:
            hdr = hdr.replace(a, b)
        open(CPP, "w").write(cpp2)
        open(HDR, "w").write(hdr)
        print("gathered the %d thread_local declarations (%d variables) of %s into one block per function (%d functions)"
              % (n_cpp, n_vars, os.path.relpath(CPP), n_funcs))
        return 0
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
