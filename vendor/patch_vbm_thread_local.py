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

--reset (2026-11-05) can be added to any of the forms above (or given alone on the fork). It
adds three public methods, ResetRootWarmStart(), GetRootWarmStart() and SetRootWarmStart(),
which clear, read and write the warm start of the binary-lens root solver: the five roots of the
previous call (`zr` in NewImages), from which cmplx_roots_gen() starts the next solve. That array
is the reason a magnification depends in its last bits on the call before it, and through
Nelder-Mead a seed's fit on the seed before it (KAM-030, KAM-012). Cleared, it holds what a fresh
thread starts with (every root 0). The library's own code never reaches the new hook (it is a call
of NewImages without coefficients), so without calls of the new methods the results are
bit-identical; the header defines VBM_KAMPYLOS_RESET.

--pool (2026-11-05) can be added to any form as well. It gives the seven small classes the binary-lens
magnification allocates and frees for every point and contour sample (_point, _curve, _skiplist_curve,
_theta, _thetas, annulus, _sols_for_skiplist_curve) their own operator new and delete with a free list:
a freed object is kept for the next allocation of the same class instead of going back to malloc. The
lists are thread_local in the thread-safe forms (the fork, --block) and plain statics with --strip; they
never shrink (the peak is bounded by one magnification's contour). Allocation does not touch the
arithmetic, so the results are bit-identical; the header defines VBM_KAMPYLOS_POOL.

Usage (from the repo root, after `git submodule update --init`):
    python3 vendor/patch_vbm_thread_local.py            # patch
    python3 vendor/patch_vbm_thread_local.py --strip    # undo (CPU app build)
    python3 vendor/patch_vbm_thread_local.py --block    # one thread_local block per function (GPU app build)
    python3 vendor/patch_vbm_thread_local.py --strip --reset   # either form, plus the warm-start methods
    python3 vendor/patch_vbm_thread_local.py --block --pool    # either form, plus the free lists
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


RESET_MARK = "#define VBM_KAMPYLOS_RESET 1"
NEWIMAGES = re.compile(r"^_curve\* VBMicrolensing::NewImages\(complex yi, complex\* coefs, _theta\* theta\) \{\r?\n", re.M)
BINARYMAG2_DECL = re.compile(r"^\tdouble BinaryMag2\(double s, double q, double y1, double y2, double rho\);(\r?\n)", re.M)
CLASS_LINE = re.compile(r"^class VBMicrolensing(\r?\n)", re.M)
RESET_HOOK = """\t// kampylos (vendor/patch_vbm_thread_local.py --reset): a call without coefficients reads (yi = 1),
\t// writes (yi = 2) or clears (yi = 0) the root solver's warm start zr, so that the caller can make
\t// a magnification independent of the calls before it (kampylos KAM-030). The library never calls
\t// NewImages without coefficients, so its own results do not change.
\tif (!coefs) {
\t\tdouble* b = (double*)(void*)theta;
\t\tfor (int i = 0; i < 5; i++) {
\t\t\tif (yi.re == 1) { b[2 * i] = zr[i].re; b[2 * i + 1] = zr[i].im; }
\t\t\telse if (yi.re == 2) zr[i] = complex(b[2 * i], b[2 * i + 1]);
\t\t\telse zr[i] = complex(0., 0.);
\t\t}
\t\treturn 0;
\t}
"""
RESET_METHODS = """
// kampylos (vendor/patch_vbm_thread_local.py --reset): the warm start of the root solver, see the
// hook at the top of NewImages(complex, complex*, _theta*).
void VBMicrolensing::ResetRootWarmStart() { NewImages(complex(0., 0.), 0, 0); }
void VBMicrolensing::GetRootWarmStart(double* re_im) { NewImages(complex(1., 0.), 0, (_theta*)(void*)re_im); }
void VBMicrolensing::SetRootWarmStart(const double* re_im) { NewImages(complex(2., 0.), 0, (_theta*)(void*)const_cast<double*>(re_im)); }
"""
RESET_DECLS = ["\t// kampylos (vendor/patch_vbm_thread_local.py --reset): the root solver's warm start, 5 roots as",
               "\t// 10 doubles (re, im); cleared = what a fresh thread starts with",
               "\tvoid ResetRootWarmStart();",
               "\tvoid GetRootWarmStart(double* re_im);",
               "\tvoid SetRootWarmStart(const double* re_im);"]


def add_reset(cpp, hdr):
    """The --reset additions (see the docstring). Returns (cpp, hdr), or None if the library is not the expected one."""
    if RESET_MARK in hdr:
        return cpp, hdr
    m = NEWIMAGES.search(cpp)
    d = BINARYMAG2_DECL.search(hdr)
    c = CLASS_LINE.search(hdr)
    if not m or not d or not c or len(NEWIMAGES.findall(cpp)) != 1:
        print("--reset: NewImages(complex, complex*, _theta*), the BinaryMag2 declaration or the class line is not as in v5.6")
        return None
    at = cpp.find("\ty = yi + coefs[11];", m.end())
    if at < 0 or "\n}" in cpp[m.end():at]:
        print("--reset: the first statement of NewImages is not where v5.6 has it")
        return None
    cpp = cpp[:at] + RESET_HOOK + cpp[at:]
    cpp = cpp.rstrip("\r\n") + "\n" + RESET_METHODS
    eol = d.group(1)
    hdr = hdr[:d.end()] + eol.join(RESET_DECLS) + eol + hdr[d.end():]
    c = CLASS_LINE.search(hdr)
    hdr = hdr[:c.start()] + RESET_MARK + c.group(1) + hdr[c.start():]
    return cpp, hdr


POOL_MARK = "#define VBM_KAMPYLOS_POOL 1"
POOL_CLASSES = ["struct annulus {", "class _theta {", "class _thetas {", "class _point {", "class _curve {",
                "class _skiplist_curve {", "class _sols_for_skiplist_curve {"]
POOL_CODE = [
    "// kampylos (vendor/patch_vbm_thread_local.py --pool): a free list per small class, see the script's docstring.",
    "#include <cstddef>",
    "#include <new>",
    "struct _vbm_free_node { _vbm_free_node* next; };",
    "template <class T> struct _vbm_pool {",
    "\tstatic _vbm_free_node*& head() { %s _vbm_free_node* h = nullptr; return h; }",
    "\tstatic void* get(std::size_t n) {",
    "\t\tif (n == sizeof(T)) { _vbm_free_node*& h = head(); if (h) { _vbm_free_node* p = h; h = p->next; return p; } }",
    "\t\treturn ::operator new(n);",
    "\t}",
    "\tstatic void put(void* p, std::size_t n) {",
    "\t\tif (!p) return;",
    "\t\tif (n == sizeof(T)) { _vbm_free_node*& h = head(); _vbm_free_node* q = (_vbm_free_node*)p; q->next = h; h = q; return; }",
    "\t\t::operator delete(p);",
    "\t}",
    "};",
    "#define VBM_POOLED(T) static void* operator new(std::size_t n) { return _vbm_pool<T>::get(n); } \\",
    "\tstatic void operator delete(void* p, std::size_t n) { _vbm_pool<T>::put(p, n); }",
]


def add_pool(hdr):
    """The --pool additions (header only). Returns the new header, or None if it is not the expected one."""
    if POOL_MARK in hdr:
        return hdr
    m = re.search(r"^struct annulus \{(\r?\n)", hdr, re.M)
    if not m or any(hdr.count(c) != 1 for c in POOL_CLASSES):
        print("--pool: the small classes of the header are not as in v5.6")
        return None
    eol = m.group(1)
    tls = "thread_local static" if ("VBM_THREAD_SAFE" in hdr or MARK in hdr) else "static"
    code = eol.join(POOL_CODE).replace("%s", tls) + eol
    for c in POOL_CLASSES:
        at = hdr.index(c)
        line_end = hdr.index("\n", at) + 1
        name = c.split()[1]
        insert = "\tVBM_POOLED(%s)%s" % (name, eol)
        if c.startswith("class"):     # after the `public:` line that follows
            pub = hdr.index("public:", line_end)
            line_end = hdr.index("\n", pub) + 1
        hdr = hdr[:line_end] + insert + hdr[line_end:]
    at = hdr.index("struct annulus {")
    hdr = hdr[:at] + POOL_MARK + eol + code + eol + hdr[at:]
    return hdr


def finish(cpp, hdr, orig_cpp, orig_hdr):
    """Applies the options that add to any form (--reset, --pool) and writes what changed."""
    if "--pool" in sys.argv:
        h2 = add_pool(hdr)
        if h2 is None:
            return 1
        if h2 != hdr:
            print("--pool: free lists for the %d small classes" % len(POOL_CLASSES))
        hdr = h2
    if "--reset" in sys.argv:
        r = add_reset(cpp, hdr)
        if r is None:
            return 1
        if r[1] != hdr:
            print("--reset: added ResetRootWarmStart(), GetRootWarmStart() and SetRootWarmStart()")
        cpp, hdr = r
    if cpp != orig_cpp:
        open(CPP, "w", newline="").write(cpp)
    if hdr != orig_hdr:
        open(HDR, "w", newline="").write(hdr)
    return 0


def main():
    check_only = "--check" in sys.argv
    strip = "--strip" in sys.argv
    block = "--block" in sys.argv
    if not os.path.isfile(CPP) or not os.path.isfile(HDR):
        print("VBMicrolensing sources not found under %s (run: git submodule update --init)" % LIB)
        return 2
    hdr = open(HDR, newline="").read()
    cpp = open(CPP, newline="").read()
    orig_cpp, orig_hdr = cpp, hdr
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
        return finish(cpp, hdr, orig_cpp, orig_hdr)
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
        print("gathered the %d thread_local declarations (%d variables) of %s into one block per function (%d functions)"
              % (n_cpp, n_vars, os.path.relpath(CPP), n_funcs))
        return finish(cpp2, hdr, orig_cpp, orig_hdr)
    if strip:
        if not patched and "thread_local static" not in cpp:
            print("already plain (no thread_local statics)")
            return finish(cpp, hdr, orig_cpp, orig_hdr)
        n_cpp = len(CPP_TLS.findall(cpp))
        n_hdr = len(HDR_TLS.findall(hdr))
        if n_cpp != EXPECTED_CPP or n_hdr != EXPECTED_HDR:
            print("unexpected library version: found %d/%d thread_local statics in .cpp/.h, expected %d/%d -- not stripping"
                  % (n_cpp, n_hdr, EXPECTED_CPP, EXPECTED_HDR))
            return 1
        cpp2 = CPP_TLS.sub(r"\1static\2", cpp)
        hdr2 = MARKER_LINES.sub("", HDR_TLS.sub(r"\1static\2", hdr))
        print("stripped %d thread_local statics in %s and %d in %s (marker removed)"
              % (n_cpp, os.path.relpath(CPP), n_hdr, os.path.relpath(HDR)))
        return finish(cpp2, hdr2, orig_cpp, orig_hdr)
    if patched:
        print("already patched")
        return finish(cpp, hdr, orig_cpp, orig_hdr)
    n_cpp = len(CPP_STATIC.findall(cpp))
    n_hdr = len(HDR_STATIC.findall(hdr))
    if n_cpp != EXPECTED_CPP or n_hdr != EXPECTED_HDR:
        print("unexpected library version: found %d/%d local statics in .cpp/.h, expected %d/%d -- not patching"
              % (n_cpp, n_hdr, EXPECTED_CPP, EXPECTED_HDR))
        return 1
    cpp2 = CPP_STATIC.sub(r"\1thread_local static\2", cpp)
    hdr2 = HDR_STATIC.sub(r"\1thread_local static\2", hdr)
    hdr2 = hdr2.replace("#pragma once", "#pragma once\n" + MARK, 1) if "#pragma once" in hdr2 else MARK + "\n" + hdr2
    print("patched %d statics in %s and %d in %s" % (n_cpp, os.path.relpath(CPP), n_hdr, os.path.relpath(HDR)))
    return finish(cpp2, hdr2, orig_cpp, orig_hdr)


if __name__ == "__main__":
    sys.exit(main())
