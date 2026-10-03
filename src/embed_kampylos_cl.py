#!/usr/bin/env python3
"""Regenerates kampylos_cl_embedded.h from kampylos_gpu_types.h + kampylos_kernel.cl -- run this
after editing either (not auto-run at build time). Same convention as this project's own
embed_cl.py (src/embed_cl.py, for Keraunos/Sphinx's search.cl): escapes each file as a C string
literal so kampylos_opencl_search.c can embed the program source directly in the binary rather
than shipping separate files alongside it.

Two separate string constants, not one concatenated blob -- clCreateProgramWithSource takes
multiple source strings natively (compiled as if concatenated), so there's no need to paste the
two files together textually just to embed them; keeping them separate also means this script
doesn't need to know or care about include order beyond "types before kernel" at the call site.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def emit(out, varname, path):
    with open(path) as f:
        lines = f.readlines()
    out.write(f"static const char {varname}[] =\n")
    for line in lines:
        line = line.rstrip("\n")
        escaped = line.replace("\\", "\\\\").replace('"', '\\"')
        out.write('"' + escaped + '\\n"\n')
    out.write(";\n\n")
    return len(lines)


with open(os.path.join(HERE, "kampylos_cl_embedded.h"), "w") as out:
    out.write("// Auto-generated from kampylos_gpu_types.h + kampylos_kernel.cl by\n")
    out.write("// embed_kampylos_cl.py -- do not hand-edit.\n")
    n1 = emit(out, "kKampylosClTypesSource", os.path.join(HERE, "kampylos_gpu_types.h"))
    n2 = emit(out, "kKampylosClKernelSource", os.path.join(HERE, "kampylos_kernel.cl"))

print(f"wrote kampylos_cl_embedded.h, {n1} + {n2} lines embedded")
