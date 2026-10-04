#!/usr/bin/env python3
"""Make Sunbird's VITS `monotonic_align` package importable on Python 3.13.

The compiled Cython extension imports fine on Python 3.11 through
`from .monotonic_align.core import maximum_path_c`, but that relative import
fails on 3.13 with ModuleNotFoundError. This patch loads the compiled .so by
file path instead (found by pattern, so the CPU/Python tag in its name does not
matter). Safe to run more than once.

usage: patch_monotonic_align.py <path/to/vits-work/training/monotonic_align/__init__.py>
"""
import sys

OLD = "from .monotonic_align.core import maximum_path_c"
NEW = '''import glob as _glob
import importlib.util as _ilu
import os as _os
# Load the compiled extension by file path: the relative package import fails on
# Python 3.13 (it worked on 3.11), so find the .so next to this file instead.
_so = _glob.glob(_os.path.join(_os.path.dirname(__file__), "monotonic_align", "core*.so"))[0]
_spec = _ilu.spec_from_file_location("core", _so)
_core = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_core)
maximum_path_c = _core.maximum_path_c'''

path = sys.argv[1]
src = open(path).read()
if "_ilu.spec_from_file_location" in src:
    print("already patched:", path)
elif OLD not in src:
    sys.exit(f"unexpected file contents, not patching: {path}")
else:
    open(path + ".orig", "w").write(src)
    open(path, "w").write(src.replace(OLD, NEW))
    print("patched:", path, "(original saved as .orig)")
