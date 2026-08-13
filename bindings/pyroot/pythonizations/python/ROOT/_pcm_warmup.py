# Author: Silia Taider, CERN 08/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

"""
One-time warm-up that builds every ROOT C++ module ahead of use.
What we do: explicitly import every module declared in ROOT.modulemap, 
once, right after the interpreter is up.
This only runs once per installation and is a no-op for builds that don't use C++
modules at all.
"""

import os
import re
import sys

_MODULE_LINE = re.compile(r'^module\s+"?([A-Za-z_][A-Za-z0-9_]*)"?\s*\{')


def _module_names(modulemap_path):
    """Extract the top-level module names declared in a Clang modulemap file"""
    names = []
    with open(modulemap_path) as f:
        for line in f:
            m = _MODULE_LINE.match(line)
            if m:
                names.append(m.group(1))
    return names


def _sentinel_path(lib_dir):
    """Empty marker file to record that the module build run once"""
    return os.path.join(lib_dir, ".pcm_warmup_complete")


def _mark_warmup_complete(lib_dir, sentinel):
    try:
        os.makedirs(lib_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write("1")
    except OSError:
        pass


def _print_progress(done, total, label):
    width = 30
    filled = width if total == 0 else int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r[{bar}] {done}/{total}  building {label:<28}")
    sys.stderr.flush()


def warmup(root_facade):
    """Build every C++ module once"""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    modulemap_path = os.path.join(this_dir, "include", "ROOT.modulemap")
    if not os.path.exists(modulemap_path):
        # runtime_cxxmodules is off in this build
        return

    lib_dir = os.path.join(this_dir, "lib")
    sentinel = _sentinel_path(lib_dir)
    if os.path.exists(sentinel):
        return

    names = _module_names(modulemap_path)
    if not names:
        return

    sys.stderr.write(f"ROOT: preparing this installation for your machine, this may take some time: {len(names)} modules...\n")
    declare = root_facade.gInterpreter.Declare
    for i, name in enumerate(names, 1):
        _print_progress(i, len(names), name)
        try:
            declare(f"#pragma clang module import {name}")
        except Exception:
            pass
    sys.stderr.write("\n")

    _mark_warmup_complete(lib_dir, sentinel)
