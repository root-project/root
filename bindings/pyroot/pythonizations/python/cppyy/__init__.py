# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

"""Compatibility alias: the cppyy API is provided by the cppjit package.

Importing this module makes `cppyy` and every `cppyy.<submodule>` resolve
to the corresponding cppjit module object, so existing `import cppyy` /
`cppyy.gbl` call sites keep working unchanged.
"""

import importlib
import importlib.util
import sys

import cppjit


class _AliasLoader:
    """Hands out the already-imported cppjit module under the cppyy name."""

    def __init__(self, module):
        self._module = module

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        pass


class _CppyyAliasFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "cppyy" and not fullname.startswith("cppyy."):
            return None
        module = importlib.import_module("cppjit" + fullname[len("cppyy") :])
        return importlib.util.spec_from_loader(fullname, _AliasLoader(module))


sys.meta_path.insert(0, _CppyyAliasFinder())

# Alias the package itself and the submodules cppjit has already registered
# (cppjit.gbl, cppjit.gbl.std, ...), then let the finder cover the rest.
sys.modules["cppyy"] = cppjit
for _name in [n for n in sys.modules if n.startswith("cppjit.")]:
    sys.modules["cppyy" + _name[len("cppjit") :]] = sys.modules[_name]
