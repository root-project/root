# Author: Enric Tejedor, Danilo Piparo CERN  06/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from os import environ

# Prevent cppyy's check for the PCH
environ["CLING_STANDARD_PCH"] = "none"

# Prevent cppyy's check for extra header directory
environ["CPPYY_API_PATH"] = "none"

# Prevent cppyy from filtering ROOT libraries
environ["CPPYY_NO_ROOT_FILTER"] = "1"

# Do setup specific to AddressSanitizer environments
from . import _asan

import cppyy
import sys, importlib
import libROOTPythonizations

# Build cache of commonly used python strings (the cache is python intern, so
# all strings are shared python-wide, not just in PyROOT).
# See: https://docs.python.org/3.2/library/sys.html?highlight=sys.intern#sys.intern
_cached_strings = []
for s in ["Branch", "FitFCN", "ROOT", "SetBranchAddress", "SetFCN", "_TClass__DynamicCast", "__class__"]:
    _cached_strings.append(sys.intern(s))

# Trigger the addition of the pythonizations
from ._pythonization import _register_pythonizations

_register_pythonizations()

# Check if we are in the IPython shell
import builtins

_is_ipython = hasattr(builtins, "__IPYTHON__")


class _PoisonedDunderAll:
    """
    Dummy class used to trigger an ImportError on wildcard imports if the
    `__all__` attribute of a module is an instance of this class.
    """

    def __getitem__(self, _):
        import textwrap

        message = """
        Wildcard import e.g. `from module import *` is bad practice, so it is disallowed in ROOT. Please import explicitly.
        """
        raise ImportError(textwrap.dedent(message))


# Prevent `from ROOT import *` by setting the __all__ attribute to something
# that will raise an ImportError on item retrieval.
__all__ = _PoisonedDunderAll()

# Configure ROOT facade module
import sys
from ._facade import ROOTFacade

_root_facade = ROOTFacade(sys.modules[__name__], _is_ipython)
sys.modules[__name__] = _root_facade

# Configure meta-path finder for ROOT namespaces, following the Python
# documentation and an example:
#
#   * https://docs.python.org/3/library/importlib.html#module-importlib.abc
#
#   * https://python.plainenglish.io/metapathfinders-or-how-to-change-python-import-behavior-a1cf3b5a13ec
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader


def _can_be_module(obj) -> bool:
    """
    Determine if an object can be used as a Python module. This is the case for
    objects that are actually of ModuleType, or C++ namespaces from cppyy.
    """

    # If the type is the module type, it can trivially be a module.
    if isinstance(obj, types.ModuleType):
        return True

    # Check if the object represents a C++ namespace. Since cppyy has no
    # dedicated Python type for C++ namespaces, we check for this using the
    # representation of the object.
    if repr(obj).startswith("<namespace "):
        return True

    return False


from typing import Optional, Union
import types


def _lookup_root_module(fullname: str) -> Optional[Union[types.ModuleType, cppyy._backend.CPPScope]]:
    """
    Recursively looks up attributes of the ROOT facade, using a full module
    name, and return it if it can be used as a ROOT submodule. This is the case
    if the attribute is a C++ namespace or an actual Python module type. If no
    matching attribute is found, return None.
    """
    keys = fullname.split(".")[1:]
    ret = _root_facade
    for part in keys:
        ret = getattr(ret, part, None)
        if ret is None or not _can_be_module(ret):
            return None
    return ret


class _RootNamespaceLoader(Loader):
    """
    Custom loader for modules under the ROOT namespace.
    """

    def is_package(self, fullname: str) -> bool:
        """
        Indicates whether the given attribute of the ROOT facade can be
        considered a package.

        This is decided by the _lookup_root_module function.
        """
        return _lookup_root_module(fullname) is not None

    def create_module(self, spec: ModuleSpec):
        out = _lookup_root_module(spec.name)
        # Prevent wildcard import for the submodule by setting the __all__
        # attribute to something that will raise an ImportError on item
        # retrieval.
        out.__all__ = _PoisonedDunderAll()
        return out

    def exec_module(self, module):
        pass


class _RootNamespaceFinder(MetaPathFinder):
    """
    Finder for modules under the ROOT namespace.
    """

    def find_spec(self, fullname: str, path, target=None) -> ModuleSpec:
        if not fullname.startswith("ROOT."):
            # This finder only finds ROOT.*
            return None
        if _lookup_root_module(fullname) is None:
            return None
        return spec_from_loader(fullname, _RootNamespaceLoader())


namespace_finder = _RootNamespaceFinder()
if namespace_finder not in sys.meta_path:
    sys.meta_path.append(namespace_finder)

# Configuration for usage from Jupyter notebooks
if _is_ipython:
    from IPython import get_ipython

    ip = get_ipython()
    if hasattr(ip, "kernel"):
        import JupyROOT
        from . import JsMVA

# Register cleanup
import atexit


def cleanup():
    # If spawned, stop thread which processes ROOT events
    facade = sys.modules[__name__]
    if "app" in facade.__dict__ and hasattr(facade.__dict__["app"], "process_root_events"):
        facade.__dict__["app"].keep_polling = False
        facade.__dict__["app"].process_root_events.join()

atexit.register(cleanup)
