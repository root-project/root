""" PyPy-specific touch-ups
"""

from . import _stdcpp_fix

import os
import sys
from cppyy_backend import loader

__all__ = [
    'gbl',
    'addressof',
    'bind_object',
    'nullptr',
    ]

# first load the dependency libraries of the backend, then
# pull in the built-in low-level cppyy
try:
    c = loader.load_cpp_backend()
except RuntimeError:
    import sysconfig
    os.environ['CPPYY_BACKEND_LIBRARY'] = "libcppyy_backend"+sysconfig.get_config_var("SO")
    c = loader.load_cpp_backend()
os.environ['CPPYY_BACKEND_LIBRARY'] = c._name

# some older versions can be fixed up through a compatibility
# module on the python side; load it, if available
try:
    import cppyy_compat
except ImportError:
    pass

import _cppyy as _backend     # built-in module
try:
    _backend._post_import_startup()
except AttributeError:
    pass
_backend._cpp_backend = c

def fixup_legacy():
    version = sys.pypy_version_info
    if version[0] < 7 or (version[0] == 7 and version[1] <= 3 and version[2] <= 3):
        _backend.gbl.CppyyLegacy = _backend.gbl
fixup_legacy()
del fixup_legacy


#- exports -------------------------------------------------------------------
_thismodule = sys.modules[__name__]
for name in __all__:
    try:
        setattr(_thismodule, name, getattr(_backend, name))
    except AttributeError:
        pass
del name
nullptr = _backend.nullptr

def load_reflection_info(name):
    sc = _backend.gbl.gSystem.Load(name)
    if sc == -1:
        raise RuntimeError("Unable to load reflection library "+name)

def _begin_capture_stderr():
    pass

def _end_capture_stderr():
    return ""

# add other exports to all
__all__.append('load_reflection_info')
__all__.append('_backend')
__all__.append('_begin_capture_stderr')
__all__.append('_end_capture_stderr')
