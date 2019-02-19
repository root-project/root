"""Dynamic C++ bindings generator.

This module provides dynamic bindings to C++ through Cling, the LLVM-based C++
interpreter, allowing interactive mixing of Python and C++. Example:

    >>> import cppyy
    >>> cppyy.cppdef(\"\"\"
    ... class MyClass {
    ... public:
    ...     MyClass(int i) : m_data(i) {}
    ...     int m_data;
    ... };\"\"\")
    True
    >>> from cppyy.gbl import MyClass
    >>> m = MyClass(42)
    >>> cppyy.cppdef(\"\"\"
    ... void say_hello(MyClass* m) {
    ...     std::cout << "Hello, the number is: " << m->m_data << std::endl;
    ... }\"\"\")
    True
    >>> MyClass.say_hello = cppyy.gbl.say_hello
    >>> m.say_hello()
    Hello, the number is: 42
    >>> m.m_data = 13
    >>> m.say_hello()
    Hello, the number is: 13
    >>>

For full documentation, see:
   https://cppyy.readthedocs.io/

"""

__author__ = 'Wim Lavrijsen <WLavrijsen@lbl.gov>'

__all__ = [
    'cppdef',                 # declare C++ source to Cling
    'include',                # load and jit a header file
    'load_library',           # load a shared library
    'sizeof',                 #  size of a C++ type
    'typeid',                 # typeid of a C++ type
    'add_include_path',       # add a path to search for headers
    'add_autoload_map',       # explicitly include an autoload map
    ]

from ._version import __version__

import os, sys, sysconfig

if not 'CLING_STANDARD_PCH' in os.environ:
    local_pch = os.path.join(os.path.dirname(__file__), 'allDict.cxx.pch')
    if os.path.exists(local_pch):
        os.putenv('CLING_STANDARD_PCH', local_pch)
        os.environ['CLING_STANDARD_PCH'] = local_pch

try:
    import __pypy__
    del __pypy__
    ispypy = True
except ImportError:
    ispypy = False

# import separately instead of in the above try/except block for easier to
# understand tracebacks
if ispypy:
    from ._pypy_cppyy import *
else:
    from ._cpython_cppyy import *
del ispypy


#- allow importing from gbl --------------------------------------------------
sys.modules['cppyy.gbl'] = gbl
sys.modules['cppyy.gbl.std'] = gbl.std


#- enable auto-loading -------------------------------------------------------
try:    gbl.gInterpreter.EnableAutoLoading()
except: pass


#- external typemap ----------------------------------------------------------
from . import _typemap
_typemap.initialize(_backend)


#- pythonization factories ---------------------------------------------------
from . import _pythonization as py
py._set_backend(_backend)


#--- CFFI style interface ----------------------------------------------------
def cppdef(src):
    """Declare C++ source <src> to Cling."""
    if not gbl.gInterpreter.Declare(src):
        return False
    return True

def load_library(name):
    """Explicitly load a shared library."""
    if name[:3] != 'lib':
        if not gbl.gSystem.FindDynamicLibrary(gbl.TString(name), True) and\
               gbl.gSystem.FindDynamicLibrary(gbl.TString('lib'+name), True):
            name = 'lib'+name
    sc = gbl.gSystem.Load(name)
    if sc == -1:
        raise RuntimeError("Unable to load library "+name)

def include(header):
    """Load (and JIT) header file <header> into Cling."""
    gbl.gInterpreter.Declare('#include "%s"' % header)

def c_include(header):
    """Load (and JIT) header file <header> into Cling."""
    gbl.gInterpreter.Declare("""extern "C" {
#include "%s"
}""" % header)

def add_include_path(path):
    """Add a path to the include paths available to Cling."""
    if not os.path.isdir(path):
        raise OSError("no such directory: %s" % path)
    gbl.gInterpreter.AddIncludePath(path)

# add access to Python C-API headers
add_include_path(sysconfig.get_path('include', 'posix_prefix' if os.name == 'posix' else os.name))

def add_autoload_map(fname):
    """Add the entries from a autoload (.rootmap) file to Cling."""
    if not os.path.isfile(fname):
        raise OSError("no such file: %s" % fname)
    gbl.gInterpreter.LoadLibraryMap(fname)

def _get_name(tt):
    if type(tt) == str:
        return tt
    try:
        ttname = tt.__cppname__
    except AttributeError:
        ttname = tt.__name__
    return ttname

_sizes = {}
def sizeof(tt):
    """Returns the storage size (in chars) of C++ type <tt>."""
    if not isinstance(tt, type) and not type(tt) == str:
        tt = type(tt)
    try:
        return _sizes[tt]
    except KeyError:
        sz = gbl.gInterpreter.ProcessLine("sizeof(%s);" % (_get_name(tt),))
        _sizes[tt] = sz
        return sz

_typeids = {}
def typeid(tt):
    """Returns the C++ runtime type information for type <tt>."""
    if not isinstance(tt, type):
        tt = type(tt)
    try:
        return _typeids[tt]
    except KeyError:
        tidname = 'typeid_'+str(len(_typeids))
        gbl.gInterpreter.ProcessLine(
            "namespace _cppyy_internal { auto* %s = &typeid(%s); }" %\
            (tidname, _get_name(tt),))
        tid = getattr(gbl._cppyy_internal, tidname)
        _typeids[tt] = tid
        return tid
