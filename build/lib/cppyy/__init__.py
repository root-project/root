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
    'cppexec',                # execute a C++ statement
    'macro',                  # attempt to evaluate a cpp macro
    'include',                # load and jit a header file
    'c_include',              # load and jit a C header file
    'load_library',           # load a shared library
    'nullptr',                # unique pointer representing NULL
    'sizeof',                 # size of a C++ type
    'typeid',                 # typeid of a C++ type
    'multi',                  # helper for multiple inheritance
    'add_include_path',       # add a path to search for headers
    'add_library_path',       # add a path to search for headers
    'add_autoload_map',       # explicitly include an autoload map
    'set_debug',              # enable/disable debug output
    ]

import ctypes
import os
import sys
import sysconfig
import warnings

try:
    import __pypy__
    del __pypy__
    ispypy = True
except ImportError:
    ispypy = False

from . import _typemap
from ._version import __version__

# import separately instead of in the above try/except block for easier to
# understand tracebacks
if ispypy:
    from ._pypy_cppyy import *
else:
    from ._cpython_cppyy import *


#- allow importing from gbl --------------------------------------------------
sys.modules['cppyy.gbl'] = gbl
sys.modules['cppyy.gbl.std'] = gbl.std


#- external typemap ----------------------------------------------------------
_typemap.initialize(_backend)               # also creates (u)int8_t mapper

try:
    gbl.std.int8_t  = gbl.int8_t            # ensures same _integer_ type
    gbl.std.uint8_t = gbl.uint8_t
except (AttributeError, TypeError):
    pass


#- pythonization factories ---------------------------------------------------
from . import _pythonization as py
py._set_backend(_backend)

def _standard_pythonizations(pyclass, name):
  # pythonization of tuple; TODO: placed here for convenience, but a custom case
  # for tuples on each platform can be made much more performant ...
    if name.find('tuple<', 0, 6) == 0:
        import cppyy
        pyclass._tuple_len = cppyy.gbl.std.tuple_size(pyclass).value
        def tuple_len(self):
            return self.__class__._tuple_len
        pyclass.__len__ = tuple_len
        def tuple_getitem(self, idx, get=cppyy.gbl.std.get):
            if idx < self.__class__._tuple_len:
                res = get[idx](self)
                try:
                    res.__life_line = self
                except Exception:
                    pass
                return res
            raise IndexError(idx)
        pyclass.__getitem__ = tuple_getitem

  # pythonization of std::string; placed here because it's simpler to write the
  # custom "npos" object (to allow easy result checking of find/rfind) in Python
    elif pyclass.__cpp_name__ == "std::string":
        class NPOS(0x3000000 <= sys.hexversion and int or long):
            def __eq__(self, other):
                return other == -1 or  int(self) == other
            def __ne__(self, other):
                return other != -1 and int(self) != other
        del pyclass.__class__.npos          # drop b/c is const data
        pyclass.npos = NPOS(pyclass.npos)

    return True

if not ispypy:
    py.add_pythonization(_standard_pythonizations, "std")
# TODO: PyPy still has the old-style pythonizations, which require the full
# class name (not possible for std::tuple ...)

# std::make_shared/unique create needless templates: rely on Python's introspection
# instead. This also allows Python derived classes to be handled correctly.
class py_make_smartptr(object):
    __slots__ = ['cls', 'ptrcls']
    def __init__(self, cls, ptrcls):
        self.cls    = cls
        self.ptrcls = ptrcls
    def __call__(self, *args):
        if len(args) == 1 and type(args[0]) == self.cls:
            obj = args[0]
        else:
            obj = self.cls(*args)
        return self.ptrcls[self.cls](obj)   # C++ takes ownership

class make_smartptr(object):
    __slots__ = ['ptrcls', 'maker']
    def __init__(self, ptrcls, maker):
        self.ptrcls = ptrcls
        self.maker  = maker
    def __call__(self, ptr):
        return py_make_smartptr(type(ptr), self.ptrcls)(ptr)
    def __getitem__(self, cls):
        try:
            if not cls.__module__ == int.__module__:
                return py_make_smartptr(cls, self.ptrcls)
        except AttributeError:
            pass
        if isinstance(cls, str) and not cls in ('int', 'float'):
            return py_make_smartptr(getattr(gbl, cls), self.ptrcls)
        return self.maker[cls]

gbl.std.make_shared = make_smartptr(gbl.std.shared_ptr, gbl.std.make_shared)
gbl.std.make_unique = make_smartptr(gbl.std.unique_ptr, gbl.std.make_unique)
del make_smartptr


#--- interface to Cling ------------------------------------------------------
class _stderr_capture(object):
    def __init__(self):
        self._capture = not gbl.gDebug and True or False
        self.err = ""

    def __enter__(self):
        if self._capture:
            _begin_capture_stderr()
        return self

    def __exit__(self, tp, val, trace):
        if self._capture:
            self.err = _end_capture_stderr()

def _cling_report(msg, errcode, msg_is_error=False):
  # errcode should be authorative, but at least on MacOS, Cling does not report an
  # error when it should, so also check for the typical compilation signature that
  # Cling puts out as an indicator than an error occurred
    if 'input_line' in msg:
        if 'warning' in msg and not 'error' in msg:
            warnings.warn(msg, SyntaxWarning)
            msg_is_error=False

        if 'error' in msg:
            errcode = 1

    if errcode or (msg and msg_is_error):
        raise SyntaxError('Failed to parse the given C++ code%s' % msg)

def cppdef(src):
    """Declare C++ source <src> to Cling."""
    with _stderr_capture() as err:
        errcode = gbl.gInterpreter.Declare(src)
    _cling_report(err.err, int(not errcode), msg_is_error=True)
    return True

def cppexec(stmt):
    """Execute C++ statement <stmt> in Cling's global scope."""
    if stmt and stmt[-1] != ';':
        stmt += ';'

  # capture stderr, but note that ProcessLine could legitimately be writing to
  # std::cerr, in which case the captured output needs to be printed as normal
    with _stderr_capture() as err:
        errcode = ctypes.c_int(0)
        try:
            gbl.gInterpreter.ProcessLine(stmt, ctypes.pointer(errcode))
        except Exception as e:
            sys.stderr.write("%s\n\n" % str(e))
            if not errcode.value:
                errcode.value = 1

    _cling_report(err.err, errcode.value)
    if err.err and err.err[1:] != '\n':
        sys.stderr.write(err.err[1:])

    return True

def macro(cppm):
    """Attempt to evalute a C/C++ pre-processor macro as a constant"""

    try:
        macro_val = getattr(getattr(gbl, '__cppyy_macros', None), cppm+'_', None)
        if macro_val is None:
            cppdef("namespace __cppyy_macros { auto %s_ = %s; }" % (cppm, cppm))
        return getattr(getattr(gbl, '__cppyy_macros'), cppm+'_')
    except Exception:
        pass

    raise ValueError('Failed to evaluate macro %s', cppm)


def load_library(name):
    """Explicitly load a shared library."""
    with _stderr_capture() as err:
        gSystem = gbl.gSystem
        if name[:3] != 'lib':
            if not gSystem.FindDynamicLibrary(gbl.TString(name), True) and\
                   gSystem.FindDynamicLibrary(gbl.TString('lib'+name), True):
                name = 'lib'+name
        sc = gSystem.Load(name)
    if sc == -1:
      # special case for Windows as of python3.8: use winmode=0, otherwise the default
      # will not consider regular search paths (such as $PATH)
        if 0x3080000 <= sys.hexversion and 'win32' in sys.platform and os.path.isabs(name):
            return ctypes.CDLL(name, ctypes.RTLD_GLOBAL, winmode=0)  # raises on error
        raise RuntimeError('Unable to load library "%s"%s' % (name, err.err))
    return True

def include(header):
    """Load (and JIT) header file <header> into Cling."""
    with _stderr_capture() as err:
        errcode = gbl.gInterpreter.Declare('#include "%s"' % header)
    if not errcode:
        raise ImportError('Failed to load header file "%s"%s' % (header, err.err))
    return True

def c_include(header):
    """Load (and JIT) header file <header> into Cling."""
    with _stderr_capture() as err:
        errcode = gbl.gInterpreter.Declare("""extern "C" {
#include "%s"
}""" % header)
    if not errcode:
        raise ImportError('Failed to load header file "%s"%s' % (header, err.err))
    return True

def add_include_path(path):
    """Add a path to the include paths available to Cling."""
    if not os.path.isdir(path):
        raise OSError('No such directory: %s' % path)
    gbl.gInterpreter.AddIncludePath(path)

def add_library_path(path):
    """Add a path to the library search paths available to Cling."""
    if not os.path.isdir(path):
        raise OSError('No such directory: %s' % path)
    gbl.gSystem.AddDynamicPath(path)

# add access to Python C-API headers
apipath = sysconfig.get_path('include', 'posix_prefix' if os.name == 'posix' else os.name)
if os.path.exists(apipath):
    add_include_path(apipath)
elif ispypy:
  # possibly structured without 'pythonx.y' in path
    apipath = os.path.dirname(apipath)
    if os.path.exists(apipath) and os.path.exists(os.path.join(apipath, 'Python.h')):
        add_include_path(apipath)

# add access to extra headers for dispatcher (CPyCppyy only (?))
if not ispypy:
    try:
        apipath_extra = os.environ['CPPYY_API_PATH']
        if os.path.basename(apipath_extra) == 'CPyCppyy':
            apipath_extra = os.path.dirname(apipath_extra)
    except KeyError:
        apipath_extra = None

    if apipath_extra is None:
        try:
            if 0x30a0000 <= sys.hexversion:
                import importlib.metadata as m

                for p in m.files('CPyCppyy'):
                    if p.match('API.h'):
                        ape = p.locate()
                        break
                del p, m
            else:
                import pkg_resources as pr

                d = pr.get_distribution('CPyCppyy')
                for line in d.get_metadata_lines('RECORD'):
                    if 'API.h' in line:
                        ape = os.path.join(d.location, line[0:line.find(',')])
                        break
                del line, d, pr

            if os.path.exists(ape):
                apipath_extra = os.path.dirname(os.path.dirname(ape))
            del ape
        except Exception:
            pass

    if apipath_extra is None:
        ldversion = sysconfig.get_config_var('LDVERSION')
        if not ldversion:
            ldversion = sys.version[:3]

        apipath_extra = os.path.join(os.path.dirname(apipath), 'site', 'python'+ldversion)
        if not os.path.exists(os.path.join(apipath_extra, 'CPyCppyy')):
            import glob, libcppyy
            ape = os.path.dirname(libcppyy.__file__)
          # a "normal" structure finds the include directory up to 3 levels up,
          # ie. dropping lib/pythonx.y[md]/site-packages
            for i in range(3):
                if os.path.exists(os.path.join(ape, 'include')):
                    break
                ape = os.path.dirname(ape)

            ape = os.path.join(ape, 'include')
            if os.path.exists(os.path.join(ape, 'CPyCppyy')):
                apipath_extra = ape
            else:
              # add back pythonx.y or site/pythonx.y if present
                for p in glob.glob(os.path.join(ape, 'python'+sys.version[:3]+'*'))+\
                         glob.glob(os.path.join(ape, '*', 'python'+sys.version[:3]+'*')):
                    if os.path.exists(os.path.join(p, 'CPyCppyy')):
                        apipath_extra = p
                        break

    if apipath_extra.lower() != 'none':
        if not os.path.exists(os.path.join(apipath_extra, 'CPyCppyy')):
            warnings.warn("CPyCppyy API not found (tried: %s); "
                          "set CPPYY_API_PATH envar to the 'CPyCppyy' API directory to fix"
                          % apipath_extra)
        else:
            add_include_path(apipath_extra)

    del apipath_extra

if os.getenv('CONDA_PREFIX'):
  # MacOS, Linux
    include_path = os.path.join(os.getenv('CONDA_PREFIX'), 'include')
    if os.path.exists(include_path):
        add_include_path(include_path)

  # Windows
    include_path = os.path.join(os.getenv('CONDA_PREFIX'), 'Library', 'include')
    if os.path.exists(include_path):
        add_include_path(include_path)

# assuming that we are in PREFIX/lib/python/site-packages/cppyy,
# add PREFIX/include to the search path
include_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), *(4*[os.path.pardir]+['include'])))
if os.path.exists(include_path):
    add_include_path(include_path)

del include_path, apipath, ispypy

def add_autoload_map(fname):
    """Add the entries from a autoload (.rootmap) file to Cling."""
    if not os.path.isfile(fname):
        raise OSError("no such file: %s" % fname)
    gbl.gInterpreter.LoadLibraryMap(fname)

def set_debug(enable=True):
    """Enable/disable debug output."""
    if enable:
        gbl.gDebug = 10
    else:
        gbl.gDebug =  0

def _get_name(tt):
    if isinstance(tt, str):
        return tt
    try:
        ttname = tt.__cpp_name__
    except AttributeError:
        ttname = tt.__name__
    return ttname

_sizes = {}
def sizeof(tt):
    """Returns the storage size (in chars) of C++ type <tt>."""
    if not isinstance(tt, type) and not isinstance(tt, str):
        tt = type(tt)
    try:
        return _sizes[tt]
    except KeyError:
        try:
            sz = ctypes.sizeof(tt)
        except TypeError:
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

def multi(*bases):      # after six, see also _typemap.py
    """Resolve metaclasses for multiple inheritance."""
  # contruct a "no conflict" meta class; the '_meta' is needed by convention
    nc_meta = type.__new__(
            type, 'cppyy_nc_meta', tuple(type(b) for b in bases if type(b) is not type), {})
    class faux_meta(type):
        def __new__(mcs, name, this_bases, d):
            return nc_meta(name, bases, d)
    return type.__new__(faux_meta, 'faux_meta', (), {})
