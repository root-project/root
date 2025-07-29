""" CPython-specific touch-ups
"""

import ctypes
import sys

from . import _stdcpp_fix

__all__ = [
    'gbl',
    'load_reflection_info',
    'addressof',
    'bind_object',
    'nullptr',
    'default',
    '_backend',
    '_begin_capture_stderr',
    '_end_capture_stderr'
    ]

# First load the dependency libraries of the backend, then pull in the libcppyy
# extension module. If the backed can't be loaded, it was probably linked
# statically into the extension module, so we don't error out at this point.
try:
    from cppyy_backend import loader
    c = loader.load_cpp_backend()
except ImportError:
    c = None

import libcppyy as _backend

if c is not None:
    _backend._cpp_backend = c

# explicitly expose APIs from libcppyy
_w = ctypes.CDLL(_backend.__file__, ctypes.RTLD_GLOBAL)


# some beautification for inspect (only on p2)
if sys.hexversion < 0x3000000:
  # TODO: this reliese on CPPOverload cooking up a func_code object, which atm
  # is simply not implemented for p3 :/

  # convince inspect that cppyy method proxies are possible drop-ins for python
  # methods and classes for pydoc
    import inspect

    inspect._old_isfunction = inspect.isfunction
    def isfunction(object):
        if isinstance(object, _backend.CPPOverload) and not object.im_class:
            return True
        return inspect._old_isfunction(object)
    inspect.isfunction = isfunction

    inspect._old_ismethod = inspect.ismethod
    def ismethod(object):
        if isinstance(object, _backend.CPPOverload):
            return True
        return inspect._old_ismethod(object)
    inspect.ismethod = ismethod
    del isfunction, ismethod


### template support ---------------------------------------------------------
class Template(object):  # expected/used by ProxyWrappers.cxx in CPyCppyy
    stl_sequence_types   = ['std::vector', 'std::list', 'std::set', 'std::deque']
    stl_unrolled_types   = ['std::pair']
    stl_fixed_size_types = ['std::array']
    stl_mapping_types    = ['std::map', 'std::unordered_map']

    def __init__(self, name):
        self.__name__     = name
        self.__cpp_name__ = name
        self._instantiations = dict()

    def __repr__(self):
        return "<cppyy.Template '%s' object at %s>" % (self.__name__, hex(id(self)))

    def __getitem__(self, *args):
      # multi-argument to [] becomes a single tuple argument
        if args and isinstance(args[0], tuple):
            args = args[0]

      # if already instantiated, return the existing class
        try:
            return self._instantiations[args]
        except KeyError:
            pass

      # construct the type name from the types or their string representation
        newargs = [self.__name__]
        for arg in args:
            if isinstance(arg, str):
                arg = ','.join(map(lambda x: x.strip(), arg.split(',')))
            newargs.append(arg)
        pyclass = _backend.MakeCppTemplateClass(*newargs)

      # memoize the class to prevent spurious lookups/re-pythonizations
        self._instantiations[args] = pyclass

      # special case pythonization (builtin_map is not available from the C-API)
        if 'push_back' in pyclass.__dict__ and not '__iadd__' in pyclass.__dict__:
            if 'reserve' in pyclass.__dict__:
                def iadd(self, ll):
                    self.reserve(len(ll))
                    for x in ll:
                        self.push_back(x)
                    return self
            else:
                def iadd(self, ll):
                    for x in ll:
                        self.push_back(x)
                    return self
            pyclass.__iadd__ = iadd

      # back-pointer for reflection
        pyclass.__cpp_template__ = self

        return pyclass

    def __call__(self, *args):
      # for C++17, we're required to derive the type when using initializer syntax
      # (i.e. a tuple or list); not sure how to do that in general, but below the
      # most common cases are covered
        if args:
            args0 = args[0]
            if args0 and isinstance(args0, (tuple, list)):
                t = type(args0[0])
                if t is float: t = 'double'

                if self.__name__ in self.stl_sequence_types:
                    return self[t](*args)
                if self.__name__ in self.stl_fixed_size_types:
                    return self[t, len(args0)](*args)
                if self.__name__ in self.stl_unrolled_types:
                    return self[tuple(type(a) for a in args0)](*args0)

            if args0 and isinstance(args0, dict):
                if self.__name__ in self.stl_mapping_types:
                    try:
                        pair = args0.items().__iter__().__next__()
                    except AttributeError:
                        pair = args0.items()[0]
                    t1 = type(pair[0])
                    if t1 is float: t1 = 'double'
                    t2 = type(pair[1])
                    if t2 is float: t2 = 'double'
                    return self[t1, t2](*args)

                return self.__getitem__(*(type(a) for a in args0))(*args)

      # old 'metaclass-style' template instantiations
        return self.__getitem__(*args)

_backend.Template = Template


#- :: and std:: namespaces ---------------------------------------------------
gbl = _backend.CreateScopeProxy('')
gbl.__class__.__repr__ = lambda cls : '<namespace cppyy.gbl at 0x%x>' % id(cls)
gbl.std =  _backend.CreateScopeProxy('std')
# for move, we want our "pythonized" one, not the C++ template
gbl.std.move  = _backend.move


#- add to the dynamic path as needed -----------------------------------------
import os
def add_default_paths():
    gSystem = gbl.gSystem
    if os.getenv('CONDA_PREFIX'):
      # MacOS, Linux
        lib_path = os.path.join(os.getenv('CONDA_PREFIX'), 'lib')
        if os.path.exists(lib_path): gSystem.AddDynamicPath(lib_path)

      # Windows
        lib_path = os.path.join(os.getenv('CONDA_PREFIX'), 'Library', 'lib')
        if os.path.exists(lib_path): gSystem.AddDynamicPath(lib_path)

  # assuming that we are in PREFIX/lib/python/site-packages/cppyy, add PREFIX/lib to the search path
    lib_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
    if os.path.exists(lib_path): gSystem.AddDynamicPath(lib_path)

    try:
        with open('/etc/ld.so.conf') as ldconf:
            for line in ldconf:
                f = line.strip()
                if (os.path.exists(f)):
                    gSystem.AddDynamicPath(f)
    except IOError:
        pass
add_default_paths()
del add_default_paths


#- exports -------------------------------------------------------------------
addressof     = _backend.addressof
bind_object   = _backend.bind_object
nullptr       = _backend.nullptr
default       = _backend.default

def load_reflection_info(name):
    sc = gbl.gSystem.Load(name)
    if sc == -1:
        raise RuntimeError("Unable to load reflection library "+name)

def _begin_capture_stderr():
    _backend._begin_capture_stderr()

def _end_capture_stderr():
    err = _backend._end_capture_stderr()
    if err:
        try:
            return "\n%s" % err
        except UnicodeDecodeError as e:
            original_error = e
        try:
            return "\n%s" % err.decode('gbk')       # not guaranteed, but common
        except UnicodeDecodeError:
            pass
        return "C++ issued an error message that could not be decoded (%s)" % str(original_error)
    return ""
