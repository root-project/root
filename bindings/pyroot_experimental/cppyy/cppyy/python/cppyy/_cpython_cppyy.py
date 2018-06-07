""" CPython-specific touch-ups
"""

from . import _stdcpp_fix
from cppyy_backend import loader

__all__ = [
    'gbl',
    'load_reflection_info',
    'addressof',
    'bind_object',
    'nullptr',
    '_backend',
    ]

# first load the dependency libraries of the backend, then pull in the
# libcppyy extension module
c = loader.load_cpp_backend()
import libcppyy as _backend
_backend._cpp_backend = c


import sys
if sys.hexversion < 0x3000000:
  # TODO: this reliese on CPPOverload cooking up a func_code object, which atm
  # is simply not implemented for p3 :/

  # convince inspect that PyROOT method proxies are possible drop-ins for python
  # methods and classes for pydoc
    import inspect

    inspect._old_isfunction = inspect.isfunction
    def isfunction(object):
        if type(object) == _backend.CPPOverload and not object.im_class:
            return True
        return inspect._old_isfunction( object )
    inspect.isfunction = isfunction

    inspect._old_ismethod = inspect.ismethod
    def ismethod(object):
        if type(object) == _backend.CPPOverload:
            return True
        return inspect._old_ismethod(object)
    inspect.ismethod = ismethod
    del isfunction, ismethod


### template support ---------------------------------------------------------
class Template(object):  # expected/used by ProxyWrappers.cxx in CPyCppyy
    def __init__(self, name):
        self.__name__ = name

    def __repr__(self):
        return "<cppyy.Template '%s' object at %s>" % (self.__name__, hex(id(self)))

    def __call__(self, *args):
        newargs = [self.__name__]
        for arg in args:
            if type(arg) == str:
                arg = ','.join(map(lambda x: x.strip(), arg.split(',')))
            newargs.append(arg)
        result = _backend.MakeCppTemplateClass(*newargs)

      # special case pythonization (builtin_map is not available from the C-API)
        if 'push_back' in result.__dict__:
            def iadd(self, ll):
                [self.push_back(x) for x in ll]
                return self
            result.__iadd__ = iadd

        return result

    def __getitem__(self, *args):
        if args and type(args[0]) == tuple:
            return self.__call__(*(args[0]))
        return self.__call__(*args)

_backend.Template = Template


#- :: and std:: namespaces ---------------------------------------------------
gbl = _backend.CreateScopeProxy('')
gbl.__class__.__repr__ = lambda cls : '<namespace cppyy.gbl at 0x%x>' % id(cls)
gbl.std =  _backend.CreateScopeProxy('std')
# for move, we want our "pythonized" one, not the C++ template
gbl.std.move  = _backend.move


#- fake namespace for interactive lazy lookups -------------------------------
class InteractiveLazy(object):
    def __getattr__(self, attr):
        if attr == '__all__':
            caller = sys.modules[sys._getframe(1).f_globals['__name__']]
            _backend._set_cpp_lazy_lookup(caller.__dict__)
        return []

sys.modules['cppyy.interactive'] = InteractiveLazy()
del InteractiveLazy


#- add to the dynamic path as needed -----------------------------------------
import os
def add_default_paths():
    try:
        for line in open('/etc/ld.so.conf'):
            f = line.strip()
            if (os.path.exists(f)):
                gbl.gSystem.AddDynamicPath(f)
    except IOError:
        pass
add_default_paths()
del add_default_paths


#- exports -------------------------------------------------------------------
addressof     = _backend.addressof
bind_object   = _backend.bind_object
nullptr       = _backend.nullptr

def load_reflection_info(name):
    sc = gbl.gSystem.Load(name)
    if sc == -1:
        raise RuntimeError("Unable to load reflection library "+name)
