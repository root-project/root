import sys

try:
    import __pypy__

    del __pypy__
    ispypy = True
except ImportError:
    ispypy = False


# - fake namespace for interactive lazy lookups -------------------------------
class InteractiveLazy(object):
    def __init__(self, hook_okay):
        self._hook_okay = hook_okay

    def __getattr__(self, attr):
        import cppjit

        if attr == "__all__":
            # copy all exported items from cppjit itself
            for v in cppjit.__all__:
                self.__dict__[v] = getattr(cppjit, v)

            # add the lookup hook into cppjit.gbl if legal, or put it under 'g'
            # if not (PyPy and IPython for now)
            if self._hook_okay:
                caller = sys.modules[sys._getframe(1).f_globals["__name__"]]
                cppjit._backend._set_cpp_lazy_lookup(caller.__dict__)
                return cppjit.__all__
            else:
                self.__dict__["g"] = cppjit.gbl
                self.__dict__["std"] = cppjit.gbl.std
                return ["g", "std"] + cppjit.__all__
        return getattr(cppjit, attr)


sys.modules["cppjit.interactive"] = InteractiveLazy(
    not ispypy
    and not (hasattr(__builtins__, "__IPYTHON__") or "IPython" in sys.modules)
)
del InteractiveLazy, ispypy
