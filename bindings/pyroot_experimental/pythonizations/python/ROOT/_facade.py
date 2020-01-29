import types
import sys
from functools import partial

import libcppyy as cppyy_backend
from cppyy import gbl as gbl_namespace
from libROOTPythonizations import gROOT

from ._application import PyROOTApplication


class PyROOTConfiguration(object):
    """Class for configuring PyROOT"""

    def __init__(self):
        self.IgnoreCommandLineOptions = False
        self.ShutDown = True


class _gROOTWrapper(object):
    """Internal class to manage lookups of gROOT in the facade.
       This wrapper calls _finalSetup on the facade when it
       receives a lookup, unless the lookup is for SetBatch.
       This allows to evaluate the command line parameters
       before checking if batch mode is on in _finalSetup
    """

    def __init__(self, facade):
        self.__dict__['_facade'] = facade
        self.__dict__['_gROOT'] = gROOT

    def __getattr__( self, name ):
        if name != 'SetBatch' and self._facade.__dict__['gROOT'] != self._gROOT:
            self._facade._finalSetup()
        return getattr(self._gROOT, name)

    def __setattr__(self, name, value):
        return setattr(self._gROOT, name, value)


class ROOTFacade(types.ModuleType):
    """Facade class for ROOT module"""

    def __init__(self, module, is_ipython):
        types.ModuleType.__init__(self, module.__name__)

        self.module = module
        # Importing all will be customised later
        self.module.__all__ = []

        self.__doc__  = module.__doc__
        self.__name__ = module.__name__
        self.__file__ = module.__file__

        # Inject gROOT global
        self.gROOT = _gROOTWrapper(self)

        # Expose some functionality from CPyCppyy extension module
        self._cppyy_exports = [ 'nullptr', 'bind_object', 'as_cobject',
                                'SetMemoryPolicy', 'kMemoryHeuristics', 'kMemoryStrict',
                                'SetOwnership' ]
        for name in self._cppyy_exports:
            setattr(self, name, getattr(cppyy_backend, name))
        # For backwards compatibility
        self.AddressOf = cppyy_backend.addressof
        self.MakeNullPointer = partial(self.bind_object, 0)
        self.BindObject = self.bind_object
        self.AsCObject = self.as_cobject

        # Initialize configuration
        self.PyConfig = PyROOTConfiguration()

        self._is_ipython = is_ipython

        # Redirect lookups to temporary helper methods
        # This lets the user do some actions before all the machinery is in place:
        # - Set batch mode in gROOT
        # - Set options in PyConfig
        self.__class__.__getattr__ = self._getattr
        self.__class__.__setattr__ = self._setattr

        # Setup import hook
        self._set_import_hook()

    def _set_import_hook(self):
        # This hook allows to write e.g:
        # from ROOT.A import a
        # instead of the longer:
        # from ROOT import A
        # from A import a
        try:
            import __builtin__
        except ImportError:
            import builtins as __builtin__  # name change in p3
        _orig_ihook = __builtin__.__import__
        def _importhook(name, *args, **kwds):
            if name[0:5] == 'ROOT.':
                try:
                    sys.modules[name] = getattr(self, name[5:])
                except Exception:
                    pass
            return _orig_ihook(name, *args, **kwds)
        __builtin__.__import__ = _importhook

    def _handle_import_all(self):
        # Called if "from ROOT import *" is executed in the app.
        # Customises lookup in Python's main module to also
        # check in C++'s global namespace

        if sys.hexversion >= 0x3000000:
            raise ImportError('"from ROOT import *" is not supported in Python 3')
        if self._is_ipython:
            import warnings
            warnings.warn('"from ROOT import *" is not supported in IPython')
            # Continue anyway, just in case it works

        # Get caller module (jump over the facade)
        caller = sys.modules[sys._getframe(2).f_globals['__name__']]

        # Inject some predefined attributes of the facade
        for name in self._cppyy_exports + [ 'gROOT', 'AddressOf' ]:
            caller.__dict__[name] = getattr(self, name)

        # Install the hook
        cppyy_backend._set_cpp_lazy_lookup(caller.__dict__)

        # Return empty list to prevent further copying
        return self.module.__all__

    def _fallback_getattr(self, name):
        # Try:
        # - in the global namespace
        # - in the ROOT namespace
        # - in gROOT (ROOT lists such as list of files,
        #   memory mapped files, functions, geometries ecc.)
        # The first two attempts allow to lookup
        # e.g. ROOT.ROOT.Math as ROOT.Math

        if name == '__all__':
            self._handle_import_all()

        try:
            return getattr(gbl_namespace, name)
        except AttributeError as err:
            try:
                return getattr(self.__dict__['ROOT_ns'], name)
            except AttributeError:
                res = gROOT.FindObject(name)
                if res:
                    return res
                else:
                    raise AttributeError(str(err))

    def _finalSetup(self):
        # Prevent this method from being re-entered through the gROOT wrapper
        self.__dict__['gROOT'] = gROOT

        # Setup interactive usage from Python
        self.__dict__['app'] = PyROOTApplication(self.PyConfig, self._is_ipython)
        if not self.gROOT.IsBatch():
            self.app.init_graphics()

        # Set memory policy to kUseHeuristics.
        # This restores the default in PyROOT which was changed
        # by new Cppyy
        self.SetMemoryPolicy(self.kMemoryHeuristics)

        # Redirect lookups to cppyy's global namespace
        self.__dict__['ROOT_ns'] = gbl_namespace.ROOT
        self.__class__.__getattr__ = self._fallback_getattr
        self.__class__.__setattr__ = lambda self, name, val: setattr(gbl_namespace, name, val)

    def _getattr(self, name):
        # Special case, to allow "from ROOT import gROOT" w/o starting the graphics
        if name == '__path__':
            raise AttributeError(name)

        self._finalSetup()

        return getattr(self, name)

    def _setattr(self, name, val):
        self._finalSetup()

        return setattr(self, name, val)

    # Inject version as __version__ property in ROOT module
    @property
    def __version__(self):
        return self.gROOT.GetVersion()
