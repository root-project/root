import types

import libcppyy as cppyy_backend
from cppyy import gbl as gbl_namespace
from libROOTPython import gROOT

from ._application import PyROOTApplication


class PyROOTConfiguration(object):
    """Class for configuring PyROOT"""

    def __init__(self):
        self.IgnoreCommandLineOptions = False


class ROOTFacade(types.ModuleType):
    """Facade class for ROOT module"""

    def __init__(self, module):
        types.ModuleType.__init__(self, module.__name__)

        self.module = module

        self.__doc__  = module.__doc__
        self.__name__ = module.__name__
        self.__file__ = module.__file__

        # Inject gROOT global
        self.gROOT = gROOT

        # Expose some functionality from CPyCppyy extension module
        cppyy_exports = [ 'Double', 'Long', 'nullptr', 'addressof' ]
        for name in cppyy_exports:
            setattr(self, name, getattr(cppyy_backend, name))

        # Initialize configuration
        self.PyConfig = PyROOTConfiguration()

        # Redirect lookups to temporary helper methods
        # This lets the user do some actions before all the machinery is in place:
        # - Set batch mode in gROOT
        # - Set options in PyConfig
        self.__class__.__getattr__ = self._getattr
        self.__class__.__setattr__ = self._setattr

    def _finalSetup(self):
        # Setup interactive usage from Python
        self.__dict__['app'] = PyROOTApplication(self.PyConfig)
        if not self.gROOT.IsBatch():
            self.app.init_graphics()

        # Redirect lookups to cppyy's global namespace
        self.__class__.__getattr__ = lambda self, name: getattr(gbl_namespace, name)
        self.__class__.__setattr__ = lambda self, name, val: setattr(gbl_namespace, name, val)

    def _getattr(self, name):
        self._finalSetup()

        return getattr(self, name)

    def _setattr(self, name, val):
        self._finalSetup()

        return setattr(self, name, val)
