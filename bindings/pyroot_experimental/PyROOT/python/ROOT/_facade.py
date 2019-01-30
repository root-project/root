import types

from cppyy import gbl as gbl_namespace
from libROOTPython import gROOT

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

        # Redirect lookups to cppyy's global namespace
        self.__class__.__getattr__ = lambda self, name: getattr(gbl_namespace, name)
        self.__class__.__setattr__ = lambda self, name, val: setattr(gbl_namespace, name, val)
