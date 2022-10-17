# Author: Harshal Shende CERN  10/2022

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from . import pythonization
from ._tgraph import _numpy_array


def _TGraphErrorsConstructor(self, *args):
    import numpy as np

    if isinstance(args[0], (np.ndarray, np.generic)):
        args = (args[0].size,) + args

    self._original__init__(*args)


def GetErrors(self):
    return self.GetErrorX(), self.GetErrorY()


def _GetErrorX(self):
    a = _numpy_array(self._Original_GetX(), self.GetN())
    return a


def _GetErrorY(self):
    a = _numpy_array(self._Original_GetY(), self.GetN())
    return a


@pythonization("TGraphErrors")
def pythonize_tgrapherrors(klass):
    # Parameters:
    # klass: class to be pythonized
    # Support hist *= scalar
    klass._original__init__ = klass.__init__
    klass.__init__ = _TGraphErrorsConstructor
    klass.GetErrors = GetErrors
    klass._Original_GetErrorX = klass.GetErrorX
    klass.GetErrorX = _GetErrorX
    klass._Original_GetErrorY = klass.GetErrorY
    klass.GetErrorY = _GetErrorY
