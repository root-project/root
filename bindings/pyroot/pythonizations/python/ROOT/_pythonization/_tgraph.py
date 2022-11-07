# Author: Enric Tejedor CERN  03/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import cppyy
from . import pythonization


def set_size(self, buf):
    # Parameters:
    # - self: graph object
    # - buf: buffer of doubles
    # Returns:
    # - buffer whose size has been set
    buf.reshape((self.GetN(),))
    return buf


# Create a composite pythonizor.
#
# A composite is a type of pythonizor, i.e. it is a callable that expects two
# parameters: a class proxy and a string with the name of that class.
# A composite is created with the following parameters:
# - A string to match the class/es to be pythonized
# - A string to match the method/s to be pythonized in the class/es
# - A callable that will post-process the return value of the matched method/s
#
# Here we create a composite that will match TGraph, TGraph2D and their error
# subclasses, and will pythonize their getter methods of the X,Y,Z coordinate
# and error arrays, which in C++ return a pointer to a double.
# The pythonization consists in setting the size of the array that the getter
# method returns, so that it is known in Python and the array is fully usable
# (its length can be obtained, it is iterable).
comp = cppyy.py.compose_method(
    "^TGraph(2D)?$|^TGraph.*Errors$", "GetE?[XYZ]$", set_size  # class to match  # method to match
)  # post-process function

# Add the composite to the list of pythonizors
cppyy.py.add_pythonization(comp)


def _numpy_array(buffer, n):
    # Helper to create a numpy array from a raw array pointer.
    #
    # Note: The data is copied.
    #
    # Args:
    #     buffer (cppyy.LowLevelView):
    #         The pointer to the beginning of the array data, usually
    #         obtained from a C++ function that returns a `double *`.
    #
    # Returns:
    #     numpy.ndarray
    import numpy as np

    if not buffer:
        return None

    a = np.copy(np.frombuffer(buffer, dtype=np.float64, count=n))
    return a


def _TGraphConstructor(self, *args):
    import numpy as np

    if isinstance(args[0], (np.ndarray, np.generic)):
        args = (args[0].size,) + args

    self._original__init__(*args)


def Get(self):
    return self.GetX(), self.GetY()


def _GetX(self):
    a = _numpy_array(self._Original_GetX(), self.GetN())
    return a


def _GetY(self):
    a = _numpy_array(self._Original_GetY(), self.GetN())
    return a


@pythonization("TGraph")
def pythonize_tgraph(klass):
    # Parameters:
    # klass: class to be pythonized
    # Support hist *= scalar
    klass._original__init__ = klass.__init__
    klass.__init__ = _TGraphConstructor
    klass.Get = Get
    klass._Original_GetX = klass.GetX
    klass.GetX = _GetX
    klass._Original_GetY = klass.GetY
    klass.GetY = _GetY
