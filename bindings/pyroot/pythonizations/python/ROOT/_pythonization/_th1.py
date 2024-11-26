# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization


# Multiplication by constant

def _imul(self, c):
    # Parameters:
    # - self: histogram
    # - c: constant by which to multiply the histogram
    # Returns:
    # - A multiplied histogram (in place)
    self.Scale(c)
    return self

# Fill with numpy array

def _FillNWithNumpyArray(self, *args):
    """
    Fill histogram with numpy array.
    Parameters:
    - self: histogram
    - args: arguments to FillN
            If the first argument is numpy.ndarray:
            - fills the histogram with this array
            - optional second argument is weights array,
              if not provided, weights of 1 are used
            Otherwise:
            - Arguments are passed directly to the original FillN method
    Returns:
    - Result of FillN
    Raises:
    - ValueError: If weights length doesn't match data length
    """
    import numpy as np

    if args and isinstance(args[0], np.ndarray):
        data = args[0]
        weights = np.ones(len(data)) if len(args) < 2 or args[1] is None else args[1]
        if len(weights) != len(data):
            raise ValueError(
                f"Length mismatch: data length ({len(data)}) != weights length ({len(weights)})"
            )
        return self._FillN(len(data), data, weights)
    else:
        return self._FillN(*args)


@pythonization('TH1')
def pythonize_th1(klass):
    # Parameters:
    # klass: class to be pythonized

    # Support hist *= scalar
    klass.__imul__ = _imul

    # Support hist.FillN(numpy_array) and hist.FillN(numpy_array, numpy_array)
    klass._FillN = klass.FillN
    klass.FillN = _FillNWithNumpyArray
