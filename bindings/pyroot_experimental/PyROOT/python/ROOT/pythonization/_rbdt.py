# Author: Stefan Wunsch CERN  09/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization
from libROOTPython import AsRVec, AsRTensor
import numpy as np


def Compute(self, x):
    # numpy.array is a factory and the actual type of a numpy array is numpy.ndarray
    if isinstance(x, np.ndarray):
        if len(x.shape) == 1:
            x_ = AsRVec(x)
            y = self._OriginalCompute(x_)
            return np.asarray(y)
        elif len(x.shape) == 2:
            x_ = AsRTensor(x)
            y = self._OriginalCompute(x_)
            return np.asarray(y)
        else:
            raise Exception("Call to Compute can process only numpy arrays of rank 1 or 2.")

    # As fall-through we go to the original compute function and use the error-handling from cppyy
    return self._OriginalCompute(x)


@pythonization()
def pythonize_rbdt(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: name of the class

    if name.startswith("TMVA::Experimental::RBDT"):
        klass._OriginalCompute = klass.Compute
        klass.Compute = Compute

    return True
