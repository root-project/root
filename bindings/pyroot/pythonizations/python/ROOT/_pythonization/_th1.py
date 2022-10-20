# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization
from ._th_utils import _numpy_getter, _numpy_content

# Multiplication by constant


def _imul(self, c):
    # Parameters:
    # - self: histogram
    # - c: constant by which to multiply the histogram
    # Returns:
    # - A multiplied histogram (in place)
    self.Scale(c)
    return self


_np_dtype_dict = {
    "float64": "TH1D",
    "float32": "TH1F",
    "int32": "TH1I",
    "int8": "TH1C",
    "int16": "TH1S",
}


def FromNumpy(*args, **kwargs):
    r"""Function to create histogram object from Numpy arrays"""
    th, npval = _numpy_getter(_np_dtype_dict, *args, **kwargs)
    th.FillN(npval[0].size, *npval)
    return th


def GetContent(self, firstbin=None, lastbin=None, retw2=None):
    import numpy as np

    arr = self.GetArray()
    arr.reshape((self.fN,))
    a = np.asarray(arr)
    if retw2:
        w2arr = self.GetSumw2().GetArray()
        w2arr.reshape((self.GetSumw2().fN,))
        w2 = np.asarray(w2arr)
        return a[firstbin:lastbin], w2[firstbin:lastbin]
    else:
        return a[firstbin:lastbin]


def GetErrors(self, firstbin=None, lastbin=None):
    import numpy as np

    arr = self.GetArray()
    arr.reshape((self.fN,))
    a = np.asarray(arr)
    if self.GetSumw2().fN > 0:
        w2arr = self.GetSumw2().GetArray()
        w2arr.reshape((self.GetSumw2().fN,))
        w2 = np.asarray(w2arr)
        err = np.sqrt(w2)
        return err[firstbin:lastbin]
    else:
        err = np.sqrt(a)
        return err[firstbin:lastbin]


def GetBinEdges(self):
    if self.GetXaxis().GetXbins().GetSize() > 0:
        edges = _numpy_content(self.GetXaxis().GetXbins())

        return edges
    else:
        import numpy as np

        nbins = self.GetXaxis().GetNbins()
        xmin = self.GetXaxis().GetXmin()
        xmax = self.GetXaxis().GetXmax()
        edges = np.linspace(xmin, xmax, nbins + 1)
        return edges


python_funcs = [FromNumpy, GetBinEdges, GetContent, GetErrors]


@pythonization("TH1")
def pythonize_th1(klass):
    # Parameters:
    # klass: class to be pythonized

    # Support hist *= scalar
    klass.__imul__ = _imul
    for func in python_funcs:
        setattr(klass, func.__name__, func)
