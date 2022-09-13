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


def _numpy_getter(*args, **kwargs):
    # Parameters:
    # - args: Arguments
    # - kwargs: Keyword arguments
    # Returns:
    # - TH1 method according to datatype of the numpy array
    # - numpy arrays passed as arguments
    import ROOT
    import numpy as np

    _np_dtype_dict = {
        "float64": ROOT.TH1D,
        "float32": ROOT.TH1F,
        "int32": ROOT.TH1I,
        "int8": ROOT.TH1C,
        "int16": ROOT.TH1S,
    }

    try:
        npargs = tuple(arg if isinstance(arg, (np.ndarray, np.generic)) else "" for arg in args)
        if kwargs:
            args = tuple(kwargs.values())

        func = _np_dtype_dict[str(npargs[0].dtype)]
        return func(*args), npargs
    except:
        raise ValueError("Unsupported value/arguments passed.")


def FromNumpy(*args, **kwargs):
    r"""Function to create histogram object from Numpy arrays"""
    th, npval = _numpy_getter(*args, **kwargs)
    th.FillN(npval[0].size, *kval)
    return th


python_funcs = [
    FromNumpy,
]


@pythonization("TH1")
def pythonize_th1(klass):
    # Parameters:
    # klass: class to be pythonized
    # Support hist *= scalar
    klass.__imul__ = _imul

    for python_func in python_funcs:
        setattr(klass, python_func.__name__, python_func)
