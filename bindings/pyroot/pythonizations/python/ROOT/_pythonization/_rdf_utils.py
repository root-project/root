# Author: Stefan Wunsch CERN, Vincenzo Eduardo Padulano (UniMiB, CERN) 07/2019

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from __future__ import annotations
from typing import Iterable
import numpy
import re


class ndarray(numpy.ndarray):
    """
    A wrapper class that inherits from `numpy.ndarray` and allows to attach the
    result pointer of the `Take` action in an `RDataFrame` event loop to the
    collection of values returned by that action. See
    https://docs.scipy.org/doc/numpy/user/basics.subclassing.html for more
    information on subclassing numpy arrays.
    """

    def __new__(cls, numpy_array, result_ptr):
        """
        Dunder method invoked at the creation of an instance of this class. It
        creates a numpy array with an `RResultPtr` as an additional
        attribute.
        """
        obj = numpy.asarray(numpy_array).view(cls)
        obj.result_ptr = result_ptr
        return obj

    def __array_finalize__(self, obj):
        """
        Dunder method that fills in the instance default `result_ptr` value.
        """
        if obj is None:
            return
        self.result_ptr = getattr(obj, "result_ptr", None)


def all_same_length(iterable: Iterable) -> bool:
    """
    Check if all elements in the given iterable have the same length. Returns True if the iterable is empty.
    """
    try:
        if len(iterable) == 0:
            return True
    except TypeError:
        return True

    reference = len(next(iter(iterable)))
    return all(len(item) == reference for item in iterable)


def is_templated_instance(obj, class_name: str) -> bool:
    """
    Check if the given object is a templated instance.
    """
    # check if it matches the pattern
    # <class cppyy.gbl.ROOT.VecOps.RVec<float> at 0x13150a350>

    regex = re.compile(rf'<class {class_name}<.*> at 0x[0-9a-f]+>')
    return bool(regex.match(str(type(obj))))
