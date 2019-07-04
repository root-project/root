# Author: Stefan Wunsch CERN  02/2019

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization
from libROOTPython import GetEndianess, GetTensorDataPointer, GetSizeOfType, AsRVec
from ROOT.pythonization._rvec import _array_interface_dtype_map


def get_array_interface(self):
    cppname = type(self).__cppname__
    for dtype in _array_interface_dtype_map:
        if not cppname.find("RTensor<{},".format(dtype)) is -1:
            dtype_numpy = _array_interface_dtype_map[dtype]
            dtype_size = GetSizeOfType(dtype)
            endianess = GetEndianess()
            shape = self.GetShape()
            strides = self.GetStrides()
            # Numpy breaks for data pointer of 0 even though the array is empty.
            # We set the pointer to 1 but the value itself is arbitrary and never accessed.
            pointer = GetTensorDataPointer(self, cppname)
            if pointer == 0:
                pointer == 1
            return {
                "shape": tuple(s for s in shape),
                "strides": tuple(s * dtype_size for s in strides),
                "typestr": "{}{}{}".format(endianess, dtype_numpy, dtype_size),
                "version": 3,
                "data": (pointer, False)
            }


def add_array_interface_property(klass, name):
    if True in [
            not name.find("RTensor<{},".format(dtype)) is -1 for dtype in _array_interface_dtype_map
    ]:
        klass.__array_interface__ = property(get_array_interface)


@pythonization()
def pythonize_rtensor(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name.startswith("TMVA::Experimental::RTensor<"):
        # Add numpy array interface
        add_array_interface_property(klass, name)

    return True
