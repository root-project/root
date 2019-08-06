# Author: Stefan Wunsch CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization
from libROOTPython import GetEndianess, GetDataPointer, GetSizeOfType, AsRVec


_array_interface_dtype_map = {
    "float": "f",
    "double": "f",
    "int": "i",
    "long": "i",
    "Long64_t": "i",
    "unsigned int": "u",
    "unsigned long": "u",
    "ULong64_t": "u",
}


def get_array_interface(self):
    cppname = type(self).__cpp_name__
    for dtype in _array_interface_dtype_map:
        if cppname.endswith("<{}>".format(dtype)):
            dtype_numpy = _array_interface_dtype_map[dtype]
            dtype_size = GetSizeOfType(dtype)
            endianess = GetEndianess()
            size = self.size()
            # Numpy breaks for data pointer of 0 even though the array is empty.
            # We set the pointer to 1 but the value itself is arbitrary and never accessed.
            if self.empty():
                pointer = 1
            else:
                pointer = GetDataPointer(self, cppname, "data")
            return {
                "shape": (size, ),
                "typestr": "{}{}{}".format(endianess, dtype_numpy, dtype_size),
                "version": 3,
                "data": (pointer, False)
            }


def add_array_interface_property(klass, name):
    if True in [
            name.endswith("<{}>".format(dtype)) for dtype in _array_interface_dtype_map
    ]:
        klass.__array_interface__ = property(get_array_interface)


@pythonization()
def pythonize_rvec(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name.startswith("ROOT::VecOps::RVec<"):
        # Add numpy array interface
        add_array_interface_property(klass, name)

    return True


# Add AsRVec feature as free function to the ROOT module
import cppyy
cppyy.gbl.ROOT.VecOps.AsRVec = AsRVec
