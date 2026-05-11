# Author: Stefan Wunsch, Enric Tejedor CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import sys

from . import pythonization
from ._rvec import _array_interface_dtype_map


def _data_vec_char(self):
    # vector<char>::data() returns char*.
    # Cppyy attempts to convert char* into Python string, but if the
    # character sequence is not null-terminated the conversion fails.
    # This is likely to happen with the result of vector<char>::data().
    # For the conversion char* -> str to succeed when calling data(),
    # temporarily append a null character to the vector<char>.
    self.push_back('\0')
    d = self._original_data()
    self.pop_back()
    return d


def _get_array_interface(self):
    import ROOT

    value_type = getattr(type(self), "value_type", None)
    dtype_numpy = _array_interface_dtype_map.get(value_type)
    if dtype_numpy is not None:
        dtype_size = ROOT._cppyy.sizeof(value_type)
        endianness = "<" if sys.byteorder == "little" else ">"
        size = self.size()
        # Numpy breaks for data pointer of 0 even though the array is empty.
        # We set the pointer to 1 but the value itself is arbitrary and never accessed.
        if self.empty():
            pointer = 1
        else:
            pointer = ROOT._cppyy.ll.addressof(self.data())
        return {
            "shape": (size,),
            "typestr": "{}{}{}".format(endianness, dtype_numpy, dtype_size),
            "version": 3,
            "data": (pointer, False),
        }


def _add_array_interface_property(klass):
    value_type = getattr(klass, "value_type", None)
    if value_type in _array_interface_dtype_map:
        klass.__array_interface__ = property(_get_array_interface)

@pythonization("vector<", ns="std", is_prefix=True)
def pythonize_stl_vector(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    # Add numpy array interface
    _add_array_interface_property(klass)

    # Inject custom vector<char>::data()
    value_type = getattr(klass, 'value_type', None)
    if value_type == 'char':
        klass._original_data = klass.data
        klass.data = _data_vec_char

    # Pretty printing at the Python prompt
    klass.__repr__ = lambda self: "{}{}".format(self.__class__.__name__, self)
