# Author: Stefan Wunsch CERN  08/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
\pythondoc ROOT::VecOps::RVec

The ROOT::RVec class has additional features in Python, which allow to adopt memory
from Numpy arrays and vice versa. The purpose of these features is the copyless
interfacing of Python and C++ using their most common data containers, Numpy arrays
and RVec with a std::vector interface.

### Conversion of RVecs to Numpy arrays

RVecs of fundamental types (int, float, ...) have in Python the `__array_interface__`
attribute attached. This information allows Numpy to adopt the memory of RVecs without
copying the content. You can find further documentation regarding the Numpy array interface
[here](https://numpy.org/doc/stable/reference/arrays.interface.html). The following code example
demonstrates the memory adoption mechanism using `numpy.asarray`.

\code{.py}
rvec = ROOT.RVec('double')((1, 2, 3))
print(rvec) # { 1.0000000, 2.0000000, 3.0000000 }

npy = numpy.asarray(rvec)
print(npy) # [1. 2. 3.]

rvec[0] = 42
print(npy) # [42. 2. 3.]
\endcode

### Conversion of Numpy arrays to RVecs

Data owned by Numpy arrays with fundamental types (int, float, ...) can be adopted by RVecs. To
create an RVec from a Numpy array, ROOT offers the facility ROOT.VecOps.AsRVec, which performs
a similar operation to `numpy.asarray`, but vice versa. A code example demonstrating the feature and
the adoption of the data owned by the Numpy array is shown below.

\code{.py}
npy = numpy.array([1.0, 2.0, 3.0])
print(npy) # [1. 2. 3.]

rvec = ROOT.VecOps.AsRVec(npy)
print(rvec) # { 1.0000000, 2.0000000, 3.0000000 }

npy[0] = 42
print(rvec) # { 42.000000, 2.0000000, 3.0000000 }
\endcode

\endpythondoc
"""

import sys

import cppyy

from . import pythonization


_numpy_dtype_typeinfo_map = {
    "i1": {"typedef": "Char_t",         "cpp": "char"},
    "u1": {"typedef": "UChar_t",        "cpp": "unsigned char"},
    "i2": {"typedef": "Short_t",        "cpp": "short"},
    "u2": {"typedef": "UShort_t",       "cpp": "unsigned short"},
    "i4": {"typedef": "Int_t",          "cpp": "int"},
    "u4": {"typedef": "UInt_t",         "cpp": "unsigned int"},
    "i8": {"typedef": "Long64_t",       "cpp": "long long"},
    "u8": {"typedef": "ULong64_t",      "cpp": "unsigned long long"},
    "f4": {"typedef": "Float_t",        "cpp": "float"},
    "f8": {"typedef": "Double_t",       "cpp": "double"},
    "b1": {"typedef": "Bool_t",         "cpp": "bool"}
}


def _get_cpp_type_from_numpy_type(dtype):
    if dtype not in _numpy_dtype_typeinfo_map:
        raise RuntimeError("Object not convertible: Python object has unknown data-type '" + dtype + "'.")

    return _numpy_dtype_typeinfo_map[dtype]['cpp']


def _AsRVec(arr):
    r"""
    Adopt memory of a Python object with array interface using an RVec.

    \param[in] self self object
    \param[in] obj PyObject with array interface

    This function returns an RVec which adopts the memory of the given
    PyObject. The RVec takes the data pointer and the size from the array
    interface dictionary.
    Note that for arrays of strings, the input strings are copied into the RVec.
    """
    import math

    import numpy as np

    import ROOT

    # Get array interface of object
    interface = arr.__array_interface__

    # Get the data-pointer
    data = interface["data"][0]

    # Get the size of the contiguous memory
    shape = interface["shape"]
    size = math.prod(shape) if len(shape) > 0 else 0

    # Get the typestring and properties thereof
    typestr = interface["typestr"]
    dtype = typestr[1:]

    # Construct an RVec of strings
    if dtype == "O" or dtype.startswith("U"):
        underlying_object_types = {type(elem) for elem in arr}
        if len(underlying_object_types) > 1:
            raise TypeError(
                "All elements in the numpy array must be of the same type. Found types: {}".format(
                    underlying_object_types
                )
            )

        if underlying_object_types and underlying_object_types.pop() in [str, np.str_]:
            return ROOT.VecOps.RVec["std::string"](arr)
        else:
            raise TypeError("Cannot create an RVec from a numpy array of data type object.")

    if len(typestr) != 3:
        raise RuntimeError(
            "Object not convertible: __array_interface__['typestr'] returned '"
            + typestr
            + "' with invalid length unequal 3."
        )

    # Construct an RVec of the correct data-type
    cppdtype = _get_cpp_type_from_numpy_type(dtype)
    out = ROOT.VecOps.RVec[cppdtype](ROOT.module.cppyy.ll.reinterpret_cast[f"{cppdtype} *"](data), size)

    # Bind pyobject holding adopted memory to the RVec
    out.__adopted__ = arr

    return out


def get_array_interface(self):
    cppname = type(self).__cpp_name__
    for np_dtype, type_info in _numpy_dtype_typeinfo_map.items():
        root_typedef = type_info["typedef"]
        cpp_type     = type_info["cpp"]
        if cppname.endswith("<{}>".format(root_typedef)) or cppname.endswith("<{}>".format(cpp_type)):
            dtype_numpy = np_dtype
            endianness = "<" if sys.byteorder == "little" else ">"
            size = self.size()
            # Numpy breaks for data pointer of 0 even though the array is empty.
            # We set the pointer to 1 but the value itself is arbitrary and never accessed.
            if self.empty():
                pointer = 1
            else:
                pointer = cppyy.ll.addressof(self.data())
            return {
                "shape": (size,),
                "typestr": "{}{}".format(endianness, dtype_numpy),
                "version": 3,
                "data": (pointer, False),
            }


def add_array_interface_property(klass, name):
    if any(
        name.endswith(f"<{type_info['typedef']}>") or name.endswith(f"<{type_info['cpp']}>")
        for type_info in _numpy_dtype_typeinfo_map.values()
    ):
        klass.__array_interface__ = property(get_array_interface)


@pythonization("RVec<", ns="ROOT::VecOps", is_prefix=True)
def pythonize_rvec(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    # Add numpy array interface
    add_array_interface_property(klass, name)
