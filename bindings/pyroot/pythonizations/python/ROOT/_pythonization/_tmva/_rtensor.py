# Author: Stefan Wunsch CERN  07/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from .. import pythonization
from .._rvec import _array_interface_dtype_map, _get_cpp_type_from_numpy_type
import cppyy
import sys


def _AsRTensor(arr):
    r"""
    Adopt memory of a Python object with array interface using an RTensor.

    \param[in] self Always null, since this is a module function.
    \param[in] obj PyObject with array interface

    This function returns an RTensor which adopts the memory of the given
    PyObject. The RTensor takes the data pointer and the shape from the array
    interface dictionary.
    """
    import ROOT
    import math
    import platform

    # Get array interface of object
    interface = arr.__array_interface__

    # Get the data-pointer
    data = interface["data"][0]

    # Get the size of the contiguous memory
    shape = interface["shape"]
    size = math.prod(shape) if len(shape) > 0 else 0

    # Get the typestring and properties thereof
    typestr = interface["typestr"]
    if len(typestr) != 3:
        raise RuntimeError(
            "Object not convertible: __array_interface__['typestr'] returned '"
            + typestr
            + "' with invalid length unequal 3."
        )

    dtype = typestr[1:]
    dtypesize = int(typestr[-1])
    cppdtype = _get_cpp_type_from_numpy_type(dtype)

    # Get strides
    strides = arr.strides

    # Infer memory layout from strides
    layout_enum = ROOT.TMVA.Experimental.MemoryLayout
    layout = layout_enum.ColumnMajor if len(strides) > 1 and strides[0] < strides[-1] else layout_enum.RowMajor

    # Construct an RTensor of the correct data-type
    out = ROOT.TMVA.Experimental.RTensor[cppdtype, ROOT.std.vector[cppdtype]](
        ROOT.module.cppyy.ll.reinterpret_cast[f"{cppdtype} *"](data),
        [s for s in shape],
        [s // dtypesize for s in strides],
        layout,
    )

    # Bind pyobject holding adopted memory to the RTensor
    out.__adopted__ = arr

    return out


def get_array_interface(self):
    """
    Return the array interface dictionary

    Parameters:
        self: RTensor object
    Returns:
        Dictionary following the Numpy array interface specifications
    """
    cppname = type(self).__cpp_name__
    idx1 = cppname.find("RTensor<")
    idx2 = cppname.find(",", idx1)
    dtype = cppname[idx1 + 8 : idx2]
    dtype_numpy = _array_interface_dtype_map[dtype]
    dtype_size = cppyy.sizeof(dtype)
    endianness = "<" if sys.byteorder == "little" else ">"
    shape = self.GetShape()
    strides = self.GetStrides()
    # Numpy breaks for data pointer of 0 even though the array is empty.
    # We set the pointer to 1 but the value itself is arbitrary and never accessed.
    pointer = cppyy.ll.addressof(self.GetData())
    if pointer == 0:
        pointer == 1
    return {
        "shape": tuple(s for s in shape),
        "strides": tuple(s * dtype_size for s in strides),
        "typestr": "{}{}{}".format(endianness, dtype_numpy, dtype_size),
        "version": 3,
        "data": (pointer, False),
    }


def add_array_interface_property(klass, name):
    """
    Attach the array interface as property if the data-type of the RTensor
    elements is one of the supported basic types

    Parameters:
        klass: class to be pythonized
        name: string containing the name of the class
    """
    if True in [name.find("RTensor<{},".format(dtype)) != -1 for dtype in _array_interface_dtype_map]:
        klass.__array_interface__ = property(get_array_interface)


def RTensorGetitem(self, idx):
    """
    Implementation of the __getitem__ special function for RTensor

    Parameters:
        self: RTensor object
        idx: Indices passed to RTensor[indices] operator
    Returns:
        New RTensor object if indices represent a slice or the requested element
    """
    # Make single index iterable and convert to list
    if not hasattr(idx, "__len__"):
        idx = [idx]
    idx = list(idx)

    # Check shape
    shape = self.GetShape()
    if shape.size() != len(idx):
        raise Exception("RTensor with rank {} got {} indices.".format(shape.size(), len(idx)))

    # Convert negative indices and Nones
    isSlice = False
    for i, x in enumerate(idx):
        if type(x) == slice:
            isSlice = True
            start = 0 if x.start is None else x.start
            stop = shape[i] if x.stop is None else x.stop
            if stop < 0:
                stop += shape[i]
            if x.step != None:
                raise Exception("RTensor does not support slices with step size unequal 1.")
            idx[i] = slice(start, stop, None)
        else:
            if x < 0:
                idx[i] += shape[i]

    # If a slice is requested, return a new RTensor
    if isSlice:
        idxVec = cppyy.gbl.std.vector("vector<size_t>")(len(idx))
        for i, x in enumerate(idx):
            idxVec[i].resize(2)
            if type(x) == slice:
                idxVec[i][0] = x.start
                idxVec[i][1] = x.stop
            else:
                idxVec[i][0] = x
                idxVec[i][1] = x + 1
        return self.Slice(idxVec)

    # Otherwise, access element by array of indices
    idxVec = cppyy.gbl.std.vector("size_t")(len(idx))
    for i, x in enumerate(idx):
        idxVec[i] = x
    return self(idxVec)


def RTensorInit(self, *args):
    if len(args) == 1:
        try:
            import numpy as np
        except ImportError:
            raise ImportError("Failed to import numpy in RTensor constructor")

        if isinstance(args[0], np.ndarray):
            data = args[0]
            # conversion from numpy to buffer float/double * works only if C order
            if data.flags.c_contiguous:
                shape = data.shape
                from cppyy.gbl import TMVA

                layout = TMVA.Experimental.MemoryLayout.RowMajor
                return self._original_init_(data, shape, layout)
            else:
                raise ValueError(
                    "Can only convert C-contiguous Numpy arrays to RTensor but input array is Fortran-contiguous"
                )

    return self._original_init_(*args)


@pythonization("RTensor<", ns="TMVA::Experimental", is_prefix=True)
def pythonize_rtensor(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    # Add numpy array interface
    add_array_interface_property(klass, name)
    # Get elements, including slices
    klass.__getitem__ = RTensorGetitem
    # add initialization of RTensor (pythonization of constructor)
    klass._original_init_ = klass.__init__
    klass.__init__ = RTensorInit
