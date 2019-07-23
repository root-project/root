# Author: Stefan Wunsch CERN  02/2019

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization
from libROOTPython import MakeNumpyDataFrame
# Import numpy lazily (needed for `ndarray` class)
try:
    import numpy
except:
    raise ImportError("Failed to import numpy.")

class ndarray(numpy.ndarray):
    """
    A wrapper class that inherits from numpy.ndarray and allows to attach the
    result pointer of the `Take` action in an `RDataFrame` event loop to the
    collection of values returned by that action.
    """
    def __new__(cls, numpy_array, result_ptr):
        obj = numpy.asarray(numpy_array).view(cls)
        obj.result_ptr = result_ptr
        obj.__class__.__name__ = "numpy.array"
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.result_ptr = getattr(obj, "result_ptr", None)


def RDataFrameAsNumpy(df, columns=None, exclude=None):
    """Read-out the RDataFrame as a collection of numpy arrays.

    The values of the dataframe are read out as numpy array of the respective type
    if the type is a fundamental type such as float or int. If the type of the column
    is a complex type, such as your custom class or a std::array, the returned numpy
    array contains Python objects of this type interpreted via PyROOT.

    Be aware that reading out custom types is much less performant than reading out
    fundamental types, such as int or float, which are supported directly by numpy.

    The reading is performed in multiple threads if the implicit multi-threading of
    ROOT is enabled.

    Note that this is an instant action of the RDataFrame graph and will trigger the
    event-loop.

    Parameters:
        columns: If None return all branches as columns, otherwise specify names in iterable.
        exclude: Exclude branches from selection.

    Returns:
        dict: Dict with column names as keys and 1D numpy arrays with content as values
    """
    # Find all column names in the dataframe if no column are specified
    if not columns:
        columns = [c for c in df.GetColumnNames()]

    # Exclude the specified columns
    if exclude == None:
        exclude = []
    columns = [col for col in columns if not col in exclude]

    # Register Take action for each column
    result_ptrs = {}
    for column in columns:
        column_type = df.GetColumnType(column)
        result_ptrs[column] = df.Take[column_type](column)

    # Convert the C++ vectors to numpy arrays
    py_arrays = {}
    for column in columns:
        cpp_reference = result_ptrs[column].GetValue()
        if hasattr(cpp_reference, "__array_interface__"):
            tmp = numpy.array(cpp_reference) # This adopts the memory of the C++ object.
            py_arrays[column] = ndarray(tmp, result_ptrs[column])
        else:
            tmp = numpy.empty(len(cpp_reference), dtype=numpy.object)
            for i, x in enumerate(cpp_reference):
                tmp[i] = x # This creates only the wrapping of the objects and does not copy.
            py_arrays[column] = ndarray(tmp, result_ptrs[column])

    return py_arrays


@pythonization()
def pythonize_rdataframe(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    # Add AsNumpy feature
    if name.startswith("ROOT::RDataFrame<") or name.startswith("ROOT::RDF::RInterface<"):
        klass.AsNumpy = RDataFrameAsNumpy

    return True

# Add MakeNumpyDataFrame feature as free function to the ROOT module
import cppyy
cppyy.gbl.ROOT.RDF.MakeNumpyDataFrame = MakeNumpyDataFrame
