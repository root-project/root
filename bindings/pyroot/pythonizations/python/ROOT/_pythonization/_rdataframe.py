# Author: Stefan Wunsch, Massimiliano Galli CERN  02/2019

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from . import pythonization

# functools.partial does not add the self argument
# this is done by functools.partialmethod which is
# introduced only in Python 3.4
try:
    from functools import partialmethod
except ImportError:
    from functools import partial

    class partialmethod(partial):
        def __get__(self, instance, owner):
            if instance is None:
                return self
            return partial(self.func, instance, *(self.args or ()), **(self.keywords or {}))


def RDataFrameAsNumpy(df, columns=None, exclude=None, lazy=False):
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
        lazy: Determines whether this action is instant (False, default) or lazy (True).

    Returns:
        dict or AsNumpyResult: if instant (default), dict with column names as keys and
            1D numpy arrays with content as values; if lazy, AsNumpyResult containing
            the result pointers obtained from the Take actions.
    """
    # Sanitize input arguments
    if isinstance(columns, str):
        raise TypeError("The columns argument requires a list of strings")
    if isinstance(exclude, str):
        raise TypeError("The exclude argument requires a list of strings")

    # Early check for numpy
    try:
        import numpy
    except:
        raise ImportError("Failed to import numpy during call of RDataFrame.AsNumpy.")

    # Find all column names in the dataframe if no column are specified
    if not columns:
        columns = [str(c) for c in df.GetColumnNames()]

    # Exclude the specified columns
    if exclude == None:
        exclude = []
    columns = [col for col in columns if not col in exclude]

    # Register Take action for each column
    result_ptrs = {}
    for column in columns:
        column_type = df.GetColumnType(column)
        result_ptrs[column] = df.Take[column_type](column)

    result = AsNumpyResult(result_ptrs, columns)

    if lazy:
        return result
    else:
        return result.GetValue()


class AsNumpyResult(object):
    """Future-like class that represents the result of an AsNumpy call.

    Provides AsNumpy with laziness when it comes to triggering the event loop.

    Attributes:
        _columns (list): list of the names of the columns returned by
            AsNumpy.
        _py_arrays (dict): results of the AsNumpy action. The key is the
            column name, the value is the NumPy array for that column.
        _result_ptrs (dict): results of the AsNumpy action. The key is the
            column name, the value is the result pointer for that column.
    """
    def __init__(self, result_ptrs, columns):
        """Constructs an AsNumpyResult object.

        Parameters:
            result_ptrs (dict): results of the AsNumpy action. The key is the
                column name, the value is the result pointer for that column.
            columns (list): list of the names of the columns returned by
                AsNumpy.
        """

        self._result_ptrs = result_ptrs
        self._columns = columns
        self._py_arrays = None

    def GetValue(self):
        """Triggers, if necessary, the event loop to run the Take actions for
        the requested columns and produce the NumPy arrays as result.

        Returns:
            dict: key is the column name, value is the NumPy array for that
                column.
        """

        if self._py_arrays is None:
            import numpy
            from ROOT._pythonization._rdf_utils import ndarray

            # Convert the C++ vectors to numpy arrays
            self._py_arrays = {}
            for column in self._columns:
                cpp_reference = self._result_ptrs[column].GetValue()
                if hasattr(cpp_reference, "__array_interface__"):
                    tmp = numpy.asarray(cpp_reference) # This adopts the memory of the C++ object.
                    self._py_arrays[column] = ndarray(tmp, self._result_ptrs[column])
                else:
                    tmp = numpy.empty(len(cpp_reference), dtype=numpy.object)
                    for i, x in enumerate(cpp_reference):
                        tmp[i] = x # This creates only the wrapping of the objects and does not copy.
                    self._py_arrays[column] = ndarray(tmp, self._result_ptrs[column])

        return self._py_arrays

    def Merge(self, other):
        """
        Merges the numpy arrays in the dictionary of this object with the numpy
        arrays in the dictionary of the other object, modifying the attribute of
        this object inplace.

        Raises:
            - RuntimeError: if either of the method arguments doesn't already
                have filled the internal dictionary of numpy arrays.
            - ImportError: if the numpy module couldn't be imported.
            - ValueError: If the dictionaries of numpy arrays of the two
                arguments don't have exactly the same keys.
        """

        if self._py_arrays is None or other._py_arrays is None:
            raise RuntimeError("Merging instances of 'AsNumpyResult' failed because either of them didn't compute "
                               "their result yet. Make sure to call the 'GetValue' method on both objects before "
                               "trying to merge again.")

        try:
            import numpy
        except ImportError:
            raise ImportError("Failed to import numpy while merging two 'AsNumpyResult' instances.")

        if not self._py_arrays.keys() == other._py_arrays.keys():
            raise ValueError("The two dictionary of numpy arrays have different keys.")

        self._py_arrays = {
            key: numpy.concatenate([self._py_arrays[key],
                                    other._py_arrays[key]])
            for key in self._py_arrays
        }

    def __getstate__(self):
        """
        This function is called during the pickle serialization step. Return the
        dictionary of numpy arrays (i.e. the actual result of this `AsNumpy`
        call). Other attributes are not needed and the RResultPtr objects are
        not serializable at all.
        """
        return self.GetValue()

    def __setstate__(self, state):
        """
        This function is called during unserialization step. Sets the dictionary
        of numpy array of the unserialized object.
        """
        self._py_arrays = state


def _histo_profile(self, fixed_args, *args):
    # Check wheter the user called one of the HistoXD or ProfileXD methods
    # of RDataFrame with a tuple as first argument; in that case,
    # extract the tuple items to construct a model object and call the
    # original implementation of the method with that object.

    # Parameters:
    # self: instantiation of RDataFrame
    # fixed_args: tuple containing the original name of the method being
    # pythonised and the class of the model object to construct
    # args: arguments passed by the user when he calls e.g Histo1D

    original_method_name, model_class = fixed_args

    # Get the "original" method of the RDataFrame instantiation
    original_method = getattr(self, original_method_name)

    if args and isinstance(args[0], tuple):
        # Construct the model with the elements of the tuple
        # as arguments
        model = model_class(*args[0])
        # Call the original implementation of the method
        # with the model as first argument
        if len(args) > 1:
            res = original_method(model, *args[1:])
        else:
            # Covers the case of the overloads with only model passed
            # as argument
            res = original_method(model)
    # If the first argument is not a tuple, nothing to do, just call
    # the original implementation
    else:
        res = original_method(*args)

    return res


@pythonization("RInterface<", ns="ROOT::RDF", is_prefix=True)
def pythonize_rdataframe(klass):
    # Parameters:
    # klass: class to be pythonized

    from cppyy.gbl.ROOT import RDF

    # Add asNumpy feature
    klass.AsNumpy = RDataFrameAsNumpy

    # Replace the implementation of the following RDF methods
    # to convert a tuple argument into a model object
    methods_with_TModel = {
            'Histo1D' : RDF.TH1DModel,
            'Histo2D' : RDF.TH2DModel,
            'Histo3D' : RDF.TH3DModel,
            'Profile1D' : RDF.TProfile1DModel,
            'Profile2D' : RDF.TProfile2DModel
            }

    # Do e.g.:
    # klass._OriginalHisto1D = klass.Histo1D
    # klass.Histo1D = TH1DModel
    for method_name, model_class in methods_with_TModel.items():
        original_method_name = '_Original' + method_name
        setattr(klass, original_method_name, getattr(klass, method_name))
        # Fixed arguments to construct a partialmethod
        fixed_args = (original_method_name, model_class)
        # Replace the original implementation of the method
        # by a generic function _histo_profile with
        # (original_method_name, model_class) as fixed argument
        setattr(klass, method_name, partialmethod(_histo_profile, fixed_args))
