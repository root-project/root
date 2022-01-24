# Author: Stefan Wunsch, Massimiliano Galli CERN  02/2019

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r'''
/**
\class ROOT::RDataFrame
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly
\anchor python
## Efficient analysis in Python

You can use RDataFrame in Python thanks to the dynamic Python/C++ translation of [PyROOT](https://root.cern/manual/python). In general, the interface
is the same as for C++, a simple example follows.

~~~{.py}
df = ROOT.RDataFrame("myTree", "myFile.root")
sum = df.Filter("x > 10").Sum("y")
print(sum.GetValue())
~~~

### User code in the RDataFrame workflow

#### C++ code

In the simple example that was shown above, a C++ expression is passed to the Filter() operation as a string
(`"x > 0"`), even if we call the method from Python. Indeed, under the hood, the analysis computations run in
C++, while Python is just the interface language.

To perform more complex operations that don't fit into a simple expression string, you can just-in-time compile
C++ functions - via the C++ interpreter cling - and use those functions in an expression. See the following
snippet for an example:

~~~{.py}
# JIT a C++ function from Python
ROOT.gInterpreter.Declare("""
bool myFilter(float x) {
    return x > 10;
}
""")

df = ROOT.RDataFrame("myTree", "myFile.root")
# Use the function in an RDF operation
sum = df.Filter("myFilter(x)").Sum("y")
print(sum.GetValue())
~~~

To increase the performance even further, you can also pre-compile a C++ library with full code optimizations
and load the function into the RDataFrame computation as follows.

~~~{.py}
ROOT.gSystem.Load("path/to/myLibrary.so") # Library with the myFilter function
ROOT.gInterpreter.Declare('#include "myLibrary.h"') # Header with the declaration of the myFilter function
df = ROOT.RDataFrame("myTree", "myFile.root")
sum = df.Filter("myFilter(x)").Sum("y")
print(sum.GetValue())
~~~

A more thorough explanation of how to use C++ code from Python can be found in the [PyROOT manual](https://root.cern/manual/python/#loading-user-libraries-and-just-in-time-compilation-jitting).

#### Python code

ROOT also offers the option to compile Python functions with fundamental types and arrays thereof using [Numba](https://numba.pydata.org/).
Such compiled functions can then be used in a C++ expression provided to RDataFrame.

The function to be compiled should be decorated with `ROOT.Numba.Declare`, which allows to specify the parameter and
return types. See the following snippet for a simple example or the full tutorial [here](pyroot004__NumbaDeclare_8py.html).

~~~{.py}
@ROOT.Numba.Declare(["float"], "bool")
def myFilter(x):
    return x > 10

df = ROOT.RDataFrame("myTree", "myFile.root")
sum = df.Filter("Numba::myFilter(x)").Sum("y")
print(sum.GetValue())
~~~

It also works with collections: `RVec` objects of fundamental types can be transparently converted to/from numpy arrays:

~~~{.py}
@ROOT.Numba.Declare(['RVec<float>', 'int'], 'RVec<float>')
def pypowarray(numpyvec, pow):
    return numpyvec**pow

df.Define('array', 'ROOT::RVecF{1.,2.,3.}')\
  .Define('arraySquared', 'Numba::pypowarray(array, 2)')
~~~

Note that this functionality requires the Python packages `numba` and `cffi` to be installed.

### Interoperability with NumPy

#### Conversion to NumPy arrays

Eventually, you probably would like to inspect the content of the RDataFrame or process the data further
with Python libraries. For this purpose, we provide the `AsNumpy()` function, which returns the columns
of your RDataFrame as a dictionary of NumPy arrays. See a simple example below or a full tutorial [here](df026__AsNumpyArrays_8py.html).

~~~{.py}
df = ROOT.RDataFrame("myTree", "myFile.root")
cols = df.Filter("x > 10").AsNumpy(["x", "y"]) # retrieve columns "x" and "y" as NumPy arrays
print(cols["x"], cols["y"]) # the values of the cols dictionary are NumPy arrays
~~~

#### Processing data stored in NumPy arrays

In case you have data in NumPy arrays in Python and you want to process the data with ROOT, you can easily
create an RDataFrame using `ROOT.RDF.MakeNumpyDataFrame`. The factory function accepts a dictionary where
the keys are the column names and the values are NumPy arrays, and returns a new RDataFrame with the provided
columns.

Only arrays of fundamental types (integers and floating point values) are supported and the arrays must have the same length.
Data is read directly from the arrays: no copies are performed.

~~~{.py}
# Read data from NumPy arrays
# The column names in the RDataFrame are taken from the dictionary keys
x, y = numpy.array([1, 2, 3]), numpy.array([4, 5, 6])
df = ROOT.RDF.MakeNumpyDataFrame({"x": x, "y": y})

# Use RDataFrame as usual, e.g. write out a ROOT file
df.Define("z", "x + y").Snapshot("tree", "file.root")
~~~

### Construct histogram and profile models from a tuple

The Histo1D(), Histo2D(), Histo3D(), Profile1D() and Profile2D() methods return
histograms and profiles, respectively, which can be constructed using a model
argument.

In Python, we can specify the arguments for the constructor of such histogram or
profile model with a Python tuple, as shown in the example below:

~~~{.py}
# First argument is a tuple with the arguments to construct a TH1D model
h = df.Histo1D(("histName", "histTitle", 64, 0., 128.), "myColumn")
~~~

### AsRNode helper function

The ROOT::RDF::AsRNode function casts an RDataFrame node to the generic ROOT::RDF::RNode type. From Python, it can be used to pass any RDataFrame node as an argument of a C++ function, as shown below:

~~~{.py}
ROOT.gInterpreter.Declare("""
ROOT::RDF::RNode MyTransformation(ROOT::RDF::RNode df) {
    auto myFunc = [](float x){ return -x;};
    return df.Define("y", myFunc, {"x"});
}
""")

# Cast the RDataFrame head node
df = ROOT.RDataFrame("myTree", "myFile.root")
df_transformed = ROOT.MyTransformation(ROOT.RDF.AsRNode(df))

# ... or any other node
df2 = df.Filter("x > 42")
df2_transformed = ROOT.MyTransformation(ROOT.RDF.AsRNode(df2))
~~~
\htmlonly
</div>
\endhtmlonly

\anchor reference
*/
'''

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
