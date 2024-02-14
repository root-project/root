# Author: Stefan Wunsch, Massimiliano Galli, Enric Tejedor (02/2019), Pawan Johnson  CERN  07/2022

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r'''
\pythondoc ROOT::RDataFrame

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
of your RDataFrame as a dictionary of NumPy arrays. See a few simple examples below or a full tutorial [here](df026__AsNumpyArrays_8py.html).

\anchor asnumpy_scalar_columns
##### Scalar columns
If your column contains scalar values of fundamental types (e.g., integers, floats), `AsNumpy()` produces NumPy arrays with the appropriate `dtype`:
~~~{.py}
rdf = ROOT.RDataFrame(10).Define("int_col", "1").Define("float_col", "2.3")
print(rdf.AsNumpy(["int_col", "float_col"]))
# Output: {'int_col': array([...], dtype=int32), 'float_col': array([...], dtype=float64)}
~~~

Columns containing non-fundamental types (e.g., objects, strings) will result in NumPy arrays with `dtype=object`.

##### Collection Columns
If your column contains collections of fundamental types (e.g., std::vector<int>), `AsNumpy()` produces a NumPy array with `dtype=object` where each 
element is a NumPy array representing the collection for its corresponding entry in the column.

If the collection at a certain entry contains values of fundamental types, or if it is a regularly shaped multi-dimensional array of a fundamental type, 
then the numpy array representing the collection for that entry will have the `dtype` associated with the value type of the collection, for example:
~~~{.py}
rdf = rdf.Define("v_col", "std::vector<int>{{1, 2, 3}}")
print(rdf.AsNumpy(["v_col", "int_col", "float_col"]))
# Output: {'v_col': array([array([1, 2, 3], dtype=int32), ...], dtype=object), ...}
~~~

If the collection at a certain entry contains values of a non-fundamental type, `AsNumpy()` will fallback on the [default behavior](\ref asnumpy_scalar_columns) and produce a NumPy array with `dtype=object` for that collection.

For more complex collection types in your entries, e.g. when every entry has a jagged array value, refer to the section on [interoperability with AwkwardArray](\ref awkward_interop).

#### Processing data stored in NumPy arrays

In case you have data in NumPy arrays in Python and you want to process the data with ROOT, you can easily
create an RDataFrame using `ROOT.RDF.FromNumpy`. The factory function accepts a dictionary where
the keys are the column names and the values are NumPy arrays, and returns a new RDataFrame with the provided
columns.

Only arrays of fundamental types (integers and floating point values) are supported and the arrays must have the same length.
Data is read directly from the arrays: no copies are performed.

~~~{.py}
# Read data from NumPy arrays
# The column names in the RDataFrame are taken from the dictionary keys
x, y = numpy.array([1, 2, 3]), numpy.array([4, 5, 6])
df = ROOT.RDF.FromNumpy({"x": x, "y": y})

# Use RDataFrame as usual, e.g. write out a ROOT file
df.Define("z", "x + y").Snapshot("tree", "file.root")
~~~


\anchor awkward_interop
### Interoperability with [AwkwardArray](https://awkward-array.org/doc/main/user-guide/how-to-convert-rdataframe.html)

The function for RDataFrame to Awkward conversion is ak.from_rdataframe(). The argument to this function accepts a tuple of strings that are the RDataFrame column names. By default this function returns ak.Array type.

~~~{.py}
import awkward as ak
import ROOT

array = ak.from_rdataframe(
    df,
    columns=(
        "x",
        "y",
        "z",
    ),
)
~~~

The function for Awkward to RDataFrame conversion is ak.to_rdataframe().

The argument to this function requires a dictionary: { <column name string> : <awkward array> }. This function always returns an RDataFrame object.

The arrays given for each column have to be equal length:

~~~{.py}
array_x = ak.Array(
    [
        {"x": [1.1, 1.2, 1.3]},
        {"x": [2.1, 2.2]},
        {"x": [3.1]},
        {"x": [4.1, 4.2, 4.3, 4.4]},
        {"x": [5.1]},
    ]
)
array_y = ak.Array([1, 2, 3, 4, 5])
array_z = ak.Array([[1.1], [2.1, 2.3, 2.4], [3.1], [4.1, 4.2, 4.3], [5.1]])

assert len(array_x) == len(array_y) == len(array_z)

df = ak.to_rdataframe({"x": array_x, "y": array_y, "z": array_z})
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

\endpythondoc
'''

from __future__ import annotations

from typing import Iterable, Optional

from . import pythonization
from ._pyz_utils import MethodTemplateGetter, MethodTemplateWrapper


def RDataFrameAsNumpy(
    df: ROOT.RDataFrame,  # noqa: F821
    columns: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    lazy: bool = False,
):
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
        df: The RDataFrame to read out.
        columns: If None return all branches as columns, otherwise specify names in iterable.
        exclude: Exclude branches from selection.
        lazy: Determines whether this action is instant (False, default) or lazy (True).

    Returns:
        dict or AsNumpyResult: if instant (default), dict with column names as keys and
            1D numpy arrays with content as values; if lazy, AsNumpyResult containing
            the result pointers obtained from the Take actions.
    """

    import ROOT

    # Sanitize input arguments
    if isinstance(columns, str):
        raise TypeError("The columns argument requires an iterable of strings")
    if isinstance(exclude, str):
        raise TypeError("The exclude argument requires an iterable of strings")

    # Early check for numpy
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise ImportError("Failed to import numpy during call of RDataFrame.AsNumpy.")

    # Find all column names in the dataframe if no column are specified
    if not columns:
        columns = [str(c) for c in df.GetColumnNames()]

    # Exclude the specified columns
    if exclude is None:
        exclude = []
    columns = [col for col in columns if col not in exclude]

    # Register Take action for each column
    result_ptrs = {}
    for column in columns:
        column_type = df.GetColumnType(column)
        # bool columns should be taken as unsigned chars, because NumPy stores
        # bools in bytes - different from the std::vector<bool> returned by the
        # action, which might do some space optimization
        column_type = "unsigned char" if column_type == "bool" else column_type

        # If the column type is a class, make sure cling knows about it
        tclass = ROOT.TClass.GetClass(column_type)
        if tclass and not tclass.GetClassInfo():
            raise RuntimeError(
                f'The column named "{column}" is of type "{column_type}", which is not known to the ROOT interpreter. Please load the corresponding header files or dictionaries.'
            )

        result_ptrs[column] = df.Take[column_type](column)

    result = AsNumpyResult(result_ptrs, columns)

    return result if lazy else result.GetValue()


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

    def GetValue(self) -> dict:
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
                    tmp = numpy.asarray(cpp_reference)  # This adopts the memory of the C++ object.
                    self._py_arrays[column] = ndarray(tmp, self._result_ptrs[column])
                else:
                    tmp = numpy.empty(len(cpp_reference), dtype=object)
                    for i, x in enumerate(cpp_reference):
                        if hasattr(x, "__array_interface__"):
                            tmp[i] = numpy.asarray(x)
                        else:
                            tmp[i] = x

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
            raise RuntimeError(
                "Merging instances of 'AsNumpyResult' failed because either of them didn't compute "
                "their result yet. Make sure to call the 'GetValue' method on both objects before "
                "trying to merge again."
            )

        try:
            import numpy
        except ImportError:
            raise ImportError("Failed to import numpy while merging two 'AsNumpyResult' instances.")

        if not self._py_arrays.keys() == other._py_arrays.keys():
            raise ValueError("The two dictionary of numpy arrays have different keys.")

        self._py_arrays = {
            key: numpy.concatenate([self._py_arrays[key], other._py_arrays[key]]) for key in self._py_arrays
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


def _clone_asnumpyresult(res: AsNumpyResult) -> AsNumpyResult:
    """
    Clones the internal actions held by the input result and returns a new
    result.
    """
    import ROOT

    return AsNumpyResult(
        {col: ROOT.Internal.RDF.CloneResultAndAction(ptr) for (col, ptr) in res._result_ptrs.items()}, res._columns
    )


class HistoProfileWrapper(MethodTemplateWrapper):
    """
    Subclass of MethodTemplateWrapper that pythonizes HistoXD and ProfileXD
    method templates.
    It relies on the `_original_method` and `_extra_args` attributes of the
    superclass, to invoke the original implementation of the method template
    and get the model class, respectively.
    """

    def __call__(self, *args):
        """
        Pythonization of HistoXD and ProfileXD method templates.
        Checks whether the user made a call with a tuple as first argument; in
        that case, extracts the tuple items to construct a model object and
        calls the original implementation of the method with that object.

        Args:
            args: arguments of a HistoXD or ProfileXD call.

        Returns:
            return value of the original HistoXD or ProfileXD implementations.
        """

        (model_class,) = self._extra_args

        if args and isinstance(args[0], tuple):
            # Construct the model with the elements of the tuple
            # as arguments
            model = model_class(*args[0])
            # Call the original implementation of the method
            # with the model as first argument
            if len(args) > 1:
                res = self._original_method(model, *args[1:])
            else:
                # Covers the case of the overloads with only model passed
                # as argument
                res = self._original_method(model)
        # If the first argument is not a tuple, nothing to do, just call
        # the original implementation
        else:
            res = self._original_method(*args)

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
        "Histo1D": RDF.TH1DModel,
        "Histo2D": RDF.TH2DModel,
        "Histo3D": RDF.TH3DModel,
        "Profile1D": RDF.TProfile1DModel,
        "Profile2D": RDF.TProfile2DModel,
    }

    for method_name, model_class in methods_with_TModel.items():
        # Replace the original implementation of the method
        # with an object that can handle template arguments
        # and stores a reference to such implementation
        getter = MethodTemplateGetter(getattr(klass, method_name), HistoProfileWrapper, model_class)
        setattr(klass, method_name, getter)

    klass._OriginalFilter = klass.Filter
    klass._OriginalDefine = klass.Define
    from ._rdf_pyz import _PyDefine, _PyFilter

    klass.Filter = _PyFilter
    klass.Define = _PyDefine


def _make_name_rvec_pair(key, value):
    import ROOT

    # Get name of key
    if not isinstance(key, str):
        raise RuntimeError("Object not convertible: Dictionary key is not convertible to a string.")

    try:
        # Convert value to RVec and attach to dictionary
        pyvec = ROOT.VecOps.AsRVec(value)
    except TypeError as e:
        if "Cannot create an RVec from a numpy array of data type object" in str(e):
            raise RuntimeError(
                f"Failure in creating column '{key}' for RDataFrame: the input column type is 'object', which is not supported. Make sure your column type is supported."
            ) from e
        else:
            raise

    # Add pairs of column name and associated RVec to signature
    return ROOT.std.pair["std::string", type(pyvec)](key, ROOT.std.move(pyvec))


# For references to keep alive the NumPy arrays that are read by
# MakeNumpyDataFrame.
_numpy_data = {}


def _MakeNumpyDataFrame(np_dict):
    r"""
    Make an RDataFrame from a dictionary of numpy arrays

    \param[in] self Always null, since this is a module function.
    \param[in] pydata Dictionary with numpy arrays

    This function takes a dictionary of numpy arrays and creates an RDataFrame
    using the keys as column names and the numpy arrays as data.
    """
    import ROOT

    if not isinstance(np_dict, dict):
        raise RuntimeError("Object not convertible: Python object is not a dictionary.")

    if len(np_dict) == 0:
        raise RuntimeError("Object not convertible: Dictionary is empty.")

    args = (_make_name_rvec_pair(key, value) for key, value in np_dict.items())

    # How we keep the NumPy arrays around as long as the RDataSource is alive:
    #
    #  1. Cache a container with references to the NumPy arrays in a global
    #     dictionary. Note that we use a copy of the original dict as the
    #     container, because otherwise the caller of _MakeNumpyDataFrame can
    #     invalidate our cache by mutating the np_dict after the call.
    #
    # 2. Together with the array data, store a deleter function to delete the
    #    cache element in the cache itself.
    #
    # 3. The C++ side gets a reference to the deleter function via
    #    std::function. Note that the C++ side can only get a non-owning
    #    reference to the Python function, which is the reason why we have to
    #    keep the deleter alive in the cache itself.
    #
    # 4. The RDataSource calls the deleter in its destructor.

    np_dict_copy = dict(**np_dict)
    key = id(np_dict_copy)
    _numpy_data[key] = (lambda: _numpy_data.pop(key), np_dict_copy)
    deleter = ROOT.std.function["void()"](_numpy_data[key][0])
    return ROOT.Internal.RDF.MakeRVecDataFrame(deleter, *args)
