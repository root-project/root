# Author: Stefan Wunsch, Massimiliano Galli CERN  02/2019

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization

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
            from ROOT.pythonization._rdf_utils import ndarray

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

class RDFPythonHelper:
    def __init__(self):
        self._jitcounter = 0

    def get_arg_names_global(self, function):
        import ROOT

        cols = None

        for method in ROOT.gROOT.GetListOfGlobalFunctions(True):
            if method.GetName() == function:
                colstmp = []
                for arg in method.GetListOfMethodArgs():
                    if not arg.GetName():
                        raise Exception(f"Argument names not available from function {function}, cannot infer column names and none were explicitly provided.")
                    colstmp.append(arg.GetName())
                if cols is not None and colstmp != cols:
                    raise Exception(f"Cannot infer column names from overloaded function {function} with different argument names.")
                cols = colstmp

        # search templated methods
        ROOT.gROOT.GetListOfFunctionTemplates().Load()
        for method in ROOT.gROOT.GetListOfFunctionTemplates():
            if method.GetName() == function:
                colstmp = []
                for arg in method.Function().GetListOfMethodArgs():
                    if not arg.GetName():
                        raise Exception(f"Argument names not available from function {function}, cannot infer column names and none were explicitly provided.")
                    colstmp.append(arg.GetName())
                if cols is not None and colstmp != cols:
                    raise Exception(f"Cannot infer column names from overloaded function {function} with different argument names.")
                cols = colstmp


        if cols is None:
            raise Exception(f"Failed to infer column names from provided function {function}, and none were explicitly provided.")

        return cols

    def get_arg_names(self, cl, function):
        import ROOT

        if not cl:
            return self.get_arg_names_global(function)

        fullname = f"{cl}::{function}"

        cols = None

        # first try to load existing class info
        classinfo = ROOT.TClass.GetClass(cl)

        if not classinfo:
            # create temporary class info, which may be needed for interpreted types
            classinfo = ROOT.TClass(cl)

        if not classinfo.HasInterpreterInfo():
            raise Exception(f"Couldn't load class information for class {cl} and therefore cannot infer column names for function {fullname}, and none were explcitly provided.")

        # TODO consolidate shared code with other cases
        # search non-templated methods first
        for method in classinfo.GetListOfAllPublicMethods():
            if method.GetName() == function:
                colstmp = []
                for arg in method.GetListOfMethodArgs():
                    if not arg.GetName():
                        raise Exception(f"Argument names not available from function {fullname}, cannot infer column names and none were explicitly provided.")
                    colstmp.append(arg.GetName())
                if cols is not None and colstmp != cols:
                    raise Exception(f"Cannot infer column names from overloaded function {fullname} with different argument names.")
                cols = colstmp

        # search templated methods
        for method in classinfo.GetListOfFunctionTemplates():
            if method.GetName() == function:
                colstmp = []
                for arg in method.Function().GetListOfMethodArgs():
                    if not arg.GetName():
                        raise Exception(f"Argument names not available from function {fullname}, cannot infer column names and none were explicitly provided.")
                    colstmp.append(arg.GetName())
                if cols is not None and colstmp != cols:
                    raise Exception(f"Cannot infer column names from overloaded function {fullname} with different argument names.")
                cols = colstmp


        if cols is None:
            raise Exception(f"Failed to infer column names from provided function {fullname}, and none were explicitly provided.")

        return cols

    def _define(self, cppself, fixed_args, name, expression, cols = None):
        original_method_name, method_name = fixed_args
        original_method = getattr(cppself, original_method_name)

        islambdaexpr = isinstance(expression, str) and expression.lstrip().startswith("[")

        import ROOT

        if cols is not None:
          # FIXME currently this means there is no way to call functions taking std::vector input
          # arguments directly, since they will always be passed as RVec arguments
          coltypes = cppself.ValidatedArgTypes(cols, method_name, True)
          # cppyy doesn't like the std::string objects from C++ somehow...
          coltypes = [str(coltype) for coltype in coltypes]

        if isinstance(expression, str):
            # C++ lambda expression string
            # wrap in a callable because cppyy doesn't play nice with lambdas currently

            lambdaname = f"lambda_{self._jitcounter}"
            lambdatypename = f"LambdaT_{self._jitcounter}"
            classname = f"DefineWrapperT_{self._jitcounter}"

            if expression.lstrip().startswith("["):
                # already a lambda expression, use it directly
                lambdaexpr = expression
            else:
                # expression string needs to be parsed to construct full lambda expression
                if cols is not None:
                    raise ValueError("Column names cannot be explicitly provided when passing a result expression as a string.")

                # DON'T unpack directly or the std::tuple gets gc'd
                lambda_cols_types = cppself.BuildLambdaWithArgsAndTypes(expression, method_name)
                lambdaexpr, cols, coltypes = lambda_cols_types

                lambdaexpr = str(lambdaexpr)
                cols = [str(col) for col in cols]
                coltypes = [str(coltype) for coltype in coltypes]

            tojit = f"""
                namespace ROOT {{
                    namespace RDFPythonHelper {{
                        auto {lambdaname} = {lambdaexpr};
                        using {lambdatypename} = decltype({lambdaname});

                        template<typename... Args>
                        class {classname} {{

                        public:
                            auto operator() (typename ROOT::Internal::RDF::argument_t<Args>... x) {{ return {lambdaname}(x...); }}
                        }};

                    }};
                }};
            """

            status = ROOT.gInterpreter.Declare(tojit)

            if not status:
                raise Exception("failed to jit helper callable with code as below:\n" + tojit)

            self._jitcounter += 1

            if cols is None:
                cols = self.get_arg_names(f"ROOT::RDFPythonHelper::{lambdatypename}", "operator()")
                coltypes = cppself.ValidatedArgTypes(cols, method_name, True)
                # cppyy doesn't like the std::string objects from C++ somehow...
                coltypes = [str(coltype) for coltype in coltypes]

            expressiontargs = tuple(coltypes)
            expressionargs = tuple()

            # the below works, but can give extremely opaque error messages
            expressiontype = eval(f"ROOT.ROOT.RDFPythonHelper.{classname}")
            expression = expressiontype[expressiontargs](*expressionargs)


        elif type(expression).__name__ in ["CPPOverload", "TemplateProxy"]:
            # wrap in a callable to transparently handle overloaded functions and work around some cppyy limitations with function pointers

            exprname = None
            ftemplateargs = ""
            im_class = None
            im_self = None

            if type(expression).__name__ == "CPPOverload":
                overloads = [expression]
            elif type(expression).__name__ == "TemplateProxy":
                if expression.im_templateargs is not None:
                    ftemplateargs = expression.im_templateargs
                overloads = [expression.im_lowpriorityoverloads, expression.im_templatedoverloads, expression.im_nontemplatedoverloads]
                overloads = [overload for overload in overloads if overload is not None]

            for overload in overloads:
                if exprname is not None and overload.__name__ != exprname:
                    raise Exception("Mismatched function names between different overloads for templated function.")
                exprname = overload.__name__
                if overload.im_class is not None:
                    if im_class is not None and overload.im_class !=  im_class:
                        raise Exception("Mismatched class types between different overloads for templated function.")

                    im_class = overload.im_class
                if overload.im_self is not None:
                    if im_self is not None and overload.im_self !=  im_self:
                        raise Exception("Mismatched bound object between different overloads for templated function.")

                    im_self = overload.im_self

            if cols is None:
                cols = self.get_arg_names(im_class.__cpp_name__, exprname)
                coltypes = cppself.ValidatedArgTypes(cols, method_name, True)
                # cppyy doesn't like the std::string objects from C++ somehow...
                coltypes = [str(coltype) for coltype in coltypes]

            classname = f"DefineWrapperT_{self._jitcounter}"

            if im_self is None:
                # free function or static class function

                if im_class is None:
                    namespace = ""
                else:
                    namespace = im_class.__cpp_name__

                fname = namespace + "::" + exprname


                tojit = f"""
                    namespace ROOT {{
                        namespace RDFPythonHelper {{
                            template<typename... Args>
                            class {classname} {{

                            public:
                                auto operator() (typename ROOT::Internal::RDF::argument_t<Args>... x) {{ return {fname}{ftemplateargs}(x...); }}
                            }};
                        }};
                    }};
                """

                expressiontargs = tuple(coltypes)
                expressionargs = tuple()

            else:
                # bound member function
                fname = exprname

                tojit = f"""
                    namespace ROOT {{
                        namespace RDFPythonHelper {{
                            template<typename C, typename... Args>
                            class {classname} {{

                            public:
                                template<typename T>
                                {classname}(T &&callable) : callable_(std::forward<T>(callable)) {{}}

                                auto operator() (typename ROOT::Internal::RDF::argument_t<Args>... x) {{ return callable_.{fname}{ftemplateargs}(x...); }}

                            private:
                                C callable_;
                            }};
                        }};
                    }};
                """

                expressiontargs = tuple([type(im_class)] + coltypes)
                expressionargs = (im_self,)

            status = ROOT.gInterpreter.Declare(tojit)

            if not status:
                raise Exception("failed to jit helper callable with code as below:\n" + tojit)

            self._jitcounter += 1

            # the below works, but can give extremely opaque error messages
            expressiontype = eval(f"ROOT.ROOT.RDFPythonHelper.{classname}")
            expression = expressiontype[expressiontargs](*expressionargs)
        else:
            # Callable object can be used directly
            if cols is None:
                cols = self.get_arg_names(type(expression).__cpp_name__, "operator()")
                coltypes = cppself.ValidatedArgTypes(cols, method_name, True)
                # cppyy doesn't like the std::string objects from C++ somehow...
                coltypes = [str(coltype) for coltype in coltypes]


        # the below works, but can give extremely opaque error messages
        targs = (ROOT.TypeTraits.TypeList[tuple(coltypes)],)
        return original_method[targs](name, expression, cols)

@pythonization()
def pythonize_rdataframe(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name.startswith("ROOT::RDataFrame<") or name.startswith("ROOT::RDF::RInterface<"):
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

        helper = RDFPythonHelper()

        methods_with_pythonization = {
                'Define'  :  helper._define,
          }

        for method_name, method, in methods_with_pythonization.items():
            print("pythonize", method_name)
            original_method_name = '_Original' + method_name
            setattr(klass, original_method_name, getattr(klass, method_name))
            fixed_args = (original_method_name, method_name)
            setattr(klass, method_name, partialmethod(method, fixed_args))

    return True
