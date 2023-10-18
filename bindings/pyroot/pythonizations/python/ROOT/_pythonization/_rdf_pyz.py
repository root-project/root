# Author: Pawan Johnson, Vincenzo Eduardo Padulano, Enric Tejedor  CERN  07/2022

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
import re
import typing
from .._numbadeclare import _NumbaDeclareDecorator

import libcppyy

class FunctionJitter:
    """
    This class allows to jit a python callable with Numba, being able to infer the signature of the function from the types of the RDF columns.
    It takes a python callable, an RDF and optionally a column list and dictionary containing any other arguments for the callable.
    It determines the function signature and Declares the function and returns the corresponding function call.

    Attributes
    ----------
    rdf: RDataFrame object on which the function's column arguments are in
    col_names: The list of columns in rdf
    func: The python callable that is to be jitted.
    return_type: Return type of above callable
    params: List of parameters of the above callable
    func_args: Dict to store the value of each param
    args_info: Information about the type and value of each param
    func_sign: Contains the function signature
    func_call: Contains the function call as a string.

    Example:
    rdf = ROOT.RDataFrame(5).Define("x", "(double) rdfentry_")
    def x_greater_than_2(x):
        return x>2
    fj = FunctionJitter(rdf)
    func_call = fj.jit(x_greater_than_2)
    fil1 = rdf.Filter( "Numba::" + func_call, "x is greater than 2")

    """
    # Variable to store previous functions so as to not rejit.
    function_cache = {}
    lambda_function_counter = 0  # Counter to name the lambda functions

    def __init__(self, rdf: 'RDataFrame') -> None:
        self.rdf = rdf
        self.col_names: typing.List[str] = rdf.GetColumnNames()
        self.func: typing.Callable
        self.return_type: str
        self.params: typing.List[str]
        self.func_args: typing.Dict(str, any)
        self.args_info: typing.Dict(str, (str, any))
        self.func_sign: str
        self.func_call: str

    def find_type(self, x):
        """
        Function to determine the type of a variable x.
        If x is a string. It maps it to the corresponding column's type in RDF. (String constants are not supported)
        Else it determines the fundamental type of x. (int, float, bool)
        Else it is an a numpy array and maps it to corresponding RVec.
        Otherwise flags a type error
        Args:
	    `   x: Variable whose type is to be determined

        Returns:
            type of the variable x
        """
        try:
            import numpy as np
        except:
            raise ImportError(
                "Failed to import numpy during call to determine function signature.")
        from ._rdf_conversion_maps import FUNDAMENTAL_PYTHON_TYPES, TREE_TO_NUMBA, NUMPY_TO_TREE
        if isinstance(x, str):
            # Can be string constant or can be column name
            if x in self.col_names:  # If x is a column
                t = self.rdf.GetColumnType(x)
                if t in TREE_TO_NUMBA:  # The column is a fundamental type from tree
                    return TREE_TO_NUMBA[t]
                elif '<' in t:  # The column type is a RVec<type>
                    if '>>' in t:  # It is a RVec<RVec<T>>
                        raise TypeError(
                            f"Only columns with 'RVec<T>' where T is is a fundamental type are supported, not '{t}'.")
                    g = re.match('(.*)<(.*)>', t).groups(0)
                    if g[1] in TREE_TO_NUMBA:
                        return "RVec<" + TREE_TO_NUMBA[g[1]] + ">"
                    # There are data type that leak into here. Not sure from where. But need to implement something here such that this condition is never met.
                    return "RVec<" + str(g[1]) + ">"

                else:
                    return t
            else:
                return 'str'
                #! Numba Declare does not support "string" type. Check _numbadeclare.Thus, Cannot pass string constants to the filter/Defines..
        elif type(x) in FUNDAMENTAL_PYTHON_TYPES:
            return FUNDAMENTAL_PYTHON_TYPES[type(x)]
        elif isinstance(x, np.ndarray):
            if x.dtype.type in NUMPY_TO_TREE:
                return "RVec<" + NUMPY_TO_TREE[x.dtype.type] + ">"
            else:
                raise TypeError(
                    f"Support for {x.dtype.type} arrays is not yet supported.")
        #! Need to work out how to map things like tuples, dicts, lists...
        else:
                raise TypeError(
                    f"Type of {type(x).__name__}:  {x} cannot be jitted.")

    def find_function_params(self, func):
        """
        Function to create a list of parameters func needs.
        Updates the class to have a new attribute params which is a list of the parameters of func.

        Arguments:
            func: A python callable

        """
        import inspect
        func_sign = inspect.signature(func)
        # Find the Return type
        if func_sign.return_annotation is inspect.Signature.empty:
            self.return_type = None
        else:
            self.return_type = str(func_sign.return_annotation)
        self.func = func
        # List of input parameters for function
        # ALl the input parameters the function needs.
        self.params = list(func_sign.parameters.keys())
        self.args_info = {}

    def generate_func_args(self, cols_list, extra_args):
        """
        Function to create a dictionary mapping the parameters of the function to the corresponding columns or values.
        Updates the class to have a new attribute func_args which contains the above mapping.

        Arguments:
            cols_list: A python list that contains the columns which the function uses.
            extra_args: A dictionary containing any extra parameters the function requires.
        """
        self.func_args = {}
        n_params = len(self.params)
        n_cols = len(cols_list)
        n_constants = len(extra_args)

        if n_cols > 0:
            # Check to see if all the parameters have been provided.
            if n_params != n_cols + n_constants:
                raise ValueError("Not Enough values provided in the column list and extra_args. The function required {} parameters only {} provided.".format(
                    n_params, n_cols+n_constants))

        # Mapping the column list to the first input parameters of the function.
        for idx, p in enumerate(self.params):
            if idx < n_cols:
                self.func_args[p] = cols_list[idx]
        # Extra args supersedes col_list
        self.func_args = {**self.func_args, **extra_args}

    def find_function_signature(self):
        """
        Calculates the function signature.
        Updates the class with a new attribute args_info which contains the mapping of the parameter of the function to its type and value.
        args_info:  dict{parameter_name(str): (type, value)}
        """
        func_args = self.func_args
        for p in self.params:
            if p in func_args:  # the parameter value has been given in func_args
                value_of_p = func_args[p]
                type_of_p = self.find_type(value_of_p)
                # Bool(s) in python are represented as True/False but in C++ are true/false. The following if statements are to account for that
                if type(value_of_p) == bool:
                    if value_of_p: value_of_p = 'true'
                    else: value_of_p = 'false'
            else:  # the parameter was not in func_args. Thus this parameter has to be mapped to a column of rdf
                if p not in self.col_names:
                    raise Exception(
                        f"Unable to map function argument {p} to a column.\nUse correct name of column or pass a list of column names.")
                value_of_p = p
                type_of_p = self.find_type(p)
            self.args_info[p] = (type_of_p, value_of_p)

    def generate_function_call(self):
        """
        Generates a function call and signature. Gives a unique name to the function if it is a lambda function.
        Updates the class with new attributes func_call and func_sign to hold them,
        """
        func = self.func
        if func.__name__ == '<lambda>':
            func.__name__ = f"_lambda_func_number_{FunctionJitter.lambda_function_counter}"
            FunctionJitter.lambda_function_counter += 1
        self.func_call = f"{func.__name__}({', '.join(str(arg_info[1]) for arg_info in self.args_info.values())})"
        self.func_sign = [str(self.args_info[p][0]) for p in self.params]

    def get_function_params_args_call(self, func, cols_list, extra_args):
        """
        Function to generate the function params, args, signature and call.
        """
        self.find_function_params(func)
        self.generate_func_args(cols_list, extra_args)
        self.find_function_signature()
        self.generate_function_call()

    def jit_function(self, func, cols_list, extra_args):
        """
        Jits the provided function using ROOT's NumbaDeclare.
        Also checks if the function was jitted earlier in which case it won't jit again but if signature does not match. It raises an error.

        Arguments:
        func: A python callable
        cols_list: A list of columns of RDF on which func depends on.
        extra_args: A dict of extra arguments that func requires.
        """

        if func.__name__ in FunctionJitter.function_cache:
            func_call, func_sign = FunctionJitter.function_cache[func.__name__]
            self.get_function_params_args_call(func, cols_list, extra_args)
            if self.func_sign != func_sign:
                raise ValueError(
                    "Trying to re-use a function. Do not change function signature.".format(func))
            return self.func_call

        self.get_function_params_args_call(func, cols_list, extra_args)
        _NumbaDeclareDecorator(self.func_sign, self.return_type)(self.func)
        FunctionJitter.function_cache[self.func.__name__] = (self.func_call, self.func_sign)
        return self.func_call

def _convert_to_vector(args):
    """
    Converts a Python list of strings into an std::vector before passing such
    argument to a Filter operation.
    The purpose of this function is to workaround issue #10092 until there is
    a proper fix in cppyy.
    Arguments:
        args: arguments passed to Filter, possibly including the list of column
    names
    Returns:
        Tuple with arguments, possibly replacing a list of column names by a
    vector
    """

    import ROOT

    if not args or not isinstance(args[0], list):
        return args

    try:
        v = ROOT.std.vector['std::string'](args[0])
    except TypeError:
        raise TypeError(
            f"The list of columns of a Filter operation can only contain strings. Please check: {args[0]}")

    return (v, *args[1:])

def _handle_cpp_callables(func, original_template, *args):
    """
    Checks whether the callable `func` is a cppyy proxy of one of these:
    1. C++ functor
    2. std::function

    The cases above are supported by cppyy, so we can just invoke the original
    cppyy TemplateProxy (Filter or Define) with the callable as argument.

    Prior to the invocation of the original cppyy TemplateProxy, though, we
    need to explicitly instantiate the template using the type of the `func`
    callable. Implicit instantiation won't work since the original
    implementation of Filter and Define is replaced by a pythonization.

    Arguments:
        func: callable to be checked
        original_rdf_method: cppyy proxy for Filter or Define
        args: arguments passed by the user to Filter or Define
    Returns:
        RDataFrame: RDataFrame node if `func` can be classified in one of the
    three cases above, None otherwise
    """

    is_cpp_functor  = lambda : isinstance(getattr(func, '__call__', None), libcppyy.CPPOverload)
    is_std_function = lambda : isinstance(getattr(func, 'target_type', None), libcppyy.CPPOverload)

    if is_cpp_functor() or is_std_function():
        return original_template[type(func)](*args)

def _PyFilter(rdf, callable_or_str, *args, extra_args={}):
    """
    Filters the entries of RDF according to a given condition.
    Arguments:
        callable_or_str: The condition according to which the RDF is to be filtered.
            It can be either a python callable or a c style string.
        *args:
            Can be at most two of them.
            a. List of columns that the callable will receive as argument.
                If not provided then it tries maps the name of the parameter to a column name of the RDF.
            b. The name of the Filter as a string.
        extra_args: non-columnar arguments to be passed to the callable.
    Returns:
        RDataFrame: result of transforming the rdf argument by applying the requested filter operation.

    Examples:
    1. rdf.Filter(lambda x,y: x>y)
        Filter using a python callable.
    2. rdf.Filter(lambda x,y: x>y, ["y", "x"])
        Filter using a python callable with column list passed explicitly.
        When the column list is passed, the function maps the starting input parameters to those columns instead.
        Here the function param x is mapped to column y, and the second function param y is mapped to column x.
        This is equivalent to the filter "y>x".
    3. rdf.Filter(lambda col1, const1: col1>const1, ["x"], extra_args = {"const1":0.5})
        Filter using a python callable with column list passed explicitly and extra_args.
        Here the function parameter col1 is mapped to col "x" of the RDF, and the const1 parameter is mapped to the float value 0.5
        This is equivalent to the filter x>0.5"
    4. y = 0.5
       def x_more_than_y(x):
            x>y
       rdf.Filter(x_more_than_y)
       Here the value of y is captured from scope. Thus this is equivalent to the filter "x>0.5".
       Note: Any modifications to the value of y will not be reflected in the function as y would be treated as a compile time constant during the first jit.


    """
    if isinstance(callable_or_str, str):  # If string argument is passed. Invoke the Original Filters.
        return rdf._OriginalFilter(callable_or_str, *args)
    # The 1st argument is either a string or a python callable.
    if not callable(callable_or_str):
        raise TypeError(
            f"The first argument of a Filter operation should be a callable. {type(callable_or_str).__name__} object is not callable.")

    if len(args) > 2:
        raise TypeError(
            f"Filter takes at most 3 positional arguments but {len(args) + 1} were given")

    func = callable_or_str
    rdf_node = _handle_cpp_callables(func, rdf._OriginalFilter, func, *_convert_to_vector(args))
    if rdf_node is not None:
        return rdf_node
    else:
        raise NotImplementedError(
            f"Passing callables of type {type(func)} will be supported in future versions of ROOT.")

def _PyDefine(rdf, col_name, callable_or_str, cols = [] , extra_args = {} ):
    """
    Defines a new column in the RDataFrame.
    Arguments:
    1. col_name: The name of the new column to be defined
    2. callable_or_str: The definition of the contents of the new columns
        It can be either a python callable or a c style string.
    3. cols: list of columns that the callable will receive as argument.
                If not provided then it tries maps the name of the parameter to a column name of the RDF.
    4. extra_args: non-columnar arguments to be passed to the callable.
    Returns:
        RDataFrame: rdf with new column defined
    
    Examples:
    1. rdf.Define("x", lambda: np.random.rand())
        Define using a python lambda
    2. rdf.Define("x2", lambda x: x*x, ["y"])
        Maps the function argument to column y.
    3. scale = 0.5
        def x_scaled(x):
            return x*scale
       rdf.Define("X_scaled", x_scaled)
       The scale is captured from scope and treated as a compile time constant during first jit.
       Note: Any changes to scale will not be reflected in the jitted function as jitting is a one time operation.
    4. def x_scaled(x, u):
            return x*u
       rdf.Define("x_scaled", x_scaled, extra_args = {"u":0.5})

    """
    if not isinstance(col_name, str):
        raise TypeError(f"First argument of Define must be a valid string for the new column name. {type(col_name).__name__} is not a string.")

    if isinstance(callable_or_str, str): # If string argument is passed. Invoke the Original Define.
        return rdf._OriginalDefine(col_name, callable_or_str)

    if not callable(callable_or_str): # The 2st argument is either a string or a python callable.
        raise TypeError(f"The second argument of a Define operation should be a callable. {type(callable_or_str).__name__} object is not callable.")
    
    if not isinstance(cols, list):        
        raise TypeError(f"Define takes a column list as third arguments but {type(cols).__name__} was given.")
    
    func = callable_or_str
    rdf_node = _handle_cpp_callables(func, rdf._OriginalDefine, col_name, func, cols)
    if rdf_node is not None:
        return rdf_node
    else:
        raise NotImplementedError(
            f"Passing callables of type {type(func)} will be supported in future versions of ROOT.")
