## \file
## \ingroup tutorial_pyroot
## \notebook -nodraw
## This tutorial illustrates how PyROOT supports declaring C++ callables from
## Python callables making them, for example, usable with RDataFrame.
##
## \macro_code
## \macro_output
##
## \date March 2020
## \author Stefan Wunsch

import ROOT

# To mark a Python callable to be used from C++, you have to use the decorator
# provided by PyROOT passing the C++ types of the input arguments and the return
# value.
@ROOT.DeclareCppCallable(["float", "int"], "float")
def pypow(x, y):
    return x**y

# The Python callable is now available from C++ in the CppCallable namespace.
# For example, we can use it from the interpreter.
ROOT.gInterpreter.ProcessLine('cout << "2^3 = " << CppCallable::pypow(2, 3) << endl;')

# Or we can use the callable as well within a RDataFrame workflow.
data = ROOT.RDataFrame(4).Define("x", "(float)rdfentry_")\
                         .Define("xpow2", "CppCallable::pypow(x, 2)")\
                         .AsNumpy()

for col in sorted(data):
    print("{}: {}".format(col, data[col]))

# ROOT uses one of two methods to create C++ functions from python ones. The more
# performant one is based on the numba Python package (http://numba.pydata.org/). With
# numba you can expect a runtime close to a C++ implementation. To request throwing
# an error in case the python callable cannot be processed by numba, you can set
# the argument numba_only to True.
@ROOT.DeclareCppCallable(["float"], "float", numba_only=True)
def numba_callable(x):
    return 2.0 * x

# For more complex python callables, e.g. ones that process complex (C++) objects,
# ROOT can fall back to a generic, although much less performant, implementation.
# You will be warned at runtime that the callable could not be compiled using numba.
@ROOT.DeclareCppCallable(["vector<float>"], "int")
def generic_callable(x):
    y = 0.0
    for v in x:
        y += v
    return y

ROOT.gInterpreter.ProcessLine("""
vector<float> x = {1, 2, 3};
cout << "sum({1, 2, 3}) =  " << CppCallable::generic_callable(x) << endl;
""")
