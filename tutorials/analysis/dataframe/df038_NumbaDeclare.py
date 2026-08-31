## \file
## \ingroup tutorial_dataframe
## \notebook -nodraw
## This tutorial illustrates how PyROOT supports declaring C++ callables from
## Python callables making them, for example, usable with RDataFrame. The Python
## function is translated to C++ source and declared to the interpreter, so the
## result is an ordinary C++ function that is inlined into the event loop.
## Fundamental types are supported, along with ROOT::RVec, std::vector and
## std::array of them, and any C++ class the interpreter already knows.
##
## \macro_code
## \macro_output
##
## \date March 2020
## \author Stefan Wunsch

import ROOT

# To mark a Python callable to be used from C++, you have to use the decorator
# provided by PyROOT passing the C++ types of the input arguments and the return
# value. The generated C++ is available afterwards as pypow.__cpp_wrapper__.
@ROOT.Numba.Declare(['float', 'int'], 'float')
def pypow(x, y):
    return x**y

# The decorator is also available as ROOT.Py.Declare, which is the same thing
# declared into the Py C++ namespace; ROOT.Numba.Declare predates the C++
# translation and is kept so that existing code keeps working.

# The Python callable is now available from C++ in the Numba namespace.
# For example, we can use it from the interpreter.
ROOT.gInterpreter.ProcessLine('cout << "2^3 = " << Numba::pypow(2, 3) << endl;')

# Or we can use the callable as well within a RDataFrame workflow.
data = ROOT.RDataFrame(4).Define('x', '(float)rdfentry_')\
                         .Define('x_pow3', 'Numba::pypow(x, 3)')\
                         .AsNumpy()

print('pypow({}, 3) = {}'.format(data['x'], data['x_pow3']))

# The supported input and return types are the fundamental types and the
# containers of them: ROOT::RVec, std::vector and std::array. See the following
# callable computing the power of the elements in an array.
@ROOT.Numba.Declare(['RVecF', 'int'], 'RVecF')
def pypowarray(x, y):
    return x**y

ROOT.gInterpreter.ProcessLine('''
ROOT::RVecF x = {0, 1, 2, 3};
cout << "pypowarray(" << x << ", 3) =  " << Numba::pypowarray(x, 3) << endl;
''')

# and now with RDataFrame
s = ROOT.RDataFrame(1).Define('x', 'ROOT::RVecF{1,2,3}')\
                      .Define('x2', 'Numba::pypowarray(x, 2)')\
                      .Sum('x2') # 1 + 4 + 9 == 14
print('sum(pypowarray({ 1, 2, 3 }, 2)) = ', s.GetValue())
