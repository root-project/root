## \file
## \ingroup tutorial_pyroot
## \notebook -nodraw
## This tutorial illustrates the pretty printing feature of PyROOT, which reveals
## the content of the object if a string representation is requested, e.g., by
## Python's print statement. The printing behaves similar to the ROOT prompt
## powered by the C++ interpreter cling.
##
## \macro_code
## \macro_output
##
## \date June 2018
## \author Stefan Wunsch, Enric Tejedor

import ROOT

# Create an object with PyROOT
obj = ROOT.std.vector("int")(3)
for i in range(obj.size()):
    obj[i] = i

# Print the object, which reveals the content. Note that `print` calls the special
# method `__str__` of the object internally.
print(obj)

# The output can be retrieved as string by any function that triggers the `__str__`
# special method of the object, e.g., `str` or `format`.
print(str(obj))
print("{}".format(obj))

# Note that the interactive Python prompt does not call `__str__`, it calls
# `__repr__`, which implements a formal and unique string representation of
# the object.
print(repr(obj))
obj

# The print output behaves similar to the ROOT prompt, e.g., here for a ROOT histogram.
hist = ROOT.TH1F("name", "title", 10, 0, 1)
print(hist)

# If cling cannot produce any nice representation for the class, we fall back to a
# "<ClassName at address>" format, which is what `__repr__` returns
ROOT.gInterpreter.Declare('class MyClass {};')
m = ROOT.MyClass()
print(m)
print(str(m) == repr(m))

