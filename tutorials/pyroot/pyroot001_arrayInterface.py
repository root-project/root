## \file
## \ingroup tutorial_pyroot
## \notebook -nodraw
## This tutorial illustrates the conversion of STL vectors and TVec to numpy
## arrays without copying the data.
## The memory-adoption is achieved by the dictionary __array_interface__, which
## is added dynamically to the Python objects by PyROOT.
##
## \macro_code
## \macro_output
##
## \date April 2018
## \author Stefan Wunsch

import ROOT
from sys import exit

try:
    import numpy as np
except:
    exit()

# Create a vector ROOT object and assign values
# Note that this works as well with a TVec
vec = ROOT.std.vector("float")(2)
vec[0] = 1
vec[1] = 2
print("Content of the ROOT vector object: {}".format([x for x in vec]))

# Interface ROOT vector with a numpy array
array = np.asarray(vec)
print("Content of the associated numpy array: {}".format([x for x in array]))

# The numpy array adopts the memory of the vector without copying the content.
# Note that the first entry of the numpy array changes when assigning a new
# value to the first entry of the ROOT vector.
vec[0] = 42
print(
    "Content of the numpy array after changing the first entry of the ROOT vector: {}".
    format([x for x in array]))

# Use numpy features on data of ROOT objects
print("Mean of the numpy array entries: {}".format(np.mean(array)))
