## \file
## \ingroup tutorial_vecops
## \notebook -nodraw
## In this tutorial we learn how elements of an RVec can be easily sorted and
## selected.
##
## \macro_code
## \macro_output
##
## \date August 2018
## \author Stefan Wunsch

import ROOT
from ROOT.VecOps import RVec, Argsort, Take, Sort, Reverse

# RVec can be sorted in Python with the inbuilt sorting function because
# PyROOT implements a Python iterator
v1 = RVec("double")(3)
v1[0], v1[1], v1[2] = 6, 4, 5
v2 = sorted(v1)
print("Sort vector {}: {}".format(v1, v2))

# For convenience, ROOT implements helpers, e.g., to get a sorted copy of
# an RVec ...
v2 = Sort(v1);
print("Sort vector {}: {}".format(v1, v2))

# ... or a reversed copy of an RVec.
v2 = Reverse(v1);
print("Reverse vector {}: {}".format(v1, v2))

# Helpers are provided to get the indices that sort the vector and to
# select these indices from an RVec.
v2 = Argsort(v1)
print("Indices that sort the vector {}: {}".format(v1, v2))

v3 = RVec("double")(3)
v3[0], v3[1], v3[2] = 9, 7, 8
v4 = Take(v3, v2)
print("Sort vector {} respective to the previously determined indices: {}".format(v3, v4))

# Take can also be used to get the first or last elements of an RVec.
v2 = Take(v1, 2)
v3 = Take(v1, -2)
print("Take the two first and last elements of vector {}: {}, {}".format(v1, v2, v3))

# Because the VecOps helpers return a copy of the input, you can chain the operations
# conveniently.
v2 = Reverse(Take(Sort(v1), -2))
print("Sort the vector {}, take the two last elements and reverse the selection: {}".format(v1, v2))
