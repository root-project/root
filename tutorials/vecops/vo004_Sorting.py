## \file
## \ingroup tutorial_vecops
## \notebook -nodraw
## In this tutorial we learn how an RVec can be sorted efficiently.
##
## \macro_code
##
## \date July 2018
## \author Stefan Wunsch

import ROOT
from ROOT.VecOps import RVec, Argsort, Take

# RVec can be sorted in Python with the inbuild sorting function
v1 = RVec("double")(3)
v1[0], v1[1], v1[2] = 6, 4, 5
v2 = v1
v2 = sorted(v2)
print("Sorting of vector {}: {}".format(v1, v2))

## Additionally, ROOT provides helpers to get the indices that sort the
## vector and to select these indices from an RVec.
i = Argsort(v1)
print("Indices that sort the vector {}: {}".format(v1, i))

v3 = RVec("double")(3)
v3[0], v3[1], v3[2] = 9, 7, 8
v4 = Take(v3, i)
print("Sorting of the vector {} respective to the previously ".format(v3) +
      "determined indices: {}".format(v4))
