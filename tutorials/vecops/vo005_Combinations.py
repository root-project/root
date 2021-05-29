## \file
## \ingroup tutorial_vecops
## \notebook -nodraw
## In this tutorial we learn how combinations of RVecs can be built.
##
## \macro_code
## \macro_output
##
## \date August 2018
## \author Stefan Wunsch

import ROOT
from ROOT.VecOps import RVec, Take, Combinations

# RVec can be sorted in Python with the inbuilt sorting function because
# PyROOT implements a Python iterator
v1 = RVec("double")(3)
v1[0], v1[1], v1[2] = 1, 2, 3
v2 = RVec("double")(2)
v2[0], v2[1] = -4, -5

# To get the indices, which result in all combinations, you can call the
# following helper.
# Note that you can also pass the size of the vectors directly.
idx = Combinations(v1, v2)

# Next, the respective elements can be taken via the computed indices.
c1 = Take(v1, idx[0])
c2 = Take(v2, idx[1])

# Finally, you can perform any set of operations conveniently.
v3 = c1 * c2

print("Combinations of {} and {}:".format(v1, v2))
for i in range(len(v3)):
    print("{} * {} = {}".format(c1[i], c2[i], v3[i]))
print

# However, if you want to compute operations on unique combinations of a
# single RVec, you can perform this as follows.

# Get the indices of unique triples for the given vector.
v4 = RVec("double")(4)
v4[0], v4[1], v4[2], v4[3] = 1, 2, 3, 4
idx2 = Combinations(v4, 3)

# Take the elements and compute any operation on the returned collections.
c3 = Take(v4, idx2[0])
c4 = Take(v4, idx2[1])
c5 = Take(v4, idx2[2])

v5 = c3 * c4 * c5

print("Unique triples of {}:".format(v4))
for i in range(len(v5)):
    print("{} * {} * {} = {}".format(c3[i], c4[i], c5[i], v5[i]))
