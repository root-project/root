## \file
## \ingroup tutorial_dataframe
## Simple RDataFrame example in Python.
##
## This tutorial shows a minimal example of RDataFrame. It starts without input
## data, generates a new column `x` with random numbers, and finally draws
## a histogram for `x`.
##
## \macro_code
## \macro_output
##
## \date September 2021
## \author Enric Tejedor (CERN)

import ROOT

# Create a data frame with 100 rows
rdf = ROOT.RDataFrame(100)

# Define a new column `x` that contains random numbers
rdf_x = rdf.Define("x", "gRandom->Rndm()")

# Create a histogram from `x` and draw it
h = rdf_x.Histo1D("x")
h.Draw()
