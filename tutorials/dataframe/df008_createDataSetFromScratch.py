## \file
## \ingroup tutorial_dataframe
## \notebook -nodraw
## This tutorial shows how to create a dataset from scratch with RDataFrame
##
## \macro_code
##
## \date June 2017
## \author Danilo Piparo

import ROOT

# We create an empty data frame of 100 entries
tdf = ROOT.ROOT.RDataFrame(100)

# We now fill it with random numbers
ROOT.gRandom.SetSeed(1)
tdf_1 = tdf.Define("rnd", "gRandom->Gaus()")

# And we write out the dataset on disk
tdf_1.Snapshot("randomNumbers", "df008_createDataSetFromScratch_py.root")

