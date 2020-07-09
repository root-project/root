## \file
## \ingroup tutorial_dataframe
## \notebook -draw
## \brief Fill a TGraph using RDataFrame.
##
## \macro_code
## \macro_image
##
## \date July 2018
## \authors Enrico Guiraud, Danilo Piparo, Massimo Tumolo

import ROOT

ROOT.ROOT.EnableImplicitMT(2)
d = ROOT.RDataFrame(160)

# Create a trivial parabola
dd = d.Alias("x", "rdfentry_").Define("y", "x*x")

graph = dd.Graph("x", "y")

# This tutorial is ran with multithreading enabled. The order in which points are inserted is not known, so to have a meaningful representation points are sorted.
graph.Sort()
graph.Draw("APL")

