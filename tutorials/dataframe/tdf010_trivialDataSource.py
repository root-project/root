## \file
## \ingroup tutorial_tdataframe
## \notebook
## This tutorial illustrates how use the TDataFrame in combination with a
## TDataSource. In this case we use a TTrivialDS, which is nothing more
## than a simple generator: it does not interface to any existing dataset.
## The TTrivialDS has a single column, col0, which has value n for entry n.
##
## \macro_code
##
## \date September 2017
## \author Danilo Piparo

import ROOT

# Create the data frame
MakeTrivialDataFrame = ROOT.ROOT.Experimental.TDF.MakeTrivialDataFrame

nEvents = 128

d_s = MakeTrivialDataFrame(nEvents)

# Now we have a regular TDataFrame: the ingestion of data is delegated to
# the TDataSource. At this point everything works as before.
h_s = d_s.Define("x", "1./(1. + col0)").Histo1D(("h_s", "h_s", 128, 0, .6), "x")

c = ROOT.TCanvas()
c.SetLogy()
h_s.Draw()

