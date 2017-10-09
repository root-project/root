## \file
## \ingroup tutorial_tdataframe
## \notebook
## This tutorial illustrates how use the TDataFrame in combination with a
## TDataSource. In this case we use a TRootDS. This data source allows to read
## a ROOT dataset from a TDataFrame in a different way, not based on the
## regular TDataFrame code. This allows to perform all sorts of consistency
## checks and illustrate the usage of the TDataSource in a didactic context.
##
## \macro_code
##
## \date September 2017
## \author Danilo Piparo

import ROOT

# A simple helper function to fill a test tree: this makes the example stand-alone.
def fill_tree(treeName, fileName):
    tdf = ROOT.ROOT.Experimental.TDataFrame(10000)
    tdf.Define("b1", "(int) tdfentry_").Snapshot(treeName, fileName)


# We prepare an input tree to run on
fileName = "tdf011_rootDataSource_py.root"
treeName = "myTree"
fill_tree(treeName, fileName)

# Create the data frame
MakeRootDataFrame = ROOT.ROOT.Experimental.TDF.MakeRootDataFrame

d = MakeRootDataFrame(treeName, fileName)

# Now we have a regular TDataFrame: the ingestion of data is delegated to
# the TDataSource. At this point everything works as before.

h = d.Define("x", "1./(b1 + 1.)").Histo1D(("h_s", "h_s", 128, 0, .6), "x")

# Now we redo the same with a TDF and we draw the two histograms
c = ROOT.TCanvas()
c.SetLogy()
h.DrawClone()
