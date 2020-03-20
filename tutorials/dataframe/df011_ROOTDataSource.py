## \file
## \ingroup tutorial_dataframe
## \notebook -draw
## This tutorial illustrates how use the RDataFrame in combination with a
## RDataSource. In this case we use a TRootDS. This data source allows to read
## a ROOT dataset from a RDataFrame in a different way, not based on the
## regular RDataFrame code. This allows to perform all sorts of consistency
## checks and illustrate the usage of the RDataSource in a didactic context.
##
## \macro_code
## \macro_image
##
## \date September 2017
## \author Danilo Piparo

import ROOT

# A simple helper function to fill a test tree: this makes the example stand-alone.
def fill_tree(treeName, fileName):
    df = ROOT.RDataFrame(10000)
    df.Define("b1", "(int) rdfentry_").Snapshot(treeName, fileName)


# We prepare an input tree to run on
fileName = "df011_rootDataSource_py.root"
treeName = "myTree"
fill_tree(treeName, fileName)

# Create the data frame
MakeRootDataFrame = ROOT.RDF.MakeRootDataFrame

d = MakeRootDataFrame(treeName, fileName)

# Now we have a regular RDataFrame: the ingestion of data is delegated to
# the RDataSource. At this point everything works as before.

h = d.Define("x", "1./(b1 + 1.)").Histo1D(("h_s", "h_s", 128, 0, .6), "x")

# Now we redo the same with a RDF and we draw the two histograms
c = ROOT.TCanvas()
c.SetLogy()
h.DrawClone()
