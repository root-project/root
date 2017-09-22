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
fill_tree_code = '''
void fill_tree(const char *fileName, const char *treeName)
{
   TFile f(fileName, "RECREATE");
   TTree t(treeName, treeName);
   int b1;
   t.Branch("b1", &b1);
   for (int i = 0; i < 10000; ++i) {
      b1 = i;
      t.Fill();
   }
   t.Write();
   f.Close();
   return;
}
'''
# We prepare an input tree to run on
fileName = "tdf011_rootDataSource_py.root"
treeName = "myTree"
ROOT.gInterpreter.Declare(fill_tree_code)
ROOT.fill_tree(fileName, treeName)

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
