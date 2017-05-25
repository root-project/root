## \file
## \ingroup tutorial_tdataframe
## This tutorial shows how to get information about the efficiency of the filters
## applied
##
## \macro_code
##
## \date May 2017
## \author Danilo Piparo

import ROOT

fill_tree_code ='''
using FourVector = ROOT::Math::XYZTVector;
using FourVectors = std::vector<FourVector>;
using CylFourVector = ROOT::Math::RhoEtaPhiVector;
void fill_tree(const char *filename, const char *treeName)
{
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   double b1;
   int b2;
   t.Branch("b1", &b1);
   t.Branch("b2", &b2);
   for (int i = 0; i < 50; ++i) {
      b1 = i;
      b2 = i * i;
      t.Fill();
   }
   t.Write();
   f.Close();
   return;
}
'''

# We prepare an input tree to run on
fileName = 'tdf004_cutFlowReport_py.root'
treeName = 'myTree'
ROOT.gInterpreter.Declare(fill_tree_code)
ROOT.fill_tree(fileName, treeName)

# We read the tree from the file and create a TDataFrame, a class that
# allows us to interact with the data contained in the tree.
TDF = ROOT.ROOT.Experimental.TDataFrame
d = TDF(treeName, fileName)

# ## Define cuts and create the report
# An optional string parameter name can be passed to the Filter method to create a named filter.
# Named filters work as usual, but also keep track of how many entries they accept and reject.
filtered1 = d.Filter('b1 > 25', 'Cut1')
filtered2 = d.Filter('0 == b2 % 2', 'Cut2')

augmented1 = filtered2.Define('b3', 'b1 / b2')
filtered3 = augmented1.Filter('b3 < .5','Cut3')

# Statistics are retrieved through a call to the Report method:
# when Report is called on the main TDataFrame object, it prints stats for all named filters declared up to that
# point when called on a stored chain state (i.e. a chain/graph node), it prints stats for all named filters in the
# section of the chain between the main TDataFrame and that node (included).
# Stats are printed in the same order as named filters have been added to the graph, and refer to the latest
# event-loop that has been run using the relevant TDataFrame.
print('Cut3 stats:')
filtered3.Report()
print('All stats:')
d.Report()
