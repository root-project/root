/// \file
/// \ingroup tutorial_tdataframe
/// This tutorial shows how to get information about the efficiency of the filters
/// applied
///
/// \macro_code
///
/// \date December 2016
/// \author Danilo Piparo

using FourVector = ROOT::Math::XYZTVector;
using FourVectors = std::vector<FourVector>;
using CylFourVector = ROOT::Math::RhoEtaPhiVector;

// A simple helper function to fill a test tree: this makes the example
// stand-alone.
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

void tdf004_cutFlowReport()
{

   // We prepare an input tree to run on
   auto fileName = "tdf004_cutFlowReport.root";
   auto treeName = "myTree";
   fill_tree(fileName, treeName);

   // We read the tree from the file and create a TDataFrame
   ROOT::Experimental::TDataFrame d(treeName, fileName, {"b1", "b2"});

   // ## Define cuts and create the report
   // Here we define two simple cuts
   auto cut1 = [](double b1) { return b1 > 25.; };
   auto cut2 = [](int b2) { return 0 == b2 % 2; };

   // An optional string parameter name can be passed to the Filter method to create a named filter.
   // Named filters work as usual, but also keep track of how many entries they accept and reject.
   auto filtered1 = d.Filter(cut1, {"b1"}, "Cut1");
   auto filtered2 = d.Filter(cut2, {"b2"}, "Cut2");

   auto augmented1 = filtered2.Define("b3", [](double b1, int b2) { return b1 / b2; });
   auto cut3 = [](double x) { return x < .5; };
   auto filtered3 = augmented1.Filter(cut3, {"b3"}, "Cut3");

   // Statistics are retrieved through a call to the Report method:
   // when Report is called on the main TDataFrame object, it prints stats for all named filters declared up to that
   // point
   // when called on a stored chain state (i.e. a chain/graph node), it prints stats for all named filters in the
   // section
   // of the chain between the main TDataFrame and that node (included).
   // Stats are printed in the same order as named filters have been added to the graph, and refer to the latest
   // event-loop that has been run using the relevant TDataFrame.
   std::cout << "Cut3 stats:" << std::endl;
   filtered3.Report();
   std::cout << "All stats:" << std::endl;
   d.Report();
}
