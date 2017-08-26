/// \file
/// \ingroup tutorial_tdataframe
/// \notebook -nodraw
/// This tutorial illustrates how simpler it can be to use a
/// TDataFrame to create a dataset with respect to the usage
/// of the TTree interfaces.
///
/// \macro_code
///
/// \date August 2017
/// \author Danilo Piparo

// ## Preparation

void tdf009_FromScratchVSTTree() {

   // ## Preparation
   auto treeName = "myTree";

   // ##This is the classic way
   TFile f("tdf009_FromScratchVSTTree_classic.root", "RECREATE");
   TTree t(treeName, treeName);
   double b1;
   int b2;
   t.Branch("b1", &b1);
   t.Branch("b2", &b2);
   for (int i = 0; i < 10; ++i) {
      b1 = i;
      b2 = i * i;
      t.Fill();
   }
   t.Write();
   f.Close();

   // ## This is the TDataFrame way
   // Few lines are needed to achieve the same result.
   // Parallel creation of the TTree is not supported in the
   // classic method.
   ROOT::Experimental::TDataFrame tdf(10);
   auto b = 0.;
   tdf.Define("b1",[&b](){return b++;})
      .Define("b2","(int) b1 * b1") // This can even be a string
      .Snapshot(treeName, "tdf009_FromScratchVSTTree_tdf.root");

}