/// \file
/// \ingroup tutorial_tdataframe
/// \notebook -nodraw
/// This tutorial illustrates how to use TProfiles in combination with the
/// TDataFrame. See the documentation of TProfile and TProfile2D to better
/// understand the analogy of this code with the example one.
///
/// \macro_code
///
/// \date February 2017
/// \author Danilo Piparo

#include "TFile.h"
#include "TH1F.h"
#include "TRandom.h"
#include "TTree.h"

#include "ROOT/TDataFrame.hxx"

// A simple helper function to fill a test tree: this makes the example
// stand-alone.
void fill_tree(const char *filename, const char *treeName)
{
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   float px, py, pz;
   t.Branch("px", &px);
   t.Branch("py", &py);
   t.Branch("pz", &pz);
   for (int i = 0; i < 25000; i++) {
      gRandom->Rannor(px, py);
      pz = px * px + py * py;
      t.Fill();
   }
   t.Write();
   f.Close();
   return;
}

void tdf003_profiles()
{
   // We prepare an input tree to run on
   auto fileName = "tdf003_profiles.root";
   auto treeName = "myTree";
   fill_tree(fileName, treeName);

   // We read the tree from the file and create a TDataFrame.
   ROOT::Experimental::TDataFrame d(treeName, fileName, {"px", "py", "pz"});

   // Create the profiles
   auto hprof1d = d.Profile1D(TProfile("hprof1d", "Profile of pz versus px", 64, -4, 4));
   auto hprof2d = d.Profile2D(TProfile2D("hprof2d", "Profile of pz versus px and py", 40, -4, 4, 40, -4, 4, 0, 20));

   // And Draw
   auto c1 = new TCanvas("c1", "Profile histogram example", 200, 10, 700, 500);
   hprof1d->DrawClone();
   auto c2 = new TCanvas("c2", "Profile2D histogram example", 200, 10, 700, 500);
   c2->cd();
   hprof2d->DrawClone();
}
