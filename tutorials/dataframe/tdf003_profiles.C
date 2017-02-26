/// \file
/// \ingroup tutorial_tdataframe
/// \notebook -nodraw
/// This tutorial illustrates how to use TProfiles in combination with the
/// TDataFrame. See the documentation of TProfile to better understand the
/// analogy of this code with the example one.
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
void fill_tree(const char* filename, const char* treeName)
{
   TFile f(filename,"RECREATE");
   TTree t(treeName,treeName);
   float px, py, pz;
   t.Branch("px", &px);
   t.Branch("pz", &pz);
   for (int i=0; i<25000; i++) {
      gRandom->Rannor(px,py);
      pz = px*px + py*py;
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
   fill_tree(fileName,treeName);

   // We read the tree from the file and create a TDataFrame.
   TFile f(fileName);
   ROOT::Experimental::TDataFrame d(treeName, &f, {"px","pz"});

   // Create the profile
   auto hprof = d.Profile1D<float, float>(TProfile("hprof","Profile of pz versus px",64,-4,4));

   // And Draw
   auto c1 = new TCanvas("c1","Profile histogram example",200,10,700,500);
   hprof->DrawClone();
}
