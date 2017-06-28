/// \file
/// \ingroup tutorial_tdataframe
/// \notebook -draw
/// This tutorial shows the possibility to use data models which are more
/// complex than flat ntuples with TDataFrame
///
/// \macro_code
///
/// \date December 2016
/// \author Danilo Piparo

// ## Preparation
// This notebook can be compiled with this invocation
// `g++ -o tdf002_dataModel tdf002_dataModel.C `root-config --cflags --libs` -lTreePlayer`

#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TRandom3.h"
#include "TFile.h"
#include "TH1D.h"
#include "TTree.h"

#include "ROOT/TDataFrame.hxx"

using FourVector = ROOT::Math::XYZTVector;
using FourVectors = std::vector<FourVector>;
using CylFourVector = ROOT::Math::RhoEtaPhiVector;

// A simple helper function to fill a test tree: this makes the example
// stand-alone.
void fill_tree(const char *filename, const char *treeName)
{
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   FourVectors tracks;
   t.Branch("tracks", &tracks);

   const double M = 0.13957; // set pi+ mass
   TRandom3 R(1);

   for (int i = 0; i < 50; ++i) {
      auto nPart = R.Poisson(15);
      tracks.clear();
      tracks.reserve(nPart);
      for (int j = 0; j < nPart; ++j) {
         double px = R.Gaus(0, 10);
         double py = R.Gaus(0, 10);
         double pt = sqrt(px * px + py * py);
         double eta = R.Uniform(-3, 3);
         double phi = R.Uniform(0.0, 2 * TMath::Pi());
         CylFourVector vcyl(pt, eta, phi);
         // set energy
         double E = sqrt(vcyl.R() * vcyl.R() + M * M);
         FourVector q(vcyl.X(), vcyl.Y(), vcyl.Z(), E);
         // fill track vector
         tracks.emplace_back(q);
      }
      t.Fill();
   }

   t.Write();
   f.Close();
   return;
}

int tdf002_dataModel()
{

   // We prepare an input tree to run on
   auto fileName = "tdf002_dataModel.root";
   auto treeName = "myTree";
   fill_tree(fileName, treeName);

   // We read the tree from the file and create a TDataFrame, a class that
   // allows us to interact with the data contained in the tree.
   ROOT::Experimental::TDataFrame d(treeName, fileName, {"tracks"});

   // ## Operating on branches which are collection of objects
   // Here we deal with the simplest of the cuts: we decide to accept the event
   // only if the number of tracks is greater than 5.
   auto n_cut = [](const FourVectors &tracks) { return tracks.size() > 8; };
   auto nentries = d.Filter(n_cut, {"tracks"}).Count();

   std::cout << *nentries << " passed all filters" << std::endl;

   // Another possibility consists in creating a new column containing the
   // quantity we are interested in.
   // In this example, we will cut on the number of tracks and plot their
   // transverse momentum.
   auto getPt = [](const FourVectors &tracks) {
      std::vector<double> pts;
      pts.reserve(tracks.size());
      for (auto &t : tracks) pts.emplace_back(t.Pt());
      return pts;
   };

   // We do the same for the weights.
   auto getPtWeights = [](const FourVectors &tracks) {
      std::vector<double> ptsw;
      ptsw.reserve(tracks.size());
      for (auto &t : tracks) ptsw.emplace_back(1. / t.Pt());
      return ptsw;
   };

   auto augmented_d = d.Define("tracks_n", [](const FourVectors &tracks) { return (int)tracks.size(); })
                         .Filter([](int tracks_n) { return tracks_n > 2; }, {"tracks_n"})
                         .Define("tracks_pts", getPt)
                         .Define("tracks_pts_weights", getPtWeights);

   auto trN = augmented_d.Histo1D(TH1D{"", "", 40, -.5, 39.5}, "tracks_n");
   auto trPts = augmented_d.Histo1D("tracks_pts");
   auto trWPts = augmented_d.Histo1D("tracks_pts", "tracks_pts_weights");

   TCanvas c1;
   trN->Draw();
   c1.Print("tracks_n.png");

   TCanvas c2;
   trPts->Draw();
   c2.Print("tracks_pt.png");

   TCanvas c3;
   trWPts->Draw();
   c3.Print("tracks_Wpt.png");

   return 0;
}

int main()
{
   return tdf002_dataModel();
}
