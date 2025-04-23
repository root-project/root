#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TMath.h"
#include "TROOT.h"
#include "TTree.h"
#include "TRandom3.h"
#include <cassert>
#include <iostream>


using FourVector = ROOT::Math::XYZTVector;
using FourVectors = std::vector<FourVector>;
using CylFourVector = ROOT::Math::RhoEtaPhiVector;

void getTracks(FourVectors& tracks) {
   static TRandom3 R(1);
   const double M = 0.13957;  // set pi+ mass
   auto nPart = R.Poisson(5);
   tracks.clear();
   tracks.reserve(nPart);
   for (int i = 0; i < static_cast<int>(nPart); ++i) {
      double px = R.Gaus(0,10);
      double py = R.Gaus(0,10);
      double pt = sqrt(px*px +py*py);
      double eta = R.Uniform(-3,3);
      double phi = R.Uniform(0.0 , 2*TMath::Pi() );
      CylFourVector vcyl( pt, eta, phi);
      // set energy
      double E = sqrt( vcyl.R()*vcyl.R() + M*M);
      FourVector q( vcyl.X(), vcyl.Y(), vcyl.Z(), E);
      // fill track vector
      tracks.emplace_back(q);
   }
}

// A simple helper function to fill a test tree and save it to file
// This makes the example stand-alone
void FillTree(const char* filename, const char* treeName) {
   TFile f(filename,"RECREATE");
   TTree t(treeName,treeName);
   double b1;
   int b2;
   float b3;
   float b4;
   std::vector<FourVector> tracks;
   std::vector<double> dv {-1,2,3,4};
   std::vector<float> sv {-1,2,3,4};
   std::list<int> sl {1,2,3,4};
   t.Branch("a", &b1);
   t.Branch("ab", &b2);
   t.Branch("aba", &b3);
   t.Branch("abab", &b4);
   t.Branch("tracks", &tracks);
   t.Branch("dv", &dv);
   t.Branch("sl", &sl);
   t.Branch("sv", &sv);

   for(int i = 0; i < 20; ++i) {
      b1 = i;
      b2 = i*i;
      b3 = sqrt(i*i*i);
      b4 = i;
      getTracks(tracks);
      dv.emplace_back(i);
      sl.emplace_back(i);
      sv.emplace_back(i * 0.5);
      t.Fill();
   }
   t.Write();
   f.Close();
   return;
}


const char* treeName = "myTree";

void run() {
   // Define data-frame
   ROOT::RDataFrame d(treeName, "test_stringFilterColumn.root");
   auto c1 = d.Count();

   auto dd = d.Filter("true", "ok");
   auto c2 = dd.Count();

   auto ddd = d.Filter("abab > 5 && (&tracks)->size() > 3");
   auto c3 = ddd.Count();

   std::cout << "c1 " << *c1 << std::endl;
   std::cout << "c2 " << *c2 << std::endl;
   std::cout << "c3 " << *c3 << std::endl;

   d.Report()->Print();

   auto c4 = d.Define("tracks_size", "tracks.size()").Filter("tracks_size > 3 && abab > 5","All Filters").Count();
   std::cout << "c4 " << *c4 << std::endl;
   d.Report()->Print();

}

int test_stringfiltercolumn() {

   FillTree("test_stringFilterColumn.root", treeName);

   run();
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif
   run();

   return 0;
}

int main() {
  return test_stringfiltercolumn();
}
