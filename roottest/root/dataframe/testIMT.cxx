#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TMath.h"
#include "TTree.h"
#include "TRandom3.h"
#include "TROOT.h"
#include "TSystem.h"
#include <cassert>
#include <iostream>
#include <chrono>
#include <thread>

// A simple class to measure time.
class TimerRAII{
   using timePoint = std::chrono::high_resolution_clock::time_point;
public:
   TimerRAII():fStart(std::chrono::high_resolution_clock::now()){};
   ~TimerRAII(){
      std::chrono::duration<double> deltaT=std::chrono::high_resolution_clock::now()-fStart;
      std::cout << "\nElapsed time: " << deltaT.count() << "s\n";
   }
private:
   const timePoint fStart;
};


using FourVector = ROOT::Math::XYZTVector;
using FourVectors = std::vector<FourVector>;
using CylFourVector = ROOT::Math::RhoEtaPhiVector;

void getTracks(unsigned int mu, FourVectors& tracks) {
   static TRandom R(1);
   const double M = 0.13957;  // set pi+ mass
   auto nPart = R.Poisson(mu);
   tracks.clear();
   tracks.reserve(nPart);
   for (size_t i = 0; i < nPart; ++i) {
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
   if (!gSystem->AccessPathName(filename)) return;
   TFile f(filename,"RECREATE");
   TTree t(treeName,treeName);
   double b1;
   int b2;
   std::vector<FourVector> tracks;
   std::vector<double> dv {-1,2,3,4};
   std::list<int> sl {1,2,3,4};
   t.Branch("b1", &b1);
   t.Branch("b2", &b2);
   t.Branch("tracks", &tracks);
   t.Branch("dv", &dv);
   t.Branch("sl", &sl);

   int nevts = 16000;
   for(int i = 0; i < nevts; ++i) {
      b1 = i;
      b2 = i*i;

      getTracks(1, tracks);

      dv.emplace_back(i);
      sl.emplace_back(i);
      t.Fill();
   }
   t.Write();
   f.Close();
   return;
}

auto fileName = "testIMT.root";
auto treeName = "myTree";

void tests() {


   std::cout << "Parallelism check" << std::endl;
   {
      ROOT::RDataFrame d(treeName, fileName);
      auto sleep = [] () { std::this_thread::sleep_for(std::chrono::microseconds(1)); return true; };
      auto sleepCount = d.Filter(sleep).Count();
      *sleepCount;
   }

   std::cout << "Simple filtering" << std::endl;
   {
      ROOT::RDataFrame d(treeName, fileName);
      auto ok = []() { return true; };
      auto ko = []() { return false; };
      auto cok = d.Filter(ok).Count();
      auto cko = d.Filter(ko).Count();
      std::cout << "Count ok " << *cok << std::endl;
      std::cout << "Count ko " << *cko << std::endl;
   }

   std::cout << "\nAdding branch and filter" << std::endl;
   {
      ROOT::RDataFrame d(treeName, fileName);

      auto r = d.Define("iseven", [](int b2) { return b2 % 2 == 0; }, {"b2"})
                .Filter([](bool iseven) { return iseven; }, {"iseven"})
                .Count();
      std::cout << "Count filter on added branch " << *r << std::endl;
   }

   std::cout << "\nGetting the mean, min and the max" << std::endl;
   {
      ROOT::RDataFrame d(treeName, fileName);

      auto min = d.Min("b2");
      auto max = d.Max("b2");
      auto mean = d.Mean("b2");

      std::cout << "Mean of b2 " << *mean << std::endl;
      std::cout << "Min of b2 " << *min << std::endl;
      std::cout << "Max of b2 " << *max << std::endl;
   }

   std::cout << "\nAdd branch, filter, getting the min and the max" << std::endl;
   {
      ROOT::RDataFrame d(treeName, fileName);

      auto fd = d.Define("iseven", [](int b2) { return b2 % 2 == 0; }, {"b2"})
                .Filter([](bool iseven) { return iseven; }, {"iseven"});

      auto min = d.Min("b1");
      auto max = d.Max("b1");

      std::cout << "After filter, min of b1 " << *min << std::endl;
      std::cout << "After filter, max of b1 " << *max << std::endl;
   }

   std::cout << "Histo1D" << std::endl;
   {
      ROOT::RDataFrame d(treeName, fileName);

      auto h = d.Histo1D("b1");

      std::cout << "Histo1D b1 entries and mean " << h->GetEntries() << " " << h->GetMean() << std::endl;
   }

   std::cout << "Histo1D with filter and new branch" << std::endl;
   {

      auto getPt = [](const FourVectors& tracks) {
         std::vector<double> pts;
         pts.reserve(tracks.size());
         for (auto& t:tracks)
            pts.emplace_back(t.Pt());
         return pts;
         };

      ROOT::RDataFrame d(treeName, fileName, {"tracks"});

      auto ad = d.Define("tracks_n", [](const FourVectors& tracks){return (int)tracks.size();})
                 .Filter([](int tracks_n){return tracks_n > 2;}, {"tracks_n"})
                 .Define("tracks_pts", getPt);
      auto trN = ad.Histo1D(TH1D{"", "", 100, -.5, 99.5}, "tracks_n");
      auto trPts = ad.Histo1D("tracks_pts");

      std::cout << "Histo1D tracks number entries and mean " << trN->GetEntries()
                << " " << trN->GetMean() << std::endl;
      std::cout << "Histo1D track Pts entries and mean " << trN->GetEntries()
                << " " << trPts->GetMean() << std::endl;
   }

   std::cout << "\nGetting a column as list" << std::endl;
   {
      ROOT::RDataFrame d(treeName, fileName);

      auto double_list = d.Take<double/*, std::vector<double>*/>("b1");
      std::cout << "Get: size of list<double> " << double_list->size() << std::endl;
   }

   std::cout << "\nGetting a column as vector" << std::endl;
   {
      ROOT::RDataFrame d(treeName, fileName);

      auto double_list = d.Take<double, std::vector<double>>("b1");
      std::cout << "Get: size of list<double> " << double_list->size() << std::endl;
   }

}

void testIMT(int argc = 1, char **argv = nullptr)
{
   // Prepare an input tree to run on
   FillTree(fileName, treeName);

   std::cout << "Running sequentially." << std::endl;
   {
      tests();
   }

   unsigned int ncores = 4;
   if (argc > 1)
      ncores = std::atoi(argv[1]);
   if (ncores == 0)
      return;
   std::cout << "\b\b***** Parallelism enabled. Running with " << ncores << "!" << std::endl;
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT(ncores);
#endif
   {
      tests();
   }
}

int main(int argc = 1, char **argv = nullptr)
{
   if (argc > 1)
      testIMT(argc, argv);
   else
      testIMT(1, nullptr);
   return 0;
}
