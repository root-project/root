#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TMath.h"
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
   t.Branch("b1", &b1);
   t.Branch("b2", &b2);
   t.Branch("b3", &b3);
   t.Branch("b4", &b4);
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

// check that value has both same value and same type as ref
template<class T>
void CheckRes(const T& v, const T& ref, const char* msg) {
   if (v!=ref) {
      std::cerr << "***FAILED*** " << msg << std::endl;
   }
}

void test_misc() {
   // Prepare an input tree to run on
   auto fileName = "test_misc.root";
   auto treeName = "myTree";
   FillTree(fileName,treeName);

   // Define data-frame
   ROOT::RDataFrame d(treeName, fileName);
   // ...and two dummy filters
   auto ok = []() { return true; };
   auto ko = []() { return false; };

   // TEST 1: no-op filter and Run
   d.Filter(ok).Foreach([](double x) { std::cout << x << std::endl; }, {"b1"});

   // TEST 2: Forked actions
   // always apply first filter before doing three different actions
   auto dd = d.Filter(ok);
   dd.Foreach([](double x) { std::cout << x << " "; }, {"b1"});
   dd.Foreach([](int y) { std::cout << y << std::endl; }, {"b2"});
   auto c = dd.Count();
   // ... and another filter-and-foreach
   auto ddd = dd.Filter(ko);
   ddd.Foreach([]() { std::cout << "ERROR" << std::endl; });
   auto cv = *c;
   std::cout << "c " << cv << std::endl;
   CheckRes(cv,20ULL,"Forked Actions");

   // TEST 3: default branches
   ROOT::RDataFrame d2(treeName, fileName, {"b1"});
   auto d2f = d2.Filter([](double b1) { return b1 < 5; }).Filter(ok);
   auto c2 = d2f.Count();
   d2f.Foreach([](double b1) { std::cout << b1 << std::endl; });
      auto c2v = *c2;
   std::cout << "c2 " << c2v << std::endl;
   CheckRes(c2v,5ULL,"Default branches");

   // TEST 4: execute Run lazily and implicitly
   ROOT::RDataFrame d3(treeName, fileName, {"b1"});
   auto d3f = d3.Filter([](double b1) { return b1 < 4; }).Filter(ok);
   auto c3 = d3f.Count();
   auto c3v = *c3;
   std::cout << "c3 " << c3v << std::endl;
   CheckRes(c3v,4ULL,"Execute Run lazily and implicitly");

   // TEST 5: non trivial branch
   ROOT::RDataFrame d4(treeName, fileName, {"tracks"});
   auto d4f = d4.Filter([](FourVectors const & tracks) { return tracks.size() > 7; });
   auto c4 = d4f.Count();
   auto c4v = *c4;
   std::cout << "c4 " << c4v << std::endl;
   CheckRes(c4v,1ULL,"Non trivial test");

   // TEST 6: Create a histogram
   ROOT::RDataFrame d5(treeName, fileName, {"b2"});
   auto h1 = d5.Histo1D();
   auto h2 = d5.Histo1D("b1");
   TH1D dvHisto("dvHisto","The DV histo", 64, -8, 8);
   auto h3 = d5.Histo1D(std::move(dvHisto),"dv");
   auto h4 = d5.Histo1D<std::list<int>>("sl");
   std::cout << "Histo1: nEntries " << h1->GetEntries() << std::endl;
   std::cout << "Histo2: nEntries " << h2->GetEntries() << std::endl;
   std::cout << "Histo3: nEntries " << h3->GetEntries() << std::endl;
   std::cout << "Histo4: nEntries " << h4->GetEntries() << std::endl;

   // TEST 7: Define
   ROOT::RDataFrame d6(treeName, fileName);
   auto r6 = d6.Define("iseven", [](int b2) { return b2 % 2 == 0; }, {"b2"})
               .Filter([](bool iseven) { return iseven; }, {"iseven"})
               .Count();
   auto c6v = *r6;
   std::cout << c6v << std::endl;
   CheckRes(c6v, 10ULL, "Define");

   // TEST 8: Define with default branches, filters, non-trivial types
   ROOT::RDataFrame d7(treeName, fileName, {"tracks"});
   auto dd7 = d7.Filter([](int b2) { return b2 % 2 == 0; }, {"b2"})
                 .Define("ptsum", [](FourVectors const & tracks) {
                    double sum = 0;
                    for(auto& track: tracks)
                       sum += track.Pt();
                    return sum; });
   auto c7 = dd7.Count();
   auto h7 = dd7.Histo1D("ptsum");
   auto c7v = *c7;
   CheckRes(c7v, 10ULL, "Define complicated");
   std::cout << "Define Histo entries: " << h7->GetEntries() << std::endl;
   std::cout << "Define Histo mean: " << h7->GetMean() << std::endl;

   // TEST 9: Get minimum, maximum, sum, mean
   ROOT::RDataFrame d8(treeName, fileName, {"b2"});
   auto min_b2 = d8.Min();
   auto min_dv = d8.Min("dv");
   auto max_b2 = d8.Max();
   auto max_dv = d8.Max("dv");
   auto sum_b2 = d8.Sum();
   auto sum_dv = d8.Sum("dv");
   auto sum_b2_init = d8.Sum<int>("b2", 1);
   auto sum_dv_init = d8.Sum<std::vector<double>>("dv", 1.);
   auto mean_b2 = d8.Mean();
   auto mean_dv = d8.Mean("dv");

   auto min_b2v = *min_b2;
   auto min_dvv = *min_dv;
   auto max_b2v = *max_b2;
   auto max_dvv = *max_dv;
   auto sum_b2v = *sum_b2;
   auto sum_dvv = *sum_dv;
   auto sum_b2_initv = *sum_b2_init;
   auto sum_dv_initv = *sum_dv_init;
   auto mean_b2v = *mean_b2;
   auto mean_dvv = *mean_dv;

   CheckRes(min_b2v, 0., "Min of ints");
   CheckRes(min_dvv, -1., "Min of vector<double>");
   CheckRes(max_b2v, 361., "Max of ints");
   CheckRes(max_dvv, 19., "Max of vector<double>");
   CheckRes(sum_b2v, 2470., "Sum of ints");
   CheckRes(sum_dvv, 1490., "Sum of vector<double>");
   CheckRes(sum_b2_initv, 2471, "Sum of ints with init");
   CheckRes(sum_dv_initv, 1491., "Sum of vector<double> with init");
   CheckRes(mean_b2v, 123.5, "Mean of ints");
   CheckRes(mean_dvv, 5.1379310344827588963, "Mean of vector<double>");

   std::cout << "Min b2: " << *min_b2 << std::endl;
   std::cout << "Min dv: " << *min_dv << std::endl;
   std::cout << "Max b2: " << *max_b2 << std::endl;
   std::cout << "Max dv: " << *max_dv << std::endl;
   std::cout << "Sum b2: " << *sum_b2 << std::endl;
   std::cout << "Sum dv: " << *sum_dv << std::endl;
   std::cout << "Sum b2 init: " << *sum_b2_init << std::endl;
   std::cout << "Sum dv init: " << *sum_dv_init << std::endl;
   std::cout << "Mean b2: " << *mean_b2 << std::endl;
   std::cout << "Mean dv: " << *mean_dv << std::endl;

   // TEST 10: Get a full column
   ROOT::RDataFrame d9(treeName, fileName, {"tracks"});
   auto dd9 = d9.Filter([](int b2) { return b2 % 2 == 0; }, {"b2"})
                 .Define("ptsum", [](FourVectors const & tracks) {
                    double sum = 0;
                    for(auto& track: tracks)
                       sum += track.Pt();
                    return sum; });
   auto b2List = dd9.Take<int>("b2");
   auto ptsumVec = dd9.Take<double, std::vector<double>>("ptsum");

   for (auto& v : b2List) { // Test also the iteration without dereferencing
      std::cout << v << std::endl;
   }

   for (auto& v : *ptsumVec) {
      std::cout << v << std::endl;
   }

   // TEST 11: Re-hang action to RDataFrameProxy after running
   ROOT::RDataFrame d10(treeName, fileName, {"tracks"});
   auto d10f = d10.Filter([](FourVectors const & tracks) { return tracks.size() > 2; });
   auto c10 = d10f.Count();
   std::cout << "Count for the first run is " << *c10 << std::endl;
   auto d10f_2 = d10f.Filter([](FourVectors const & tracks) { return tracks.size() < 5; });
   auto c10_2 = d10f_2.Count();
   std::cout << "Count for the second run after adding a filter is " << *c10_2 << std::endl;
   std::cout << "Count for the first run was " << *c10 << std::endl;

   // TEST 12: head node which goes out of scope should remain valid
   auto l = [](FourVectors const & tracks) { return tracks.size() > 2; };
   auto giveMeFilteredDF = [&](){
      ROOT::RDataFrame d11(treeName, fileName, {"tracks"});
      auto a = d11.Filter(l);
      return a;
   };
   auto filteredDF = giveMeFilteredDF();
   auto c11 = filteredDF.Count();
   std::cout << *c11 << std::endl;

   // TEST 13: an action result pointer goes out of scope and the chain is ran
   ROOT::RDataFrame d11(treeName, fileName);
   auto d11c = d.Count();
   {
      std::vector<decltype(d11c)> v;
      for (int i=0;i<10000;++i)
         v.emplace_back(d.Count());
   }
   std::cout << "Count with action pointers which went out of scope: " << *d11c << std::endl;

   // TEST 14: fill 1D histograms
   ROOT::RDataFrame d12(treeName, fileName, {"b1","b2"});
   auto wh1 = d12.Histo1D<double, int>();
   auto wh2 = d12.Histo1D<std::vector<double>, std::list<int>>("dv","sl");
   std::cout << "Wh1 Histo entries: " << wh1->GetEntries() << std::endl;
   std::cout << "Wh1 Histo mean: " << wh1->GetMean() << std::endl;
   std::cout << "Wh2 Histo entries: " << wh2->GetEntries() << std::endl;
   std::cout << "Wh2 Histo mean: " << wh2->GetMean() << std::endl;

   // TEST 15: fill 2D histograms
   ROOT::RDataFrame d13(treeName, fileName, {"b1","b2","b3"});
   auto h12d = d13.Histo2D<double, int>(TH2D("h1","",64,0,1024,64,0,1024));
   auto h22d = d13.Histo2D<std::vector<double>, std::list<int>>(TH2D("h2","",64,0,1024,64,0,1024),"dv","sl");
   auto h32d = d13.Histo2D<double, int, float>(TH2D("h3","",64,0,1024,64,0,1024));
   std::cout << "h12d Histo entries: " << h12d->GetEntries() << std::endl;
   std::cout << "h22d Histo entries: " << h22d->GetEntries() << std::endl;
   std::cout << "h32d Histo entries: " << h32d->GetEntries() << " sum of weights: " << h32d->GetSumOfWeights() << std::endl;

   // TEST 15: fill 3D histograms
   ROOT::RDataFrame d14(treeName, fileName, {"b1","b2","b3","b4"});
   auto h13d = d14.Histo3D<double, int, float>(TH3D("h4","",64,0,1024,64,0,1024,64,0,1024));
   auto h23d = d14.Histo3D<std::vector<double>,
                           std::list<int>,
                           std::vector<float>>(TH3D("h5","",64,0,1024,64,0,1024,64,0,1024),"dv","sl","sv");
   auto h33d = d14.Histo3D<double, int, float, float>(TH3D("h6","",64,0,1024,64,0,1024,64,0,1024));
   std::cout << "h13d Histo entries: " << h13d->GetEntries() << std::endl;
   std::cout << "h23d Histo entries: " << h23d->GetEntries() << std::endl;
   std::cout << "h33d Histo entries: " << h33d->GetEntries() << " sum of weights: " << h33d->GetSumOfWeights() << std::endl;

   // TEST 16: Take a collection of collections
   ROOT::RDataFrame d15(1);
   auto vb = d15.Define("v", [](){std::vector<int> v {1,2,3}; return v;}).Take<std::vector<int>>("v");
   int nentries = 0;
   for (auto&& el : vb) {
      std::cout << "Entry " << nentries++ << std::endl;
      for (auto&& i: el) {
         std::cout << i << std::endl;
      }
   }
}

int main() {
   test_misc();
   return 0;
}
