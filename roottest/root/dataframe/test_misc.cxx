#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TMath.h"
#include "TTree.h"
#include "TRandom3.h"
#include <cassert>
#include <iostream>

#include <gtest/gtest.h>

using FourVector = ROOT::Math::XYZTVector;
using FourVectors = std::vector<FourVector>;
using CylFourVector = ROOT::Math::RhoEtaPhiVector;

void getTracks(FourVectors &tracks)
{
   static TRandom3 R(1);
   const double M = 0.13957; // set pi+ mass
   auto nPart = R.Poisson(5);
   tracks.clear();
   tracks.reserve(nPart);
   for (size_t i = 0; i < nPart; ++i) {
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
}

struct RootTestRDFMisc : public ::testing::Test {
   constexpr static auto fFileName = "test_misc.root";
   constexpr static auto fTreeName = "myTree";
   static void SetUpTestCase()
   {
      auto f = std::make_unique<TFile>(fFileName, "RECREATE");
      auto t = std::make_unique<TTree>(fTreeName, fTreeName);

      double b1;
      int b2;
      float b3;
      float b4;
      std::vector<FourVector> tracks;
      std::vector<double> dv{-1, 2, 3, 4};
      std::vector<float> sv{-1, 2, 3, 4};
      std::list<int> sl{1, 2, 3, 4};
      t->Branch("b1", &b1);
      t->Branch("b2", &b2);
      t->Branch("b3", &b3);
      t->Branch("b4", &b4);
      t->Branch("tracks", &tracks);
      t->Branch("dv", &dv);
      t->Branch("sl", &sl);
      t->Branch("sv", &sv);

      for (int i = 0; i < 20; ++i) {
         b1 = i;
         b2 = i * i;
         b3 = sqrt(i * i * i);
         b4 = i;
         getTracks(tracks);
         dv.emplace_back(i);
         sl.emplace_back(i);
         sv.emplace_back(i * 0.5);
         t->Fill();
      }
      f->Write();
   }
   static void TearDownTestCase() { std::remove(fFileName); }
};

template <typename T0, typename T1>
void expect_vec_eq(const T0 &v1, const T1 &v2)
{
   ASSERT_EQ(v1.size(), v2.size()) << "Vectors 'v1' and 'v2' are of unequal length";
   for (std::size_t i = 0ull; i < v1.size(); ++i) {
      if constexpr (std::is_floating_point<typename T0::value_type>::value)
         EXPECT_FLOAT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
      else
         EXPECT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
   }
}

TEST_F(RootTestRDFMisc, Test1)
{

   ROOT::RDataFrame d(fTreeName, fFileName);
   auto ok = []() { return true; };

   // Foreach with an upstream filter returning always true: should see all values
   std::vector<double> b1Values;
   d.Filter(ok).Foreach([&b1Values](double x) { b1Values.push_back(x); }, {"b1"});
   expect_vec_eq(b1Values, std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
}

TEST_F(RootTestRDFMisc, Test2)
{
   // Ensure filters are applied before doing any action
   ROOT::RDataFrame d(fTreeName, fFileName);
   auto ok = []() { return true; };
   auto ko = []() { return false; };
   auto dd = d.Filter(ok);

   std::vector<int> b2Values;
   dd.Foreach([&b2Values](int y) { b2Values.push_back(y); }, {"b2"});
   expect_vec_eq(
      b2Values, std::vector<int>{0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361});
   auto c = dd.Count();
   EXPECT_EQ(*c, 20ULL) << "Count after a filter that returns always true should be 20.";

   auto ddd = dd.Filter(ko);
   std::vector<int> b2ValuesEmpty;
   ddd.Foreach([&b2ValuesEmpty](int y) { b2ValuesEmpty.push_back(y); }, {"b2"});
   expect_vec_eq(b2ValuesEmpty, std::vector<int>{});
}

TEST_F(RootTestRDFMisc, Test3)
{
   auto ok = []() { return true; };

   // Use default column names
   ROOT::RDataFrame d(fTreeName, fFileName, {"b1"});
   auto df = d.Filter([](double b1) { return b1 < 5; }).Filter(ok);
   std::vector<double> b1Values;
   df.Foreach([&b1Values](double b1) { b1Values.push_back(b1); }, {"b1"});
   expect_vec_eq(b1Values, std::vector<double>{0, 1, 2, 3, 4});
}

TEST_F(RootTestRDFMisc, Test4)
{
   ROOT::RDataFrame d(fTreeName, fFileName, {"tracks"});
   auto df = d.Filter([](FourVectors const &tracks) { return tracks.size() > 7; });
   auto c = df.Count();
   EXPECT_EQ(*c, 1);
}

TEST_F(RootTestRDFMisc, Test5)
{
   ROOT::RDataFrame d(fTreeName, fFileName, {"b2"});
   auto h1 = d.Histo1D();
   auto h2 = d.Histo1D("b1");
   TH1D dvHisto("dvHisto", "The DV histo", 64, -8, 8);
   auto h3 = d.Histo1D(std::move(dvHisto), "dv");
   auto h4 = d.Histo1D<std::list<int>>("sl");
   EXPECT_EQ(h1->GetEntries(), 20);
   EXPECT_EQ(h2->GetEntries(), 20);
   EXPECT_EQ(h3->GetEntries(), 290);
   EXPECT_EQ(h4->GetEntries(), 290);
}

TEST_F(RootTestRDFMisc, Test6)
{
   ROOT::RDataFrame d(fTreeName, fFileName);
   auto r = d.Define("iseven", [](int b2) { return b2 % 2 == 0; }, {"b2"})
               .Filter([](bool iseven) { return iseven; }, {"iseven"})
               .Count();
   EXPECT_EQ(*r, 10);
}

TEST_F(RootTestRDFMisc, Test7)
{
   ROOT::RDataFrame d(fTreeName, fFileName, {"tracks"});
   auto dd = d.Filter([](int b2) { return b2 % 2 == 0; }, {"b2"}).Define("ptsum", [](FourVectors const &tracks) {
      double sum = 0;
      for (auto &track : tracks)
         sum += track.Pt();
      return sum;
   });
   auto c = dd.Count();
   auto h = dd.Histo1D("ptsum");

   EXPECT_EQ(*c, 10);
   EXPECT_EQ(h->GetEntries(), 10);
   EXPECT_FLOAT_EQ(h->GetMean(), 60.952423);
}

TEST_F(RootTestRDFMisc, Test8)
{
   // TEST 9: Get minimum, maximum, sum, mean
   ROOT::RDataFrame d8(fTreeName, fFileName, {"b2"});
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

   EXPECT_FLOAT_EQ(min_b2v, 0.);
   EXPECT_FLOAT_EQ(min_dvv, -1.);
   EXPECT_FLOAT_EQ(max_b2v, 361.);
   EXPECT_FLOAT_EQ(max_dvv, 19.);
   EXPECT_FLOAT_EQ(sum_b2v, 2470.);
   EXPECT_FLOAT_EQ(sum_dvv, 1490.);
   EXPECT_FLOAT_EQ(sum_b2_initv, 2471.);
   EXPECT_FLOAT_EQ(sum_dv_initv, 1491.);
   EXPECT_FLOAT_EQ(mean_b2v, 123.5);
   EXPECT_FLOAT_EQ(mean_dvv, 5.1379310344827588963);
}

TEST_F(RootTestRDFMisc, Test9)
{
   ROOT::RDataFrame d(fTreeName, fFileName, {"tracks"});
   auto dd = d.Filter([](int b2) { return b2 % 2 == 0; }, {"b2"}).Define("ptsum", [](FourVectors const &tracks) {
      double sum = 0;
      for (auto &track : tracks)
         sum += track.Pt();
      return sum;
   });
   auto b2List = dd.Take<int>("b2");
   auto ptsumVec = dd.Take<double, std::vector<double>>("ptsum");

   expect_vec_eq(*b2List, std::vector<int>{0, 4, 16, 36, 64, 100, 144, 196, 256, 324});
   expect_vec_eq(*ptsumVec, std::vector<double>{61.0508, 61.095249, 34.602367, 39.667374, 77.2068, 28.842136, 90.5761,
                                                115.61565, 50.1965, 50.671272});
}

TEST_F(RootTestRDFMisc, Test10)
{

   // Different filters can be applied correctly even in-between runs
   ROOT::RDataFrame d(fTreeName, fFileName, {"tracks"});
   auto df = d.Filter([](FourVectors const &tracks) { return tracks.size() > 2; });
   auto c = df.Count();
   EXPECT_EQ(*c, 18);
   auto df_2 = df.Filter([](FourVectors const &tracks) { return tracks.size() < 5; });
   auto c_2 = df_2.Count();
   EXPECT_EQ(*c_2, 8);
}

TEST_F(RootTestRDFMisc, Test11)
{
   // head node which goes out of scope does not invalidate computation graph
   auto l = [](FourVectors const &tracks) { return tracks.size() > 2; };
   auto giveMeFilteredDF = [&]() {
      ROOT::RDataFrame d(fTreeName, fFileName, {"tracks"});
      auto a = d.Filter(l);
      return a;
   };
   auto filteredDF = giveMeFilteredDF();
   auto c11 = filteredDF.Count();
   EXPECT_EQ(*c11, 18);
}

TEST_F(RootTestRDFMisc, Test12)
{
   // Even if action pointers attached to the same computation graph go out of scope, the graph remains valid for the
   // action pointer that is still in scope.
   ROOT::RDataFrame d(fTreeName, fFileName);
   auto c = d.Count();
   {
      std::vector<decltype(c)> v;
      for (int i = 0; i < 10000; ++i)
         v.emplace_back(d.Count());
   }
   EXPECT_EQ(*c, 20);
}

TEST_F(RootTestRDFMisc, Test13)
{
   // Fill 1D histograms also using default column names
   ROOT::RDataFrame d(fTreeName, fFileName, {"b1", "b2"});
   auto wh1 = d.Histo1D<double, int>();
   auto wh2 = d.Histo1D<std::vector<double>, std::list<int>>("dv", "sl");

   EXPECT_EQ(wh1->GetEntries(), 20);
   EXPECT_FLOAT_EQ(wh1->GetMean(), 14.615385);
   EXPECT_EQ(wh2->GetEntries(), 290);
   EXPECT_FLOAT_EQ(wh2->GetMean(), 9.05882);
}

TEST_F(RootTestRDFMisc, Test14)
{
   // Fill 2D histograms also using default column names
   ROOT::RDataFrame d(fTreeName, fFileName, {"b1", "b2", "b3"});
   auto h2d = d.Histo2D<double, int>(TH2D("h1", "", 64, 0, 1024, 64, 0, 1024));
   auto h2dd = d.Histo2D<std::vector<double>, std::list<int>>(TH2D("h2", "", 64, 0, 1024, 64, 0, 1024), "dv", "sl");
   auto h2ddd = d.Histo2D<double, int, float>(TH2D("h3", "", 64, 0, 1024, 64, 0, 1024));

   EXPECT_EQ(h2d->GetEntries(), 20);
   EXPECT_EQ(h2dd->GetEntries(), 290);
   EXPECT_EQ(h2ddd->GetEntries(), 20);
   EXPECT_FLOAT_EQ(h2ddd->GetSumOfWeights(), 671.354);
}

TEST_F(RootTestRDFMisc, Test15)
{
   // Fill 3D histograms also using default column names
   ROOT::RDataFrame d(fTreeName, fFileName, {"b1", "b2", "b3", "b4"});
   auto h3d = d.Histo3D<double, int, float>(TH3D("h4", "", 64, 0, 1024, 64, 0, 1024, 64, 0, 1024));
   auto h3dd = d.Histo3D<std::vector<double>, std::list<int>, std::vector<float>>(
      TH3D("h5", "", 64, 0, 1024, 64, 0, 1024, 64, 0, 1024), "dv", "sl", "sv");
   auto h3ddd = d.Histo3D<double, int, float, float>(TH3D("h6", "", 64, 0, 1024, 64, 0, 1024, 64, 0, 1024));

   EXPECT_EQ(h3d->GetEntries(), 20);
   EXPECT_EQ(h3dd->GetEntries(), 290);
   EXPECT_EQ(h3ddd->GetEntries(), 20);
   EXPECT_FLOAT_EQ(h3ddd->GetSumOfWeights(), 190.);
}

TEST_F(RootTestRDFMisc, Test16)
{
   // Take a collection of collections
   ROOT::RDataFrame d15(1);
   auto vb = d15.Define("v",
                        []() {
                           std::vector<int> v{1, 2, 3};
                           return v;
                        })
                .Take<std::vector<int>>("v");
   auto vbv = *vb;
   ASSERT_EQ(vbv.size(), 1);
   expect_vec_eq(vbv[0], std::vector<int>{1, 2, 3});
}

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
