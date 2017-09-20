#include "ROOT/TDataFrame.hxx"

#include "gtest/gtest.h"

using namespace ROOT::Experimental;

TEST(TDataFrameHistoModels, Histo1D)
{
   TDataFrame tdf(10);
   auto x = 0.;
   auto d = tdf.Define("x", [&x]() { return x++; }).Define("w", [&x]() { return x + 1.; });
   auto h1 = d.Histo1D(::TH1D("h1", "h1", 64, 0, 10), "x");
   auto h2 = d.Histo1D({"h2", "h2", 64, 0, 10}, "x");
   auto h1w = d.Histo1D(::TH1D("h0w", "h0w", 64, 0, 10), "x", "w");
   auto h2w = d.Histo1D({"h2w", "h2w", 64, 0, 10}, "x", "w");

   TDF::TH1DModel m0("m0", "m0", 64, 0, 10);
   TDF::TH1DModel m1(::TH1D("m1", "m1", 64, 0, 10));

   auto hm0 = d.Histo1D(m0, "x");
   auto hm1 = d.Histo1D(m1, "x");
   auto hm0w = d.Histo1D(m0, "x", "w");
   auto hm1w = d.Histo1D(m1, "x", "w");
}

TEST(TDataFrameHistoModels, Histo2D)
{
   TDataFrame tdf(10);
   auto x = 0.;
   auto d = tdf.Define("x", [&x]() { return x++; }).Define("y", [&x]() { return x + .1; });
   auto h1 = d.Histo2D(::TH2D("h1", "h1", 64, 0, 10, 64, 0, 10), "x", "y");
   auto h2 = d.Histo2D({"h2", "h2", 64, 0, 10, 64, 0, 10}, "x", "y");

   TDF::TH2DModel m0("m0", "m0", 64, 0, 10, 64, 0, 10);
   TDF::TH2DModel m1(::TH2D("m1", "m1", 64, 0, 10, 64, 0, 10));

   auto hm0 = d.Histo2D(m0, "x", "y");
   auto hm1 = d.Histo2D(m1, "x", "y");
   auto hm0w = d.Histo2D(m0, "x", "y", "x");
   auto hm1w = d.Histo2D(m1, "x", "y", "x");
}

TEST(TDataFrameHistoModels, Histo3D)
{
   TDataFrame tdf(10);
   auto x = 0.;
   auto d = tdf.Define("x", [&x]() { return x++; }).Define("y", [&x]() { return x + .1; }).Define("z", [&x]() {
      return x + .1;
   });
   auto h1 = d.Histo3D(::TH3D("h1", "h1", 64, 0, 10, 64, 0, 10, 64, 0, 10), "x", "y", "z");
   auto h2 = d.Histo3D({"h2", "h2", 64, 0, 10, 64, 0, 10, 64, 0, 10}, "x", "y", "z");

   TDF::TH3DModel m0("m0", "m0", 64, 0, 10, 64, 0, 10, 64, 0, 10);
   TDF::TH3DModel m1(::TH3D("m1", "m1", 64, 0, 10, 64, 0, 10, 64, 0, 10));

   auto hm0 = d.Histo3D(m0, "x", "y", "z");
   auto hm1 = d.Histo3D(m1, "x", "y", "z");
   auto hm0w = d.Histo3D(m0, "x", "y", "z", "z");
   auto hm1w = d.Histo3D(m1, "x", "y", "z", "z");
}
