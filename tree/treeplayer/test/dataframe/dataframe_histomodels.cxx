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
   std::vector<float> edgesf{1, 2, 3, 4, 5, 6, 10};
   auto h1edgesf = d.Histo1D(::TH1D("h1edgesf", "h1edgesf", (int)edgesf.size(), edgesf.data()), "x");
   auto h2edgesf = d.Histo1D({"h2edgesf", "h2edgesf", (int)edgesf.size(), edgesf.data()}, "x");
   std::vector<double> edgesd{1, 2, 3, 4, 5, 6, 10};
   auto h1edgesd = d.Histo1D(::TH1D("h1edgesd", "h1edgesd", (int)edgesd.size(), edgesd.data()), "x");
   auto h2edgesd = d.Histo1D({"h2edgesd", "h2edgesd", (int)edgesd.size(), edgesd.data()}, "x");

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
   std::vector<double> edgesX{1, 2, 3, 4, 5, 6, 10};
   std::vector<double> edgesY{1, 2, 3, 4, 5, 6, 10};
   auto h1eX = d.Histo2D(::TH2D("h1eX", "h1eX", (int)edgesX.size(), edgesX.data(), 64, 0, 10), "x", "y");
   auto h2eX = d.Histo2D({"h2eX", "h2eX", (int)edgesX.size(), edgesX.data(), 64, 0, 10}, "x", "y");
   auto h1eY = d.Histo2D(::TH2D("h1eX", "h1eX", 64, 0, 10, (int)edgesX.size(), edgesX.data()), "x", "y");
   auto h2eY = d.Histo2D({"h2eX", "h2eX", 64, 0, 10, (int)edgesX.size(), edgesX.data()}, "x", "y");
   auto h1eXeY = d.Histo2D(
      ::TH2D("h1eXeY", "h1eXeY", (int)edgesX.size(), edgesX.data(), (int)edgesY.size(), edgesY.data()), "x", "y");
   auto h2eXeY =
      d.Histo2D({"h2eXeY", "h2eXeY", (int)edgesX.size(), edgesX.data(), (int)edgesY.size(), edgesY.data()}, "x", "y");
   std::vector<float> edgesXf{1, 2, 3, 4, 5, 6, 10};
   std::vector<float> edgesYf{1, 2, 3, 4, 5, 6, 10};
   auto h1eXeYf = d.Histo2D(
      ::TH2D("h1eXeYf", "h1eXeYf", (int)edgesXf.size(), edgesXf.data(), (int)edgesYf.size(), edgesYf.data()), "x", "y");
   auto h2eXeYf = d.Histo2D(
      {"h2eXeYf", "h2eXeYf", (int)edgesXf.size(), edgesXf.data(), (int)edgesYf.size(), edgesYf.data()}, "x", "y");

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

   std::vector<double> edgesXd{1, 2, 3, 4, 5, 6, 10};
   std::vector<double> edgesYd{1, 2, 3, 4, 5, 6, 10};
   std::vector<double> edgesZd{1, 2, 3, 4, 5, 6, 10};
   auto h1e = d.Histo3D(::TH3D("h1e", "h1e", (int)edgesXd.size(), edgesXd.data(), (int)edgesYd.size(), edgesYd.data(),
                               (int)edgesZd.size(), edgesZd.data()),
                        "x", "y", "z");
   auto h2e = d.Histo3D({"h2e", "h2e", (int)edgesXd.size(), edgesXd.data(), (int)edgesYd.size(), edgesYd.data(),
                         (int)edgesZd.size(), edgesZd.data()},
                        "x", "y", "z");

   TDF::TH3DModel m0("m0", "m0", 64, 0, 10, 64, 0, 10, 64, 0, 10);
   TDF::TH3DModel m1(::TH3D("m1", "m1", 64, 0, 10, 64, 0, 10, 64, 0, 10));

   auto hm0 = d.Histo3D(m0, "x", "y", "z");
   auto hm1 = d.Histo3D(m1, "x", "y", "z");
   auto hm0w = d.Histo3D(m0, "x", "y", "z", "z");
   auto hm1w = d.Histo3D(m1, "x", "y", "z", "z");
}
