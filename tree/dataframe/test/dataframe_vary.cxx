#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <TSystem.h>

#include <thread> // std::thread::hardware_concurrency

#include <gtest/gtest.h>

using ROOT::RDF::Experimental::VariationsFor;

class RDFVary : public ::testing::TestWithParam<bool> {
protected:
   RDFVary() : NSLOTS(GetParam() ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam())
         ROOT::EnableImplicitMT(NSLOTS);
   }
   ~RDFVary()
   {
      if (GetParam())
         ROOT::DisableImplicitMT();
   }
   const unsigned int NSLOTS;
};

auto SimpleVariation()
{
   return ROOT::RVecI{-1, 2};
}

TEST(RDFVary, RequireExistingColumn)
{
   ROOT::RDataFrame df(10);
   EXPECT_THROW(df.Vary("x", SimpleVariation, {}, 2), std::runtime_error);
}

TEST(RDFVary, RequireVariationsHaveConsistentType)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1.f; });
   EXPECT_THROW(df.Vary("x", SimpleVariation, {}, 2), std::runtime_error);
}

// FIXME requires jitting
//TEST(RDFVary, RequireReturnTypeIsRVec)
//{
//   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
//   df.Vary(
//      "x", "0", {}, 1);
//}

TEST(RDFVary, RequireNVariationsIsConsistent)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto s = df.Vary("x", SimpleVariation, {}, /*wrong=*/3).Sum<int>("x");
   auto all_s = VariationsFor(s);
   // now, when evaluating `SimpleVariation`, we should notice that it returns 2 values, not 3, and complain.
   ASSERT_DEATH(all_s["nominal"], "Variation expression has wrong size");
}

TEST_P(RDFVary, SimpleSum)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto sum = df.Vary("x", SimpleVariation, {}, 2).Sum<int>("x");
   EXPECT_EQ(*sum, 10);

   auto sums = VariationsFor(sum);

   EXPECT_EQ(sums["nominal"], 10);
   EXPECT_EQ(sums["x:0"], -10);
   EXPECT_EQ(sums["x:1"], 20);
}

TEST_P(RDFVary, SimpleHisto)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto h = df.Vary("x", SimpleVariation, {}, 2).Histo1D<int>("x");
   auto hs = VariationsFor(h);

   EXPECT_DOUBLE_EQ(h->GetMean(), 1);
   EXPECT_DOUBLE_EQ(hs["nominal"].GetMean(), 1);
   EXPECT_DOUBLE_EQ(hs["x:0"].GetMean(), -1);
   EXPECT_DOUBLE_EQ(hs["x:1"].GetMean(), 2);
}

TEST_P(RDFVary, SimpleHistoWithAxes) // uses FillParHelper instead of FillHelper
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto h = df.Vary("x", SimpleVariation, {}, 2).Histo1D<int>({"", "", 20, -10, 10}, "x");
   auto hs = VariationsFor(h);

   EXPECT_DOUBLE_EQ(h->GetMean(), 1);
   EXPECT_DOUBLE_EQ(hs["nominal"].GetMean(), 1);
   EXPECT_DOUBLE_EQ(hs["x:0"].GetMean(), -1);
   EXPECT_DOUBLE_EQ(hs["x:1"].GetMean(), 2);
}

TEST_P(RDFVary, SimpleGraph)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto g = df.Vary("x", SimpleVariation, {}, 2).Graph<int, int>("x", "x");
   auto gs = VariationsFor(g);

   EXPECT_DOUBLE_EQ(g->GetMean(), 1);
   EXPECT_DOUBLE_EQ(gs["nominal"].GetMean(), 1);
   EXPECT_DOUBLE_EQ(gs["x:0"].GetMean(), -1);
   EXPECT_DOUBLE_EQ(gs["x:1"].GetMean(), 2);
}

TEST_P(RDFVary, SimpleTake)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto v = df.Vary("x", SimpleVariation, {}, 2).Take<int>("x");
   auto vs = VariationsFor(v);

   EXPECT_EQ(*v, std::vector<int>(10, 1));
   EXPECT_EQ(vs["nominal"], std::vector<int>(10, 1));
   EXPECT_EQ(vs["x:0"], std::vector<int>(10, -1));
   EXPECT_EQ(vs["x:1"], std::vector<int>(10, 2));
}

TEST_P(RDFVary, SimpleMin)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto m = df.Vary("x", SimpleVariation, {}, 2).Min<int>("x");
   auto ms = VariationsFor(m);

   EXPECT_EQ(*m, 1);
   EXPECT_EQ(ms["nominal"], 1);
   EXPECT_EQ(ms["x:0"], -1);
   EXPECT_EQ(ms["x:1"], 2);
}

TEST_P(RDFVary, SimpleMax)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto m = df.Vary("x", SimpleVariation, {}, 2).Max<int>("x");
   auto ms = VariationsFor(m);

   EXPECT_EQ(*m, 1);
   EXPECT_EQ(ms["nominal"], 1);
   EXPECT_EQ(ms["x:0"], -1);
   EXPECT_EQ(ms["x:1"], 2);
}

TEST_P(RDFVary, SimpleMean)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto m = df.Vary("x", SimpleVariation, {}, 2).Mean<int>("x");
   auto ms = VariationsFor(m);

   EXPECT_DOUBLE_EQ(*m, 1.);
   EXPECT_DOUBLE_EQ(ms["nominal"], 1.);
   EXPECT_DOUBLE_EQ(ms["x:0"], -1.);
   EXPECT_DOUBLE_EQ(ms["x:1"], 2.);
}

TEST_P(RDFVary, SimpleStdDev)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto m = df.Vary("x", SimpleVariation, {}, 2).StdDev<int>("x");
   auto ms = VariationsFor(m);

   EXPECT_DOUBLE_EQ(*m, 0.);
   EXPECT_DOUBLE_EQ(ms["nominal"], 0.);
   EXPECT_DOUBLE_EQ(ms["x:0"], 0.);
   EXPECT_DOUBLE_EQ(ms["x:1"], 0.);
}

TEST_P(RDFVary, SimpleAggregate)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto a = df.Vary("x", SimpleVariation, {}, 2).Aggregate(std::plus<int>{}, std::plus<int>{}, "x");

   auto as = VariationsFor(a);

   EXPECT_EQ(*a, 10);
   EXPECT_EQ(as["nominal"], 10);
   EXPECT_EQ(as["x:0"], -10);
   EXPECT_EQ(as["x:1"], 20);
}

TEST_P(RDFVary, MixedVariedNonVaried)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; }).Define("y", [] { return 42; });
   auto h = df.Vary("x", SimpleVariation, {}, {"down", "up"}).Histo1D<int, int>("x", "y");
   auto histos = VariationsFor(h);

   const auto expectedKeys = std::vector<std::string>{"nominal", "x:down", "x:up"};
   auto keys = histos.GetKeys();
   std::sort(keys.begin(), keys.end()); // key ordering is not guaranteed
   EXPECT_EQ(keys, expectedKeys);
   EXPECT_DOUBLE_EQ(histos["nominal"].GetMaximum(), 42. * 10.);
   EXPECT_DOUBLE_EQ(histos["nominal"].GetMean(), 1.);
   EXPECT_DOUBLE_EQ(histos["x:down"].GetMaximum(), 42. * 10.);
   EXPECT_DOUBLE_EQ(histos["x:down"].GetMean(), -1.);
   EXPECT_DOUBLE_EQ(histos["x:up"].GetMaximum(), 42. * 10.);
   EXPECT_DOUBLE_EQ(histos["x:up"].GetMean(), 2.);
}

TEST_P(RDFVary, MultipleVariations)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; }).Define("y", [] { return 42; });
   auto h = df.Vary("x", SimpleVariation, {}, {"down", "up"})
               .Vary("y", SimpleVariation, {}, {"down", "up"})
               .Histo1D<int, int>("x", "y");
   auto histos = VariationsFor(h);

   const auto expectedKeys = std::vector<std::string>{"nominal", "x:down", "x:up", "y:down", "y:up"};
   auto keys = histos.GetKeys();
   std::sort(keys.begin(), keys.end()); // key ordering is not guaranteed
   EXPECT_EQ(keys, expectedKeys);
   EXPECT_DOUBLE_EQ(histos["nominal"].GetMaximum(), 42. * 10.);
   EXPECT_DOUBLE_EQ(histos["nominal"].GetMean(), 1.);
   EXPECT_DOUBLE_EQ(histos["x:down"].GetMaximum(), 42. * 10.);
   EXPECT_DOUBLE_EQ(histos["x:down"].GetMean(), -1.);
   EXPECT_DOUBLE_EQ(histos["x:up"].GetMaximum(), 42. * 10.);
   EXPECT_DOUBLE_EQ(histos["x:up"].GetMean(), 2.);
   EXPECT_DOUBLE_EQ(histos["y:down"].GetMinimum(), -1. * 10.);
   EXPECT_DOUBLE_EQ(histos["y:down"].GetMean(), 1.);
   EXPECT_DOUBLE_EQ(histos["y:up"].GetMaximum(), 2. * 10.);
   EXPECT_DOUBLE_EQ(histos["y:up"].GetMean(), 1.);
}

TEST_P(RDFVary, SimultaneousVariations)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; }).Define("y", [] { return 42; });
   auto h = df.Vary(
                 {"x", "y"},
                 [] {
                    return ROOT::RVec<ROOT::RVecI>{{-1, 2}, {41, 43}};
                 },
                 {}, {"down", "up"}, "xy")
               .Histo1D<int, int>("x", "y");
   auto histos = VariationsFor(h);

   const auto expectedKeys = std::vector<std::string>{"nominal", "xy:down", "xy:up"};
   auto keys = histos.GetKeys();
   std::sort(keys.begin(), keys.end()); // key ordering is not guaranteed
   EXPECT_EQ(keys, expectedKeys);
   EXPECT_DOUBLE_EQ(histos["nominal"].GetMaximum(), 42. * 10.);
   EXPECT_DOUBLE_EQ(histos["nominal"].GetMean(), 1.);
   EXPECT_DOUBLE_EQ(histos["xy:down"].GetMaximum(), 41. * 10.);
   EXPECT_DOUBLE_EQ(histos["xy:down"].GetMean(), -1.);
   EXPECT_DOUBLE_EQ(histos["xy:up"].GetMaximum(), 43. * 10.);
   EXPECT_DOUBLE_EQ(histos["xy:up"].GetMean(), 2.);
}

TEST_P(RDFVary, VaryTTreeBranch)
{
   const auto fname = "rdfvary_varyttreebranch.root";
   ROOT::RDataFrame(10).Define("x", [] { return 1; }).Snapshot<int>("t", fname, {"x"});

   ROOT::RDataFrame df("t", fname);
   auto sum = df.Vary("x", SimpleVariation, {}, 2).Sum<int>("x");
   EXPECT_EQ(*sum, 10);

   auto sums = VariationsFor(sum);

   EXPECT_EQ(sums["nominal"], 10);
   EXPECT_EQ(sums["x:0"], -10);
   EXPECT_EQ(sums["x:1"], 20);

   gSystem->Unlink(fname);
}

TEST_P(RDFVary, DefineDependingOnVariation)
{
   // have a Define that depends on a varied column
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto sum =
      df.Vary("x", SimpleVariation, {}, {"down", "up"}).Define("y", [](int x) { return x * 2; }, {"x"}).Sum<int>("y");
   auto sums = VariationsFor(sum);

   EXPECT_EQ(sums["nominal"], 20);
   EXPECT_EQ(sums["x:up"], 40);
   EXPECT_EQ(sums["x:down"], -20);
}

TEST_P(RDFVary, DefineDependingOnVariations)
{
   // have a Define that depends on multiple varied columns
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; }).Define("y", [] { return 42; });
   auto sum = df.Vary("x", SimpleVariation, {}, {"down", "up"})
                 .Vary(
                    "y",
                    [](int y) {
                       return ROOT::RVecI{y - 2, y + 8};
                    },
                    {"y"}, {"low", "high"}, "yshift")
                 .Define("z", [](int x, int y) { return x + y; }, {"x", "y"})
                 .Sum<int>("z");
   auto sums = VariationsFor(sum);

   EXPECT_EQ(sums.GetKeys().size(), 5);
   EXPECT_EQ(sums["nominal"], 430);
   EXPECT_EQ(sums["x:up"], 440);
   EXPECT_EQ(sums["x:down"], 410);
   EXPECT_EQ(sums["yshift:high"], 510);
   EXPECT_EQ(sums["yshift:low"], 410);
}

TEST(RDFVary, VaryAndAlias)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; }).Alias("y", "x").Vary("x", SimpleVariation, {}, 2);
   auto s1 = df.Sum<int>("y");
   auto s2 = df.Define("z", [](int y) { return y; }, {"y"}).Sum<int>("z");

   auto sums1 = VariationsFor(s1);
   EXPECT_EQ(sums1["nominal"], 10);
   EXPECT_EQ(sums1["x:0"], -10);
   EXPECT_EQ(sums1["x:1"], 20);

   auto sums2 = VariationsFor(s2);
   EXPECT_EQ(sums2["nominal"], 10);
   EXPECT_EQ(sums2["x:0"], -10);
   EXPECT_EQ(sums2["x:1"], 20);
}

TEST_P(RDFVary, DifferentVariationsInDifferentBranches)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto snovary = df.Sum<int>("x");
   auto s1 = df.Vary("x", SimpleVariation, {}, 2).Sum<int>("x");
   auto s2 = df.Vary(
                  "x",
                  []() {
                     return ROOT::RVecI{-42, 42};
                  },
                  {}, 2)
                .Sum<int>("x");

   auto sumsnovary = VariationsFor(snovary);
   EXPECT_EQ(sumsnovary.GetKeys(), std::vector<std::string>{"nominal"});
   EXPECT_EQ(sumsnovary["nominal"], 10);

   auto sums1 = VariationsFor(s1);
   EXPECT_EQ(sums1["nominal"], 10);
   EXPECT_EQ(sums1["x:0"], -10);
   EXPECT_EQ(sums1["x:1"], 20);

   auto sums2 = VariationsFor(s2);
   EXPECT_EQ(sums2["nominal"], 10);
   EXPECT_EQ(sums2["x:0"], -420);
   EXPECT_EQ(sums2["x:1"], 420);
}

TEST_P(RDFVary, FilterDependingOnVariation)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto sum = df.Vary("x", SimpleVariation, {}, 2).Filter([](int x) { return x > 0; }, {"x"}).Sum<int>("x");
   EXPECT_EQ(*sum, 10);

   auto sums = VariationsFor(sum);

   EXPECT_EQ(sums["nominal"], 10);
   EXPECT_EQ(sums["x:0"], 0);
   EXPECT_EQ(sums["x:1"], 20);
}

TEST_P(RDFVary, FilterChainDependingOnVariations)
{
   auto sum = ROOT::RDataFrame(10)
                 .Define("x", [] { return 1; })
                 .Vary("x", SimpleVariation, {}, 2)
                 .Define("y", [](int x) { return x * 2; }, {"x"})
                 .Vary("y", SimpleVariation, {}, 2)
                 .Filter([](int x) { return x > 0; }, {"x"})
                 .Filter([](int x, int y) { return x + y > 0; }, {"x", "y"})
                 .Define("z", [](int x, int y) { return x + y; }, {"x", "y"})
                 .Sum<int>("z");
   EXPECT_EQ(*sum, 30);

   auto sums = VariationsFor(sum);

   EXPECT_EQ(sums["nominal"], 30);
   EXPECT_EQ(sums["x:0"], 0);
   EXPECT_EQ(sums["x:1"], 60);
   EXPECT_EQ(sums["y:0"], 0);
   EXPECT_EQ(sums["y:1"], 30);
}

TEST_P(RDFVary, JittedAction)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto h = df.Vary("x", SimpleVariation, {}, 2).Histo1D("x");
   auto hs = VariationsFor(h);

   EXPECT_DOUBLE_EQ(h->GetMean(), 1);
   EXPECT_DOUBLE_EQ(hs["nominal"].GetMean(), 1);
   EXPECT_DOUBLE_EQ(hs["x:0"].GetMean(), -1);
   EXPECT_DOUBLE_EQ(hs["x:1"].GetMean(), 2);
}

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, RDFVary, ::testing::Values(false));

// instantiate multi-thread tests
#ifdef R__USE_IMT
INSTANTIATE_TEST_SUITE_P(MT, RDFVary, ::testing::Values(true));
#endif

// TODO
// - jitted Define depending on variation
// - jitted Filter depending on variation
// - try calling VariationsFor twice on the same result (with different filters in between)
// - check that after running a second event loop, the results from the first are still correct
// - jitted Vary expression
// - test with TH1 (in particular running the varied event loop _after_ the nominal to check that results are reset)
// - test calling VariationsFor on something that's not varied
// - interaction with Redefine
// - interaction with SaveGraph
// - throw a Range into the mix (for now, we should throw if Range + Vary I guess?)
// - interaction with TriggerChildrenCount
// - interaction with PartialUpdate
// - Vary of a result that does not have Clone
// - all missing actions, e.g. Report, Display, Snapshot
