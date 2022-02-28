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

TEST(RDFVary, VaryTwiceTheSameColumn)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   EXPECT_THROW(df.Vary(
                   {"x", "x"},
                   [] {
                      return ROOT::RVec<ROOT::RVecI>{{0}, {0}};
                   },
                   {}, 1, "broken"),
                std::logic_error);

   // and now the jitted version
   EXPECT_THROW(df.Vary({"x", "x"}, "ROOT::RVec<ROOT::RVecI>{{0}, {0}}", 1, "broken"), std::logic_error);
}

TEST(RDFVary, RequireVariationsHaveConsistentType)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1.f; });
   EXPECT_THROW(df.Vary("x", SimpleVariation, {}, 2), std::runtime_error);
}

// throwing exceptions from jitted code cause problems on windows
#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
TEST(RDFVary, RequireVariationsHaveConsistentTypeJitted)
{
     auto df = ROOT::RDataFrame(10).Define("x", [] { return 1.f; });
     {
        auto s = df.Vary("x", "ROOT::RVecD{x*0.1}", 1).Sum<float>("x");
        auto ss = VariationsFor(s);
        // before starting the event loop, we jit and notice the mismatch in types
        EXPECT_THROW(ss["nominal"], std::runtime_error);
     }

     {
        auto s2 = df.Define("y", [] { return 1; })
                     .Vary({"x", "y"}, "ROOT::RVec<ROOT::RVecD>{{x*0.1}, {y*0.1}}", 1, "broken")
                     .Sum("y");
        auto ss2 = VariationsFor(s2);
        // before starting the event loop, we jit and notice the mismatch in types
        EXPECT_THROW(ss2["nominal"], std::runtime_error);
     }
}
#endif

TEST(RDFVary, RequireReturnTypeIsRVec)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   EXPECT_THROW(df.Vary("x", "0", /*nVariations=*/2), std::runtime_error);
}

TEST(RDFVary, RequireNVariationsIsConsistent)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto s = df.Vary("x", SimpleVariation, {}, /*wrong=*/3).Sum<int>("x");
   auto all_s = VariationsFor(s);
   // now, when evaluating `SimpleVariation`, we should notice that it returns 2 values, not 3, and throw
   EXPECT_THROW(all_s["nominal"], std::runtime_error);
}

TEST(RDFVary, VariationsForDoesNotTriggerRun)
{
   ROOT::RDataFrame df(10);
   auto h = df.Define("x", [] { return 1; }).Histo1D<int>("x");
   auto hs = VariationsFor(h);
   EXPECT_EQ(df.GetNRuns(), 0);
}

TEST(RDFVary, VariationsForWithNoVariations)
{
   ROOT::RDataFrame df(10);
   auto h = df.Define("x", [] { return 1; }).Histo1D<int>("x");
   auto hs = VariationsFor(h);
   EXPECT_EQ(hs.GetKeys(), std::vector<std::string>{"nominal"});
}

TEST(RDFVary, GetVariations)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 0; }).Define("y", [] { return 10; });
   auto df2 = df.Vary("x", SimpleVariation, {}, 2)
                 .Vary(
                    {"x", "y"},
                    [] {
                       return ROOT::RVec<ROOT::RVecI>{{-1, 1}, {9, 11}};
                    },
                    {}, 2, "xy");
   auto variations = df2.GetVariations();

   // the internal list of variations is unordered, we might get either ordering
   EXPECT_TRUE(variations.AsString() ==
                  "Variations {xy:0, xy:1} affect columns {x, y}\nVariations {x:0, x:1} affect column x\n" ||
               variations.AsString() ==
                  "Variations {x:0, x:1} affect column x\nVariations {xy:0, xy:1} affect columns {x, y}\n");
}

TEST(RDFVary, VaryDefinePerSample)
{
   auto df = ROOT::RDataFrame(10).DefinePerSample("x", [](unsigned int, const ROOT::RDF::RSampleInfo &) { return 1; });
   auto s = df.Vary("x", SimpleVariation, {}, 2).Sum<int>("x");
   auto ss = ROOT::RDF::Experimental::VariationsFor(s);
   EXPECT_EQ(ss["nominal"], 1 * 10);
   EXPECT_EQ(ss["x:0"], -1 * 10);
   EXPECT_EQ(ss["x:1"], 2 * 10);
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

TEST_P(RDFVary, MultipleVariationsOnSameColumn)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto h = df.Vary("x", SimpleVariation, {}, {"down", "up"})
               .Vary("x",
                     [] {
                        return ROOT::RVecI{-2, 4};
                     },
                     {}, {"down2", "up2"})
               .Histo1D<int>("x");
   auto histos = VariationsFor(h);

   const auto expectedKeys = std::vector<std::string>{"nominal", "x:down", "x:down2", "x:up", "x:up2"};
   auto keys = histos.GetKeys();
   std::sort(keys.begin(), keys.end()); // key ordering is not guaranteed
   EXPECT_EQ(keys, expectedKeys);
   EXPECT_DOUBLE_EQ(histos["nominal"].GetMaximum(), 10.);
   EXPECT_DOUBLE_EQ(histos["nominal"].GetMean(), 1.);
   EXPECT_DOUBLE_EQ(histos["x:down"].GetMaximum(), 10.);
   EXPECT_DOUBLE_EQ(histos["x:down"].GetMean(), -1.);
   EXPECT_DOUBLE_EQ(histos["x:up"].GetMaximum(), 10.);
   EXPECT_DOUBLE_EQ(histos["x:up"].GetMean(), 2.);
   EXPECT_DOUBLE_EQ(histos["x:down2"].GetMaximum(), 10.);
   EXPECT_DOUBLE_EQ(histos["x:down2"].GetMean(), -2.);
   EXPECT_DOUBLE_EQ(histos["x:up2"].GetMaximum(), 10.);
   EXPECT_DOUBLE_EQ(histos["x:up2"].GetMean(), 4.);
}

TEST_P(RDFVary, SimultaneousVariations)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; }).Define("y", [] { return 42; });
   auto h = df.Vary(
                 {"x", "y"},
                 [] {
                    return ROOT::RVec<ROOT::RVecI>{{-1, 2, 3}, {41, 43, 44}};
                 },
                 {}, {"down", "up", "other"}, "xy")
               .Histo1D<int, int>("x", "y");
   auto histos = VariationsFor(h);

   const auto expectedKeys = std::vector<std::string>{"nominal", "xy:down", "xy:other", "xy:up"};
   auto keys = histos.GetKeys();
   std::sort(keys.begin(), keys.end()); // key ordering is not guaranteed
   EXPECT_EQ(keys, expectedKeys);
   EXPECT_DOUBLE_EQ(histos["nominal"].GetMaximum(), 42. * 10.);
   EXPECT_DOUBLE_EQ(histos["nominal"].GetMean(), 1.);
   EXPECT_DOUBLE_EQ(histos["xy:down"].GetMaximum(), 41. * 10.);
   EXPECT_DOUBLE_EQ(histos["xy:down"].GetMean(), -1.);
   EXPECT_DOUBLE_EQ(histos["xy:up"].GetMaximum(), 43. * 10.);
   EXPECT_DOUBLE_EQ(histos["xy:up"].GetMean(), 2.);
   EXPECT_DOUBLE_EQ(histos["xy:other"].GetMaximum(), 44 * 10.);
   EXPECT_DOUBLE_EQ(histos["xy:other"].GetMean(), 3.);
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

TEST_P(RDFVary, JittedDefine)
{
   // have a jitted Define that depends on a varied column
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto sum = df.Vary("x", SimpleVariation, {}, {"down", "up"}).Define("y", "x * 2").Sum<int>("y");
   auto sums = VariationsFor(sum);

   EXPECT_EQ(sums["nominal"], 20);
   EXPECT_EQ(sums["x:up"], 40);
   EXPECT_EQ(sums["x:down"], -20);
}

TEST_P(RDFVary, JittedFilter)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto sum = df.Vary("x", SimpleVariation, {}, 2).Filter("x > 0").Sum<int>("x");
   EXPECT_EQ(*sum, 10);

   auto sums = VariationsFor(sum);

   EXPECT_EQ(sums["nominal"], 10);
   EXPECT_EQ(sums["x:0"], 0);
   EXPECT_EQ(sums["x:1"], 20);

   // Now let's have the variation be propagated only through the Filter:
   // Sum("y") per se does not require any variation
   auto df2 = df.Define("y", [] { return 42; });
   auto sum2 = df2.Vary("x", SimpleVariation, {}, 2).Filter("x > 0").Sum<int>("y");
   EXPECT_EQ(*sum2, 420);

   auto sums2 = VariationsFor(sum2);

   EXPECT_EQ(sums2["nominal"], 420);
   EXPECT_EQ(sums2["x:0"], 0);
   EXPECT_EQ(sums2["x:1"], 420);
}

TEST_P(RDFVary, JittedVary)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto sum = df.Vary("x", "ROOT::RVecI{-1*x, 2*x}", 2).Sum<int>("x");
   EXPECT_EQ(*sum, 10);

   auto sums = VariationsFor(sum);

   EXPECT_EQ(sums["nominal"], 10);
   EXPECT_EQ(sums["x:0"], -10);
   EXPECT_EQ(sums["x:1"], 20);
}

TEST_P(RDFVary, JittedVaryDefineFilterAndAction)
{
   auto df = ROOT::RDataFrame(10).Define("x", "1");
   auto sum = df.Vary("x", "ROOT::RVecI{-1*x, 2*x}", {"down", "up"}, "myvariation").Filter("x > 0").Sum("x");
   EXPECT_EQ(*sum, 10);

   auto sums = VariationsFor(sum);

   EXPECT_EQ(sums["nominal"], 10);
   EXPECT_EQ(sums["myvariation:down"], 0);
   EXPECT_EQ(sums["myvariation:up"], 20);
}

TEST_P(RDFVary, VariationsForOnSameResult)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto h = df.Vary("x", SimpleVariation, {}, 2).Histo1D<int>("x");
   EXPECT_EQ(df.GetNRuns(), 0);
   EXPECT_EQ(int(h->GetEntries()), 10);
   EXPECT_DOUBLE_EQ(h->GetMean(), 1.);
   EXPECT_EQ(df.GetNRuns(), 1);

   auto checkHistos = [&h](ROOT::RDF::Experimental::RResultMap<TH1D> &histos) {
      EXPECT_EQ(int(h->GetEntries()), 10);
      EXPECT_EQ(int(histos["nominal"].GetEntries()), 10);
      EXPECT_DOUBLE_EQ(histos["nominal"].GetMean(), 1.);
      EXPECT_EQ(int(histos["x:0"].GetEntries()), 10);
      EXPECT_DOUBLE_EQ(histos["x:0"].GetMean(), -1.);
      EXPECT_EQ(int(histos["x:1"].GetEntries()), 10);
      EXPECT_DOUBLE_EQ(histos["x:1"].GetMean(), 2.);
   };

   auto histos = VariationsFor(h);
   checkHistos(histos);
   EXPECT_EQ(df.GetNRuns(), 2);
   auto histos2 = VariationsFor(h);
   checkHistos(histos2);
   checkHistos(histos); // check that these results are still correct
   EXPECT_EQ(df.GetNRuns(), 3);
}

TEST_P(RDFVary, VariedColumnIsRVec)
{
   // this is a tricky case for our internal logic as we have to distinguish varying a column of RVec type
   // from varying multiple columns: in both cases the Vary expression returns an RVec<RVec<..>>
   auto df = ROOT::RDataFrame(10).Define("x", [] { return ROOT::RVecI{1, 2, 3}; });
   auto h = df.Vary(
                 "x",
                 [] {
                    return ROOT::RVec<ROOT::RVecI>{{}, {4, 5}};
                 },
                 {}, 2)
               .Histo1D<ROOT::RVecI>("x");
   auto histos = VariationsFor(h);

   EXPECT_DOUBLE_EQ(histos["nominal"].GetEntries(), 3 * 10.);
   EXPECT_DOUBLE_EQ(histos["nominal"].GetMean(), 2.);
   EXPECT_DOUBLE_EQ(histos["nominal"].GetMaximum(), 10.);

   EXPECT_DOUBLE_EQ(histos["x:0"].GetEntries(), 0.);

   EXPECT_DOUBLE_EQ(histos["x:1"].GetEntries(), 2 * 10.);
   EXPECT_DOUBLE_EQ(histos["x:1"].GetMean(), 4.5);
   EXPECT_DOUBLE_EQ(histos["x:1"].GetMaximum(), 10.);
}

TEST_P(RDFVary, VariedHistosMustHaveNoDirectory)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto h = df.Vary("x", SimpleVariation, {}, 2).Histo1D<int>("x");
   auto hs = VariationsFor(h);

   EXPECT_EQ(h->GetDirectory(), nullptr);
   EXPECT_EQ(hs["nominal"].GetDirectory(), nullptr);
   EXPECT_EQ(hs["x:0"].GetDirectory(), nullptr);
   EXPECT_EQ(hs["x:1"].GetDirectory(), nullptr);
}

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, RDFVary, ::testing::Values(false));

// instantiate multi-thread tests
#ifdef R__USE_IMT
INSTANTIATE_TEST_SUITE_P(MT, RDFVary, ::testing::Values(true));
#endif

// TODO
// - interaction with Redefine
// - interaction with SaveGraph
// - throw a Range into the mix (for now, we should throw if Range + Vary I guess?)
// - interaction with TriggerChildrenCount
// - interaction with PartialUpdate
// - Vary of a result that does not have Clone
// - all missing actions, e.g. Report, Display, Snapshot
