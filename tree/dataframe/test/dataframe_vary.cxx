#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <TSystem.h>

#include <thread> // std::thread::hardware_concurrency

#include "SimpleFiller.h" // for VaryFill

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
   EXPECT_THROW(
      try { df.Vary("x", SimpleVariation, {}, 2); } catch (const std::runtime_error &err) {
         const auto msg = "RDataFrame::Vary: cannot redefine or vary column \"x\". "
                          "No column with that name was found in the dataset. "
                          "Use Define to create a new column.";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::runtime_error);
}

TEST(RDFVary, VaryTwiceTheSameColumn)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   EXPECT_THROW(
      try {
         df.Vary(
            {"x", "x"},
            [] {
               return ROOT::RVec<ROOT::RVecI>{{0}, {0}};
            },
            {}, 1, "broken");
      } catch (const std::logic_error &err) {
         const auto msg = "A column name was passed to the same Vary invocation multiple times.";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::logic_error);

   // and now the jitted version
   EXPECT_THROW(
      try {
         df.Vary({"x", "x"}, "ROOT::RVec<ROOT::RVecI>{{0}, {0}}", 1, "broken");
      } catch (const std::logic_error &err) {
         const auto msg = "A column name was passed to the same Vary invocation multiple times.";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::logic_error);
}

TEST(RDFVary, RequireVariationsHaveConsistentType)
{
   auto df1 = ROOT::RDataFrame(10).Define("x", [] { return 1.f; });
   // x is float, variation expression cannot return RVec<int>, must be RVec<float>
   EXPECT_THROW(
      try { df1.Vary("x", SimpleVariation, {}, 2); } catch (const std::runtime_error &err) {
         const auto msg = "Varied values for column \"x\" have a different type (int) than the nominal value (float).";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::runtime_error);

   auto df2 = df1.Define("z", [] { return 0.f; });
   // now with multiple columns: x and z are float, variation expression cannot return RVec<RVecI>, must be RVec<RVecF>
   EXPECT_THROW(
      try {
         df2.Vary(
            {"x", "z"},
            [] {
               return ROOT::RVec<ROOT::RVecI>{{0}, {1}};
            },
            {}, 1, "broken");
      } catch (const std::runtime_error &err) {
         const auto msg = "Varied values for column \"x\" have a different type (int) than the nominal value (float).";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::runtime_error);

   auto df3 = df1.Define("y", [] { return 1; });
   // cannot simultaneously vary x and y if they don't have the same type
   EXPECT_THROW(
      try {
         df3.Vary(
            {"x", "y"},
            [] {
               return ROOT::RVec<ROOT::RVecF>{{0.f}, {1.f}};
            },
            {}, 1, "broken");
      } catch (const std::runtime_error &err) {
         const auto msg = "Cannot simultaneously vary multiple columns of different types.";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::runtime_error);
}

// throwing exceptions from jitted code cause problems on windows and MacOS+M1
#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
#if !(defined(R__MACOSX) && defined(__arm64__))
TEST(RDFVary, RequireVariationsHaveConsistentTypeJitted)
{
   // non-jitted Define, jitted Vary with incompatible type
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1.f; });
   {
      auto s = df.Vary("x", "ROOT::RVecD{x*0.1}", 1).Sum<float>("x");
      auto ss = VariationsFor(s);
      // before starting the event loop, we jit and notice the mismatch in types
      EXPECT_THROW(
         try { ss["nominal"]; } catch (const std::runtime_error &err) {
            const auto msg = "RVariationReader: type mismatch: column \"x\" is being used as float but the "
                             "Define or Vary node advertises it as double";
            EXPECT_STREQ(err.what(), msg);
            throw;
         },
         std::runtime_error);
   }

   // non-jitted Define, jitted Vary with incompatible type (multiple columns varied simultaneously
   {
      auto s2 = df.Define("y", [] { return 1; })
                   .Vary({"x", "y"}, "ROOT::RVec<ROOT::RVecD>{{x*0.1}, {y*0.1}}", 1, "broken")
                   .Sum("y");
      auto ss2 = VariationsFor(s2);
      // before starting the event loop, we jit and notice the mismatch in types
      EXPECT_THROW(
         try { ss2["nominal"]; } catch (const std::runtime_error &err) {
            const auto msg = "RVariationReader: type mismatch: column \"y\" is being used as int but the Define "
                             "or Vary node advertises it as double";
            EXPECT_STREQ(err.what(), msg);
            throw;
         },
         std::runtime_error);
   }

   {
      auto d2 = df.Define("z", "42");

      // Jitted Define, non-jitted Vary with incompatible type
      EXPECT_THROW(
         try {
            d2.Vary(
               "z",
               [] {
                  return ROOT::RVecF{-1.f, 2.f};
               },
               {}, 2, "broken");
         } catch (const std::runtime_error &err) {
            const auto expected =
               "Varied values for column \"z\" have a different type (float) than the nominal value (int).";
            EXPECT_STREQ(err.what(), expected);
            throw;
         },
         std::runtime_error);

      // Jitted Define, jitted Vary with incompatible type
      auto s = d2.Vary("z", "ROOT::RVecF{-1.f, 2.f}", 2, "broken").Sum<int>("z");
      auto ss = ROOT::RDF::Experimental::VariationsFor(s);
      EXPECT_THROW(
         try { ss["broken:0"]; } catch (const std::runtime_error &err) {
            const auto expected = "RVariationReader: type mismatch: column \"z\" is being used as int but the Define "
                                  "or Vary node advertises it as float";
            EXPECT_STREQ(err.what(), expected);
            throw;
         },
         std::runtime_error);
   }
}
#endif
#endif

TEST(RDFVary, RequireReturnTypeIsRVec)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   EXPECT_THROW(
      try { df.Vary("x", "0", /*nVariations=*/2); } catch (const std::runtime_error &err) {
         const auto msg = "Jitted Vary expressions must return an RVec object. "
                          "The following expression returns a int instead:\n0";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::runtime_error);
}

TEST(RDFVary, RequireNVariationsIsConsistent)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });
   auto s = df.Vary("x", SimpleVariation, {}, /*wrong=*/3).Sum<int>("x");
   auto all_s = VariationsFor(s);

   std::cerr << std::flush;
   std::streambuf *oldCerrStreamBuf = std::cerr.rdbuf();
   std::ostringstream strCerr;
   std::cerr.rdbuf(strCerr.rdbuf());

   // now, when evaluating `SimpleVariation`, we should notice that it returns 2 values, not 3, and throw
   EXPECT_THROW(
      try { all_s["nominal"]; } catch (const std::runtime_error &err) {
         const auto msg =
            "The evaluation of the expression for variation \"x\" resulted in 2 values, but 3 were expected.";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::runtime_error);

   std::cerr.rdbuf(oldCerrStreamBuf);
   EXPECT_EQ(strCerr.str(), "RDataFrame::Run: event loop was interrupted\n");
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
   auto ss = VariationsFor(s);
   EXPECT_EQ(ss["nominal"], 1 * 10);
   EXPECT_EQ(ss["x:0"], -1 * 10);
   EXPECT_EQ(ss["x:1"], 2 * 10);
}

TEST(RDFVary, SaveGraph)
{
   ROOT::RDataFrame df(1);
   auto c = df.Define("x", [] { return 0; })
               .Vary(
                  "x",
                  [] {
                     return ROOT::RVecI{0, 0};
                  },
                  {}, 2)
               .Count();
   auto cs = VariationsFor(c);
   const auto s = ROOT::RDF::SaveGraph(df);

   // `c` does not depend on `x`, so we don't expect any varied action in the output
   // (at the moment, `Vary` calls are not displayed)
   EXPECT_EQ(
      s,
      "digraph {\n\t1 [label=<Count>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n\t2 "
      "[label=<Define<BR/>x>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n\t0 [label=<Empty "
      "source<BR/>Entries: 1>, style=\"filled\", fillcolor=\"#f4b400\", shape=\"ellipse\"];\n\t2 -> 1;\n\t0 -> 2;\n}");
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

TEST_P(RDFVary, SimpleHistoWithAxes) // uses FillHelper instead of DelayedFillHelper
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

TEST_P(RDFVary, FilterAfterJittedFilter)
{
   auto c = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Filter("x > 1")
               .Filter([](int x) { return x > 5; }, {"x"})
               .Count();
   auto cs = ROOT::RDF::Experimental::VariationsFor(c);
   EXPECT_EQ(*c, 4);
   EXPECT_EQ(cs["nominal"], *c);
   EXPECT_EQ(cs["x:0"], 3);
   EXPECT_EQ(cs["x:1"], 5);
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

// If VariationsFor is called after the nominal result has already been produced/filled, the copies
// of the result object used to produced the varied results have to be reset to an empty/initial state.
// Here we test that this is the case for histograms and TStatistic objects.
TEST_P(RDFVary, FillHelperResets)
{
   auto df = ROOT::RDataFrame(10).Define("x", [] { return 1; });

   auto h = df.Vary("x", SimpleVariation, {}, 2).Histo1D<int>({"", "", 20, -10, 10}, "x");
   auto hs1 = VariationsFor(h);
   EXPECT_EQ(hs1["x:0"].GetMean(), -1);
   EXPECT_EQ(hs1["x:1"].GetMean(), 2);
   auto hs2 = VariationsFor(h);
   EXPECT_EQ(hs2["x:0"].GetMean(), -1);
   EXPECT_EQ(hs2["x:1"].GetMean(), 2);

   auto s = df.Vary("x", SimpleVariation, {}, 2).Stats<int>("x");
   auto ss1 = VariationsFor(s);
   EXPECT_EQ(ss1["x:0"].GetMean(), -1);
   EXPECT_EQ(ss1["x:1"].GetMean(), 2);
   auto ss2 = VariationsFor(s);
   EXPECT_EQ(ss2["x:0"].GetMean(), -1);
   EXPECT_EQ(ss2["x:1"].GetMean(), 2);
}

TEST(RDFVary, WithRange) // no Range in multithreaded runs
{
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 5, x + 5};
                  },
                  {"x"}, 2)
               .Range(7)
               .Filter("x > 1")
               .Range(3)
               .Sum<int>("x");
   auto hs = VariationsFor(h);

   EXPECT_EQ(*h, 9);
   EXPECT_EQ(hs["nominal"], 9);
   EXPECT_EQ(hs["x:0"], 0);
   EXPECT_EQ(hs["x:1"], 18);
}

TEST_P(RDFVary, VaryRedefine)
{
   // first redefine and then vary
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Redefine("x", [] { return 25; })
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Sum<int>("x");
   auto hs = VariationsFor(h);

   EXPECT_EQ(*h, 250);
   EXPECT_EQ(hs["nominal"], 250);
   EXPECT_EQ(hs["x:0"], 240);
   EXPECT_EQ(hs["x:1"], 260);
}

TEST_P(RDFVary, RedefineVariedColumn)
{
   // first vary and then redefine --> not legal
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2);
   EXPECT_THROW(
      try { h.Redefine("x", [] { return 25; }); } catch (const std::runtime_error &err) {
         const auto msg = "RDataFrame::Redefine: cannot redefine column \"x\". "
                          "The column depends on one or more systematic variations "
                          "and re-defining varied columns is not supported.";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::runtime_error);
}

TEST_P(RDFVary, VaryAggregate)
{
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Aggregate(std::plus<int>{}, std::plus<int>{}, "x");
   auto hs = VariationsFor(h);

   EXPECT_EQ(*h, 45);
   EXPECT_EQ(hs["nominal"], 45);
   EXPECT_EQ(hs["x:0"], 35);
   EXPECT_EQ(hs["x:1"], 55);
}

struct MyCounter : public ROOT::Detail::RDF::RActionImpl<MyCounter> {
   std::shared_ptr<int> fFinalResult = std::make_shared<int>(0);
   std::vector<int> fPerThreadResults;
   using Result_t = int;

   MyCounter(unsigned int nSlots) : fPerThreadResults(nSlots) {}

   MyCounter(const std::shared_ptr<int> &myc, const unsigned int nSlots) : fFinalResult(myc), fPerThreadResults(nSlots)
   {
   }

   std::shared_ptr<int> GetResultPtr() const { return fFinalResult; }

   void Initialize() {}

   void InitTask(TTreeReader *, int) {}

   void Exec(unsigned int slot) { fPerThreadResults[slot]++; }

   void Finalize() { *fFinalResult = std::accumulate(fPerThreadResults.begin(), fPerThreadResults.end(), 0); }

   std::string GetActionName() const { return "MyCounter"; }

   MyCounter MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<int> *>(newResult);
      return MyCounter(result, fPerThreadResults.size());
   }
};

TEST(RDFVary, VaryBook)
{
   auto d = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 5, x + 5};
                  },
                  {"x"}, 2);
   auto h = d.Filter([](int x) { return x > 7; }, {"x"}).Book<>(MyCounter{d.GetNSlots()}, {});
   auto hs = VariationsFor(h);
   EXPECT_EQ(*h, 2);
   EXPECT_EQ(hs["nominal"], 2);
   EXPECT_EQ(hs["x:0"], 0);
   EXPECT_EQ(hs["x:1"], 7);

   auto g = d.Filter([](int x) { return x < 8; }, {"x"}).Book<>(MyCounter{d.GetNSlots()}, {});
   auto gs = VariationsFor(g);
   EXPECT_EQ(*g, 8);
   EXPECT_EQ(gs["nominal"], 8);
   EXPECT_EQ(gs["x:0"], 10);
   EXPECT_EQ(gs["x:1"], 3);
}

struct CounterWithoutVariations : public ROOT::Detail::RDF::RActionImpl<CounterWithoutVariations> {
   std::shared_ptr<int> fFinalResult = std::make_shared<int>(0);
   std::vector<int> fPerThreadResults;
   using Result_t = int;

   CounterWithoutVariations(unsigned int nSlots) : fPerThreadResults(nSlots) {}

   std::shared_ptr<int> GetResultPtr() const { return fFinalResult; }

   void Initialize() {}

   void InitTask(TTreeReader *, int) {}

   void Exec(unsigned int slot) { fPerThreadResults[slot]++; }

   void Finalize() { *fFinalResult = std::accumulate(fPerThreadResults.begin(), fPerThreadResults.end(), 0); }

   std::string GetActionName() const { return "CounterWithoutVariations"; }
};

TEST_P(RDFVary, VaryClassWithoutMakeNew)
{
   auto d = ROOT::RDataFrame(10)
               .Define("x", [] { return 25; })
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 5, x + 5};
                  },
                  {"x"}, 2);
   auto h = d.Filter([](int x) { return x > 24; }, {"x"}).Book<>(CounterWithoutVariations{10u}, {});
   EXPECT_EQ(*h, 10);
   EXPECT_THROW(
      try { VariationsFor(h); } catch (const std::logic_error &err) {
         const auto msg = "The MakeNew method is not implemented for this action helper (CounterWithoutVariations). "
                          "Cannot Vary its result.";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::logic_error);
}

TEST_P(RDFVary, VaryCache)
{
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Cache<int>({"x"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Sum<int>("x");
   auto hs = VariationsFor(h);

   EXPECT_EQ(*h, 45);
   EXPECT_EQ(hs["nominal"], 45);
   EXPECT_EQ(hs["x:0"], 35);
   EXPECT_EQ(hs["x:1"], 55);
}

TEST_P(RDFVary, VaryCount)
{
   auto h = ROOT::RDataFrame(3)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Filter([](int x) { return x > 1; }, {"x"})
               .Count();
   auto hs = VariationsFor(h);
   EXPECT_EQ(*h, 1);
   EXPECT_EQ(hs["nominal"], 1);
   EXPECT_EQ(hs["x:0"], 0);
   EXPECT_EQ(hs["x:1"], 2);
}

// must update this test when https://github.com/root-project/root/issues/9894 is addressed
TEST(RDFVary, VaryDisplay) // TEST instead of TEST_P because Display is single-thread only
{
   auto d = ROOT::RDataFrame(1)
               .Define("x", [] { return 0; })
               .Vary(
                  "x",
                  [] {
                     return ROOT::RVecI{-1, 2};
                  },
                  {}, 2)
               .Display<int>({"x"});
   // Display ignores variations, only displays the nominal values
   EXPECT_EQ(d->AsString(), "+-----+---+\n| Row | x | \n+-----+---+\n| 0   | 0 | \n|     |   | \n+-----+---+\n");
   // cannot vary a Display
   EXPECT_THROW(
      try { VariationsFor(d); } catch (const std::logic_error &err) {
         const auto msg = "The MakeNew method is not implemented for this action helper (Display). "
                          "Cannot Vary its result.";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::logic_error);
}

struct Jet {
   double a, b;
};

struct CustomFiller {

   TH2D h{"", "", 10, 0, 10, 10, 0, 10};

   void Fill(const Jet &j) { h.Fill(j.a, j.b); }

   void Merge(const std::vector<CustomFiller *> &jets)
   {
      TList l;
      for (auto *j : jets)
         l.Add(&j->h);
      h.Merge(&l);
   }

   // needed for VariationsFor
   void Reset() { h.Reset(); }

   double GetMeanX() const { return h.GetMean(1); }
   double GetMeanY() const { return h.GetMean(2); }
};

TEST_P(RDFVary, VaryCustomObject)
{
   auto h = ROOT::RDataFrame(10)
               .Define("Jet",
                       [] {
                          return Jet{1., 2.};
                       })
               .Vary(
                  "Jet",
                  [] {
                     return ROOT::RVec<Jet>{{}, {4., 5.}};
                  },
                  {}, 2)
               .Fill<Jet>(CustomFiller{}, {"Jet"});
   auto hs = VariationsFor(h);

   EXPECT_DOUBLE_EQ(h->GetMeanX(), 1.);
   EXPECT_DOUBLE_EQ(h->GetMeanY(), 2.);
   EXPECT_DOUBLE_EQ(hs["nominal"].GetMeanX(), 1.);
   EXPECT_DOUBLE_EQ(hs["nominal"].GetMeanY(), 2.);
   EXPECT_DOUBLE_EQ(hs["Jet:0"].GetMeanX(), 0.);
   EXPECT_DOUBLE_EQ(hs["Jet:0"].GetMeanY(), 0.);
   EXPECT_DOUBLE_EQ(hs["Jet:1"].GetMeanX(), 4.);
   EXPECT_DOUBLE_EQ(hs["Jet:1"].GetMeanY(), 5.);
}

TEST_P(RDFVary, VaryFill)
{
   SimpleFiller sf;
   auto h = ROOT::RDataFrame(10)
               .Define("x", [] { return 42.; })
               .Vary(
                  "x",
                  [](double x) {
                     return ROOT::RVecD{x - 1., x + 1.};
                  },
                  {"x"}, 2)
               .Fill<double>(sf, {"x"});
   auto hs = VariationsFor(h);

   EXPECT_DOUBLE_EQ(h->GetHisto().GetMean(), 42.);
   EXPECT_DOUBLE_EQ(hs["nominal"].GetHisto().GetMean(), 42.);
   EXPECT_DOUBLE_EQ(hs["x:0"].GetHisto().GetMean(), 41.);
   EXPECT_DOUBLE_EQ(hs["x:1"].GetHisto().GetMean(), 43.);
}

TEST_P(RDFVary, VaryGraph)
{
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Graph<int, int>("x", "x");
   auto hs = VariationsFor(h);

   EXPECT_DOUBLE_EQ(h->GetMean(), 4.5);
   EXPECT_DOUBLE_EQ(hs["nominal"].GetMean(), 4.5);
   EXPECT_DOUBLE_EQ(hs["x:0"].GetMean(), 3.5);
   EXPECT_DOUBLE_EQ(hs["x:1"].GetMean(), 5.5);
}

TEST_P(RDFVary, VaryGraphAsymmErrors)
{
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .GraphAsymmErrors<int, int, int, int, int, int>("x", "x", "x", "x", "x", "x");
   auto hs = VariationsFor(h);

   EXPECT_DOUBLE_EQ(h->GetMean(), 4.5);
   EXPECT_DOUBLE_EQ(hs["nominal"].GetMean(), 4.5);
   EXPECT_DOUBLE_EQ(hs["x:0"].GetMean(), 3.5);
   EXPECT_DOUBLE_EQ(hs["x:1"].GetMean(), 5.5);
}

TEST_P(RDFVary, VaryHistos)
{
   auto df = ROOT::RDataFrame(10)
                .Define("x",
                        [] {
                           return ROOT::RVecI{1, 2, 3};
                        })
                .Vary(
                   "x",
                   [] {
                      return ROOT::RVec<ROOT::RVecI>{{}, {4, 5}};
                   },
                   {}, 2);
   auto h1 = df.Histo1D<ROOT::RVecI>("x");
   auto h1s = VariationsFor(h1);
   auto h2 = df.Histo2D<ROOT::RVecI, ROOT::RVecI>({}, "x", "x");
   auto h2s = VariationsFor(h2);
   auto h3 = df.Histo3D<ROOT::RVecI, ROOT::RVecI, ROOT::RVecI>({}, "x", "x", "x");
   auto h3s = VariationsFor(h3);

   EXPECT_DOUBLE_EQ(h1->GetMean(), 2.);
   EXPECT_DOUBLE_EQ(h2->GetMean(), 2.);
   EXPECT_DOUBLE_EQ(h3->GetMean(), 2.);

   EXPECT_DOUBLE_EQ(h1s["nominal"].GetMean(), 2.);
   EXPECT_DOUBLE_EQ(h2s["nominal"].GetMean(), 2.);
   EXPECT_DOUBLE_EQ(h3s["nominal"].GetMean(), 2.);

   EXPECT_DOUBLE_EQ(h1s["x:0"].GetMean(), 0.);
   EXPECT_DOUBLE_EQ(h2s["x:0"].GetMean(), 0.);
   EXPECT_DOUBLE_EQ(h3s["x:0"].GetMean(), 0.);

   EXPECT_DOUBLE_EQ(h1s["x:1"].GetMean(), 4.5);
   EXPECT_DOUBLE_EQ(h2s["x:1"].GetMean(), 4.5);
   EXPECT_DOUBLE_EQ(h3s["x:1"].GetMean(), 4.5);

   int nbins[4] = {10, 10, 10, 10};
   double xmin[4] = {0., 0., 0., 0.};
   double xmax[4] = {100., 100., 100., 100.};
   auto hN = df.HistoND<ROOT::RVecI, ROOT::RVecI, ROOT::RVecI, ROOT::RVecI>({"", "", 4, nbins, xmin, xmax},
                                                                            {"x", "x", "x", "x"});
   auto hNs = VariationsFor(hN);

   auto res = hN->Projection(3);
   EXPECT_DOUBLE_EQ(res->GetMean(), 5.);
   delete res;
   res = hNs["nominal"].Projection(3);
   EXPECT_DOUBLE_EQ(res->GetMean(), 5.);
   delete res;
   res = hNs["x:0"].Projection(3);
   EXPECT_DOUBLE_EQ(res->GetMean(), 0.);
   delete res;
   res = hNs["x:1"].Projection(3);
   EXPECT_DOUBLE_EQ(res->GetMean(), 5.);
   delete res;
}

TEST_P(RDFVary, VaryMax)
{
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Max<int>("x");
   auto hs = VariationsFor(h);

   EXPECT_EQ(*h, 9);
   EXPECT_EQ(hs["nominal"], 9);
   EXPECT_EQ(hs["x:0"], 8);
   EXPECT_EQ(hs["x:1"], 10);
}

TEST_P(RDFVary, VaryMean)
{
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Mean<int>("x");
   auto hs = VariationsFor(h);

   EXPECT_DOUBLE_EQ(*h, 4.5);
   EXPECT_DOUBLE_EQ(hs["nominal"], 4.5);
   EXPECT_DOUBLE_EQ(hs["x:0"], 3.5);
   EXPECT_DOUBLE_EQ(hs["x:1"], 5.5);
}

TEST_P(RDFVary, VaryMin)
{
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Min<int>("x");
   auto hs = VariationsFor(h);

   EXPECT_EQ(*h, 0);
   EXPECT_EQ(hs["nominal"], 0);
   EXPECT_EQ(hs["x:0"], -1);
   EXPECT_EQ(hs["x:1"], 1);
}

TEST_P(RDFVary, VaryProfiles)
{
   auto df = ROOT::RDataFrame(10)
                .Define("x",
                        [] {
                           return ROOT::RVecI{1, 2, 3};
                        })
                .Vary(
                   "x",
                   [] {
                      return ROOT::RVec<ROOT::RVecI>{{}, {4, 5}};
                   },
                   {}, 2);
   auto h1 = df.Profile1D<ROOT::RVecI, ROOT::RVecI>({"", "", 100, 0, 100, 0, 100}, "x", "x");
   auto h1s = VariationsFor(h1);
   auto h2 = df.Profile2D<ROOT::RVecI, ROOT::RVecI, ROOT::RVecI>({"", "", 100, 0, 100, 100, 0, 100}, "x", "x", "x");
   auto h2s = VariationsFor(h2);

   EXPECT_DOUBLE_EQ(h1->GetMean(), 2.);
   EXPECT_DOUBLE_EQ(h2->GetMean(), 2.);

   EXPECT_DOUBLE_EQ(h1s["nominal"].GetMean(), 2.);
   EXPECT_DOUBLE_EQ(h2s["nominal"].GetMean(), 2.);

   EXPECT_DOUBLE_EQ(h1s["x:0"].GetMean(), 0.);
   EXPECT_DOUBLE_EQ(h2s["x:0"].GetMean(), 0.);

   EXPECT_DOUBLE_EQ(h1s["x:1"].GetMean(), 4.5);
   EXPECT_DOUBLE_EQ(h2s["x:1"].GetMean(), 4.5);
}

TEST(RDFVary, VaryReduce)
{
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Reduce([](int x, int y) { return x + y; }, "x", 0);
   auto hs = VariationsFor(h);
   EXPECT_EQ(*h, 45);
   EXPECT_EQ(hs["nominal"], 45);
   EXPECT_EQ(hs["x:0"], 35);
   EXPECT_EQ(hs["x:1"], 55);
}

// Varying Reports is not implemented yet, tracked by https://github.com/root-project/root/issues/10551
TEST_P(RDFVary, VaryReport)
{
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Filter([](int x) { return x > 5; }, {"x"}, "before")
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Filter([](int x) { return x > 7; }, {"x"}, "after")
               .Report();
   auto &report = *h;

   EXPECT_EQ(report["before"].GetAll(), 10);
   EXPECT_FLOAT_EQ(report["before"].GetEff(), 40.);
   EXPECT_EQ(report["before"].GetPass(), 4);
   EXPECT_EQ(report["after"].GetAll(), 4);
   EXPECT_FLOAT_EQ(report["after"].GetEff(), 50.);
   EXPECT_EQ(report["after"].GetPass(), 2);
   EXPECT_THROW(
      try { VariationsFor(h); } catch (const std::logic_error &err) {
         const auto msg = "The MakeNew method is not implemented for this action helper (Report). "
                          "Cannot Vary its result.";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::logic_error);
}

TEST_P(RDFVary, VaryStdDev)
{
   auto h = ROOT::RDataFrame(3)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{1, x + 1};
                  },
                  {"x"}, 2)
               .StdDev<int>("x");
   auto hs = VariationsFor(h);

   EXPECT_DOUBLE_EQ(*h, 1.);
   EXPECT_DOUBLE_EQ(hs["nominal"], 1.);
   EXPECT_DOUBLE_EQ(hs["x:0"], 0.);
   EXPECT_DOUBLE_EQ(hs["x:1"], 1.);
}

TEST_P(RDFVary, VarySum)
{
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Sum<int>("x");
   auto hs = VariationsFor(h);

   EXPECT_EQ(*h, 45);
   EXPECT_EQ(hs["nominal"], 45);
   EXPECT_EQ(hs["x:0"], 35);
   EXPECT_EQ(hs["x:1"], 55);
}

TEST_P(RDFVary, VaryTake)
{
   auto sorted = [](const std::vector<int> &v) {
      std::vector<int> r(v);
      std::sort(r.begin(), r.end());
      return r;
   };

   auto r = ROOT::RDataFrame(3)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Take<int>("x");
   EXPECT_EQ(sorted(*r), std::vector<int>({0, 1, 2}));

   auto rs = VariationsFor(r);
   EXPECT_EQ(sorted(rs["nominal"]), std::vector<int>({0, 1, 2}));
   EXPECT_EQ(sorted(rs["x:0"]), std::vector<int>({-1, 0, 1}));
   EXPECT_EQ(sorted(rs["x:1"]), std::vector<int>({1, 2, 3}));
}

TEST_P(RDFVary, VarySnapshot)
{
   const auto fname = "dummy.root";
   auto h = ROOT::RDataFrame(10)
               .Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Vary(
                  "x",
                  [](int x) {
                     return ROOT::RVecI{x - 1, x + 1};
                  },
                  {"x"}, 2)
               .Snapshot<int>("t", fname, {"x"});
   EXPECT_THROW(
      try { VariationsFor(h); } catch (const std::logic_error &err) {
         const auto msg = "The MakeNew method is not implemented for this action helper (Snapshot). "
                          "Cannot Vary its result.";
         EXPECT_STREQ(err.what(), msg);
         throw;
      },
      std::logic_error);
}

// this is a regression test, we used to read from wrong addresses in this case
TEST_P(RDFVary, MoreVariedColumnsThanVariations)
{
   auto d = ROOT::RDataFrame(10)
               .Define("x", [] { return 0; })
               .Define("y", [] { return 0; })
               .Vary(
                  {"x", "y"},
                  [] {
                     return ROOT::RVec<ROOT::RVecI>{{1}, {2}};
                  },
                  {}, 1, "syst");
   auto h = d.Sum<int>("y");
   auto hs = ROOT::RDF::Experimental::VariationsFor(h);

   EXPECT_EQ(hs["syst:0"], 20);
}

// this is a regression test for an issue that was hidden by RVec's small buffer optimization
// when the variations don't fit in the smalll buffer and we are varying multiple columns simultaneously,
// RVariation was changing the address of the varied values between entries, resulting in invalid reads
// on the part of the RVariationReader.
TEST_P(RDFVary, ManyVariationsManyColumns)
{
   auto d = ROOT::RDataFrame(10)
               .Define("x", [] { return 0; })
               .Define("y", [] { return 0; })
               .Vary(
                  {"x", "y"},
                  [] {
                     return ROOT::RVec<ROOT::RVecI>{ROOT::RVecI(100, 42), ROOT::RVecI(100, 8)};
                  },
                  {}, 100, "syst");

   auto sx = d.Sum<int>("x");
   auto sxs = ROOT::RDF::Experimental::VariationsFor(sx);
   auto sy = d.Sum<int>("y");
   auto sys = ROOT::RDF::Experimental::VariationsFor(sy);

   for (int i = 0; i < 100; ++i) {
      EXPECT_EQ(sxs["syst:" + std::to_string(i)], 420);
      EXPECT_EQ(sys["syst:" + std::to_string(i)], 80);
   }
}

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, RDFVary, ::testing::Values(false));

// instantiate multi-thread tests
#ifdef R__USE_IMT
INSTANTIATE_TEST_SUITE_P(MT, RDFVary, ::testing::Values(true));
#endif
