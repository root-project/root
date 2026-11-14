// Regression tests for observable-based bin volumes and version-1 persistence.
#include <RooBinWidthFunction.h>
#include <RooBinning.h>
#include <RooCategory.h>
#include <RooConstVar.h>
#include <RooDataHist.h>
#include <RooFit/Detail/NormalizationHelpers.h>
#include <RooFit/Evaluator.h>
#include <RooFormulaVar.h>
#include <RooHistFunc.h>
#include <RooLinearVar.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <TBufferFile.h>
#include <TFile.h>
#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace {
const double edges[] = {0., 1., 3., 6.};

// Count width computations independently of the function's cache implementation.
class CountingBinning : public RooBinning {
public:
   explicit CountingBinning(int &calls) : RooBinning(3, edges), _calls(&calls) {}
   RooAbsBinning *clone(const char * = nullptr) const override { return new CountingBinning(*this); }
   double binWidth(int bin) const override
   {
      ++*_calls;
      return RooBinning::binWidth(bin);
   }

private:
   int *_calls;
};

class CountingFunction : public RooBinWidthFunction {
public:
   using RooBinWidthFunction::RooBinWidthFunction;
   mutable int calls = 0;
   double evaluate() const override
   {
      ++calls;
      return RooBinWidthFunction::evaluate();
   }
};

struct RestoreEnabled {
   bool enabled = RooBinWidthFunction::isClassEnabled();
   ~RestoreEnabled()
   {
      if (enabled)
         RooBinWidthFunction::enableClass();
      else
         RooBinWidthFunction::disableClass();
   }
};

void checkWorkspace(RooWorkspace &ws)
{
   auto *volume = dynamic_cast<RooBinWidthFunction *>(ws.function("volume"));
   auto *inverse = dynamic_cast<RooBinWidthFunction *>(ws.function("inverse"));
   ASSERT_NE(volume, nullptr);
   ASSERT_NE(inverse, nullptr);
   ASSERT_EQ(volume->variables().size(), 2u);
   ASSERT_EQ(inverse->variables().size(), 2u);
   EXPECT_EQ(volume->variables().find("x"), ws.var("x"));
   EXPECT_EQ(volume->variables().find("y"), ws.var("y"));
   EXPECT_EQ(volume->servers().size(), 2u);
   EXPECT_EQ(volume->numProxies(), 1);
   for (int i = 0; i < 3; ++i) {
      ws.var("x")->setVal((edges[i] + edges[i + 1]) / 2.);
      for (double y : {2., 8.}) {
         ws.var("y")->setVal(y);
         double expected = (edges[i + 1] - edges[i]) * 5.;
         EXPECT_DOUBLE_EQ(volume->getVal(), expected);
         EXPECT_DOUBLE_EQ(inverse->getVal(), 1. / expected);
      }
   }
}
} // namespace

TEST(RooBinWidthFunction, VolumeAndInverse)
{
   RooRealVar x{"x", "x", 0., 6.};
   RooRealVar y{"y", "y", 0., 10.};
   RooRealVar z{"z", "z", -4., 4.};
   x.setBinning(RooBinning{3, edges});
   y.setBins(2);
   z.setBins(4);
   RooDataHist data{"data", "data", {x, y, z}};
   RooHistFunc hist{"hist", "hist", {x, y, z}, data};
   for (bool inverse : {false, true}) {
      RooBinWidthFunction volume{"volume", "volume", RooArgList{x, y, z}, inverse};
      RooBinWidthFunction legacy{"legacy", "legacy", hist, inverse};
      for (int i = 0; i < data.numEntries(); ++i) {
         const auto *row = data.get(i);
         x.setVal(row->getRealValue("x"));
         y.setVal(row->getRealValue("y"));
         z.setVal(row->getRealValue("z"));
         double expected = data.binVolume();
         if (inverse)
            expected = 1. / expected;
         EXPECT_DOUBLE_EQ(volume.getVal(), expected);
         EXPECT_DOUBLE_EQ(legacy.getVal(), expected);
      }
   }
}

TEST(RooBinWidthFunction, LiveBinningAndHistogramLifetime)
{
   RooRealVar x{"x", "x", 0., 6.};
   x.setBins(2);
   std::unique_ptr<RooBinWidthFunction> volume;
   {
      RooDataHist data{"data", "data", x};
      RooHistFunc hist{"hist", "hist", x, data};
      x.setBinning(RooBinning{3, edges});
      volume = std::make_unique<RooBinWidthFunction>("volume", "volume", hist, false);
      EXPECT_FALSE(volume->dependsOn(hist));
   }
   for (int i = 0; i < 3; ++i) {
      x.setBin(i);
      EXPECT_DOUBLE_EQ(volume->getVal(), edges[i + 1] - edges[i]);
   }
}

TEST(RooBinWidthFunction, BatchBoundariesAndScalarBroadcast)
{
   RooRealVar x{"x", "x", 0., 6.};
   RooRealVar y{"y", "y", 0., 10.};
   x.setBinning(RooBinning{3, edges});
   y.setBins(2);
   const std::vector<double> values{-0.1, 0., 0.5, 1., 2., 3., 5., 6., 6.1};
   const std::vector<double> expected{1., 5., 5., 10., 10., 15., 15., 15., 1.};
   for (bool inverse : {false, true}) {
      RooBinWidthFunction volume{"volume", "volume", RooArgList{x, y}, inverse};
      RooFit::Evaluator evaluator{volume};
      evaluator.setInput("x", values, false);
      auto output = evaluator.run();
      ASSERT_EQ(output.size(), values.size());
      for (std::size_t i = 0; i < values.size(); ++i) {
         EXPECT_DOUBLE_EQ(output[i], inverse ? 1. / expected[i] : expected[i]);
         if (values[i] >= 0. && values[i] <= 6.) {
            x.setVal(values[i]);
            EXPECT_DOUBLE_EQ(volume.evaluate(), output[i]);
         }
      }
      // Multiple out-of-range dimensions must still produce one result/event.
      const std::vector<double> yValues(values.size(), 11.);
      evaluator.setInput("y", yValues, false);
      for (double result : evaluator.run())
         EXPECT_DOUBLE_EQ(result, 1.);
   }
}

TEST(RooBinWidthFunction, Categories)
{
   RooRealVar x{"x", "x", 0., 6.};
   x.setBinning(RooBinning{3, edges});
   RooCategory category{"category", "category"};
   category.defineType("a", 2);
   category.defineType("b", 7);
   RooBinWidthFunction volume{"volume", "volume", RooArgList{category, x}, false};
   for (int state : {2, 7}) {
      category.setIndex(state);
      x.setVal(2.);
      EXPECT_DOUBLE_EQ(volume.getVal(), 2.);
   }
   RooFit::Evaluator evaluator{volume};
   const std::vector<double> states{2., 7., 2.};
   const std::vector<double> values{0.5, 2., 4.};
   evaluator.setInput("category", states, false);
   evaluator.setInput("x", values, false);
   auto output = evaluator.run();
   ASSERT_EQ(output.size(), 3u);
   for (std::size_t i = 0; i < 3; ++i)
      EXPECT_DOUBLE_EQ(output[i], i + 1.);
}

TEST(RooBinWidthFunction, Caches)
{
   int widthCalls = 0;
   RooRealVar x{"x", "x", 0., 6.};
   x.setBinning(CountingBinning{widthCalls});
   CountingFunction volume{"volume", "volume", RooArgList{x}, false};
   x.setVal(0.5);
   EXPECT_DOUBLE_EQ(volume.getVal(), 1.);
   const int initialWidthCalls = widthCalls;
   const int initialEvaluations = volume.calls;
   EXPECT_GT(initialWidthCalls, 0);
   EXPECT_DOUBLE_EQ(volume.getVal(), 1.);
   EXPECT_EQ(volume.calls, initialEvaluations);
   x.setVal(2.);
   EXPECT_DOUBLE_EQ(volume.getVal(), 2.);
   EXPECT_GT(volume.calls, initialEvaluations);
   EXPECT_EQ(widthCalls, initialWidthCalls);
   x.setRange(0., 3.);
   x.setVal(0.5);
   EXPECT_DOUBLE_EQ(volume.getVal(), 1.);
   EXPECT_GT(widthCalls, initialWidthCalls);
}

TEST(RooBinWidthFunction, ReplacingBinningInvalidatesCachedValue)
{
   RooRealVar x{"x", "x", 0., 6.};
   x.setBinning(RooBinning{3, edges});
   RooBinWidthFunction volume{"volume", "volume", RooArgList{x}, false};
   x.setVal(0.5);
   EXPECT_DOUBLE_EQ(volume.getVal(), 1.);
   // No change to x's value, and setBinning itself does not propagate dirtiness.
   x.setBins(2);
   EXPECT_DOUBLE_EQ(volume.getVal(), 3.);
}

TEST(RooBinWidthFunction, CloneAndRedirect)
{
   RooRealVar x{"x", "x", 0., 6.};
   x.setBinning(RooBinning{3, edges});
   RooBinWidthFunction volume{"volume", "volume", RooArgList{x}, false};
   x.setVal(2.);
   EXPECT_DOUBLE_EQ(volume.getVal(), 2.);
   std::unique_ptr<RooBinWidthFunction> copy{static_cast<RooBinWidthFunction *>(volume.clone("copy"))};
   RooRealVar replacement{"x", "x", 0., 6.};
   replacement.setBins(2);
   copy->redirectServers(RooArgSet{replacement});
   EXPECT_DOUBLE_EQ(copy->getVal(), 3.);
   EXPECT_EQ(copy->variables().find("x"), &replacement);
   EXPECT_DOUBLE_EQ(volume.getVal(), 2.);
}

TEST(RooBinWidthFunction, PlottingHints)
{
   RooRealVar x{"x", "x", 0., 6.};
   RooRealVar unrelated{"unrelated", "unrelated", 0., 6.};
   x.setBinning(RooBinning{3, edges});
   RooDataHist data{"data", "data", x};
   RooHistFunc hist{"hist", "hist", x, data};
   RooBinWidthFunction volume{"volume", "volume", RooArgList{x}, false};
   EXPECT_TRUE(volume.isBinnedDistribution(RooArgSet{x}));
   std::unique_ptr<std::list<double>> boundaries{volume.binBoundaries(x, 0., 6.)};
   ASSERT_NE(boundaries, nullptr);
   EXPECT_EQ(std::vector<double>(boundaries->begin(), boundaries->end()), std::vector<double>(edges, edges + 4));
   std::unique_ptr<std::list<double>> hint{volume.plotSamplingHint(x, 0., 6.)};
   std::unique_ptr<std::list<double>> reference{hist.plotSamplingHint(x, 0., 6.)};
   ASSERT_NE(hint, nullptr);
   ASSERT_NE(reference, nullptr);
   EXPECT_EQ(*hint, *reference);
   EXPECT_EQ(volume.binBoundaries(unrelated, 0., 6.), nullptr);
   EXPECT_EQ(volume.plotSamplingHint(unrelated, 0., 6.), nullptr);
}

TEST(RooBinWidthFunction, DisabledAndBinnedLikelihood)
{
   RestoreEnabled restore;
   RooRealVar x{"x", "x", 0., 6.};
   x.setBins(2);
   RooBinWidthFunction volume{"volume", "volume", RooArgList{x}, true};
   RooBinWidthFunction::disableClass();
   EXPECT_DOUBLE_EQ(volume.evaluate(), 1.);
   {
      RooFit::Evaluator evaluator{volume};
      const std::vector<double> values{1., 4.};
      evaluator.setInput("x", values, false);
      for (double result : evaluator.run())
         EXPECT_DOUBLE_EQ(result, 1.);
   }
   RooBinWidthFunction::enableClass();
   EXPECT_DOUBLE_EQ(volume.evaluate(), 1. / 3.);
   RooArgSet normSet{x};
   RooFit::Detail::CompileContext context{normSet};
   context.setBinnedLikelihoodMode(true);
   auto compiled = volume.compileForNormSet(normSet, context);
   EXPECT_DOUBLE_EQ(dynamic_cast<RooAbsReal &>(*compiled).getVal(), 1.);
   EXPECT_TRUE(context.binWidthFuncFlag());
}

TEST(RooBinWidthFunction, InputValidationAndTransformedBinning)
{
   RooRealVar x{"x", "x", 0., 6.};
   x.setBinning(RooBinning{3, edges});
   RooFormulaVar formula{"formula", "2*x", {x}};
   EXPECT_THROW((RooBinWidthFunction{"bad", "bad", RooArgList{formula}, false}), std::invalid_argument);
   EXPECT_THROW((RooBinWidthFunction{"bad", "bad", RooArgList{x, x}, false}), std::invalid_argument);
   RooBinWidthFunction empty{"empty", "empty", RooArgList{}, false};
   EXPECT_DOUBLE_EQ(empty.getVal(), 1.);
   RooRealVar slope{"slope", "slope", 2., 1., 4.};
   RooConstVar offset{"offset", "offset", 1.};
   RooLinearVar transformed{"transformed", "transformed", x, slope, offset};
   RooBinWidthFunction volume{"volume", "volume", RooArgList{transformed}, false};
   for (int i = 0; i < 3; ++i) {
      x.setBin(i);
      EXPECT_DOUBLE_EQ(volume.getVal(), 2. * (edges[i + 1] - edges[i]));
   }
   slope.setVal(3.);
   EXPECT_DOUBLE_EQ(volume.getVal(), 9.);
}

TEST(RooBinWidthFunction, WorkspaceRoundTrip)
{
   RooRealVar x{"x", "x", 0., 6.};
   RooRealVar y{"y", "y", 0., 10.};
   x.setBinning(RooBinning{3, edges});
   y.setBins(2);
   RooBinWidthFunction volume{"volume", "volume", RooArgList{x, y}, false};
   RooBinWidthFunction inverse{"inverse", "inverse", RooArgList{x, y}, true};
   RooWorkspace ws{"ws"};
   ws.import(volume);
   ws.import(inverse, RooFit::RecycleConflictNodes());
   checkWorkspace(ws);
   TBufferFile buffer{TBuffer::kWrite};
   buffer.WriteObject(&ws);
   buffer.SetReadMode();
   buffer.SetBufferOffset(0);
   std::unique_ptr<RooWorkspace> restored{static_cast<RooWorkspace *>(buffer.ReadObjectAny(RooWorkspace::Class()))};
   ASSERT_NE(restored, nullptr);
   checkWorkspace(*restored);
}

// Historical input must be generated with class version 1, not with the class
// under test. Save the following recipe as makeRooBinWidthFunction_v1.cxx and
// run it with a ROOT build containing that old version.
//
// // Run with ROOT containing RooBinWidthFunction class version 1:
// // root -l -b -q 'makeRooBinWidthFunction_v1.cxx("rooBinWidthFunction_v1.root")'
// #include <RooBinWidthFunction.h>
// #include <RooBinning.h>
// #include <RooDataHist.h>
// #include <RooHistFunc.h>
// #include <RooRealVar.h>
// #include <RooWorkspace.h>
// #include <TFile.h>
//
// #include <stdexcept>
//
// void makeRooBinWidthFunction_v1(const char *filename = "rooBinWidthFunction_v1.root")
// {
//    if (RooBinWidthFunction::Class_Version() != 1)
//       throw std::runtime_error("This fixture must be generated with RooBinWidthFunction version 1");
//
//    RooRealVar x{"x", "x", 0., 6.};
//    RooRealVar y{"y", "y", 0., 10.};
//    const double edges[] = {0., 1., 3., 6.};
//    x.setBinning(RooBinning{3, edges});
//    y.setBins(2);
//    RooDataHist data{"data", "data", {x, y}};
//    RooHistFunc hist{"hist", "hist", {x, y}, data};
//    RooBinWidthFunction volume{"volume", "volume", hist, false};
//    RooBinWidthFunction inverse{"inverse", "inverse", hist, true};
//    RooWorkspace ws{"ws"};
//    ws.import(volume);
//    ws.import(inverse, RooFit::RecycleConflictNodes());
//
//    // The live observable binning takes precedence after migration.
//    x.setBins(2);
//    RooBinWidthFunction different{"different", "different", hist, false};
//    RooWorkspace mismatch{"mismatch"};
//    mismatch.import(different);
//
//    TFile file{filename, "RECREATE"};
//    ws.Write();
//    mismatch.Write();
//    ws.function("volume")->Write("standalone");
// }
TEST(RooBinWidthFunction, SchemaEvolutionV1)
{
   TFile file{"rooBinWidthFunction_v1.root"};
   ASSERT_FALSE(file.IsZombie());
   auto *ws = file.Get<RooWorkspace>("ws");
   ASSERT_NE(ws, nullptr);
   checkWorkspace(*ws);
   // Importing just the migrated function must not pull in its old histogram.
   RooWorkspace clean{"clean"};
   clean.import(*ws->function("volume"));
   clean.import(*ws->function("inverse"), RooFit::RecycleConflictNodes());
   EXPECT_EQ(clean.function("hist"), nullptr);
   checkWorkspace(clean);
   TBufferFile buffer{TBuffer::kWrite};
   buffer.WriteObject(&clean);
   buffer.SetReadMode();
   buffer.SetBufferOffset(0);
   std::unique_ptr<RooWorkspace> restored{static_cast<RooWorkspace *>(buffer.ReadObjectAny(RooWorkspace::Class()))};
   ASSERT_NE(restored, nullptr);
   checkWorkspace(*restored);
   auto *mismatch = file.Get<RooWorkspace>("mismatch");
   ASSERT_NE(mismatch, nullptr);
   mismatch->var("x")->setVal(0.5);
   EXPECT_DOUBLE_EQ(mismatch->function("different")->getVal(), 15.);
   // Also support a function written as a top-level key, outside a workspace.
   auto *standalone = file.Get<RooBinWidthFunction>("standalone");
   ASSERT_NE(standalone, nullptr);
   ASSERT_EQ(standalone->variables().size(), 2u);
   auto &x = static_cast<RooRealVar &>(standalone->variables()[0]);
   x.setVal(0.5);
   EXPECT_DOUBLE_EQ(standalone->getVal(), 5.);
   x.setVal(4.);
   EXPECT_DOUBLE_EQ(standalone->getVal(), 15.);
}
