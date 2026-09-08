// Regression tests for RooBinWidthFunction evaluation, caching and persistence.
#include <RooBinWidthFunction.h>
#include <RooBinning.h>
#include <RooDataHist.h>
#include <RooFit/Detail/NormalizationHelpers.h>
#include <RooFit/Evaluator.h>
#include <RooHistFunc.h>
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
      RooBinWidthFunction volume{"volume", "volume", hist, inverse};
      for (int i = 0; i < data.numEntries(); ++i) {
         const auto *row = data.get(i);
         x.setVal(row->getRealValue("x"));
         y.setVal(row->getRealValue("y"));
         z.setVal(row->getRealValue("z"));
         double expected = data.binVolume();
         if (inverse)
            expected = 1. / expected;
         EXPECT_DOUBLE_EQ(volume.getVal(), expected);
      }
   }
}

TEST(RooBinWidthFunction, Caches)
{
   int widthCalls = 0;
   RooRealVar x{"x", "x", 0., 6.};
   x.setBinning(CountingBinning{widthCalls});
   RooDataHist data{"data", "data", x};
   RooHistFunc hist{"hist", "hist", x, data};
   CountingFunction volume{"volume", "volume", hist, false};
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
}

TEST(RooBinWidthFunction, PlottingHints)
{
   RooRealVar x{"x", "x", 0., 6.};
   RooRealVar unrelated{"unrelated", "unrelated", 0., 6.};
   x.setBinning(RooBinning{3, edges});
   RooDataHist data{"data", "data", x};
   RooHistFunc hist{"hist", "hist", x, data};
   RooBinWidthFunction volume{"volume", "volume", hist, false};
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
   RooDataHist data{"data", "data", x};
   RooHistFunc hist{"hist", "hist", x, data};
   RooBinWidthFunction volume{"volume", "volume", hist, true};
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

TEST(RooBinWidthFunction, WorkspaceRoundTrip)
{
   RooRealVar x{"x", "x", 0., 6.};
   RooRealVar y{"y", "y", 0., 10.};
   x.setBinning(RooBinning{3, edges});
   y.setBins(2);
   RooDataHist data{"data", "data", {x, y}};
   RooHistFunc hist{"hist", "hist", {x, y}, data};
   RooBinWidthFunction volume{"volume", "volume", hist, false};
   RooBinWidthFunction inverse{"inverse", "inverse", hist, true};
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

// Generate the historical input with class version 1. Save the following
// recipe as makeRooBinWidthFunction_v1.cxx and run it with a ROOT build
// containing that version.
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
}
