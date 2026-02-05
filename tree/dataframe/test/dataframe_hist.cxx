#include <gtest/gtest.h>

#include <ROOT/TestSupport.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RBinWithError.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RHistEngine.hxx>
#include <ROOT/RRegularAxis.hxx>
#include <ROOT/RVariableBinAxis.hxx>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using ROOT::RDataFrame;
using ROOT::Experimental::RBinWithError;
using ROOT::Experimental::RHist;
using ROOT::Experimental::RHistEngine;
using ROOT::Experimental::RRegularAxis;
using ROOT::Experimental::RVariableBinAxis;
using ROOT::RDF::RunGraphs;

// Fixture for all tests in this file to optionally run with multi-threading.
class RDFHist : public ::testing::TestWithParam<bool> {
   ROOT::TestSupport::CheckDiagsRAII fDiag;

public:
   RDFHist()
   {
      fDiag.optionalDiag(kWarning, "", "Filling RHist is experimental", /*matchFullMessage=*/false);
      if (GetParam())
         ROOT::EnableImplicitMT(4);
   }
   ~RDFHist() override
   {
      if (GetParam())
         ROOT::DisableImplicitMT();
   }
};

TEST_P(RDFHist, Regular)
{
   RDataFrame df(10);
   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Hist</*BinContentType=*/double, double>({axis}, {"x"});
   EXPECT_EQ(hist->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist->GetBinContent(index), 1.0);
   }
}

TEST_P(RDFHist, RegularJit)
{
   RDataFrame df(10);
   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = df.Define("x", "rdfentry_ + 5.5").Hist({axis}, {"x"});
   EXPECT_EQ(hist->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist->GetBinContent(index), 1.0);
   }
}

TEST_P(RDFHist, MultiDim)
{
   RDataFrame df(10);
   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto hist = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("y", [](ULong64_t e) { return 2 * e + 0.5; }, {"rdfentry_"})
                  .Hist</*BinContentType=*/double, double, double>({regularAxis, variableBinAxis}, {"x", "y"});
   EXPECT_EQ(hist->GetNEntries(), 10);
   for (auto x : regularAxis.GetNormalRange()) {
      for (auto y : variableBinAxis.GetNormalRange()) {
         if (2 * x.GetIndex() == y.GetIndex()) {
            EXPECT_EQ(hist->GetBinContent(x, y), 1.0);
         } else {
            EXPECT_EQ(hist->GetBinContent(x, y), 0.0);
         }
      }
   }
}

TEST_P(RDFHist, MultiDimJit)
{
   RDataFrame df(10);
   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto hist = df.Define("x", "rdfentry_ + 5.5")
                  .Define("y", "2 * rdfentry_ + 0.5")
                  .Hist({regularAxis, variableBinAxis}, {"x", "y"});
   EXPECT_EQ(hist->GetNEntries(), 10);
   for (auto x : regularAxis.GetNormalRange()) {
      for (auto y : variableBinAxis.GetNormalRange()) {
         if (2 * x.GetIndex() == y.GetIndex()) {
            EXPECT_EQ(hist->GetBinContent(x, y), 1.0);
         } else {
            EXPECT_EQ(hist->GetBinContent(x, y), 0.0);
         }
      }
   }
}

TEST_P(RDFHist, Ptr)
{
   RDataFrame df(10);
   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   auto resPtr = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"}).Hist<double>(hist, {"x"});
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   EXPECT_EQ(hist->GetNEntries(), 10);
}

TEST_P(RDFHist, PtrJit)
{
   RDataFrame df(10);
   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   auto resPtr = df.Define("x", "rdfentry_ + 5.5").Hist(hist, {"x"});
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   EXPECT_EQ(hist->GetNEntries(), 10);
}

TEST_P(RDFHist, PtrRunGraphs)
{
   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));

   RDataFrame df1(10);
   auto resPtr1 = df1.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"}).Hist<double>(hist, {"x"});

   RDataFrame df2(7);
   auto resPtr2 = df2.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"}).Hist<double>(hist, {"x"});

   RunGraphs({resPtr1, resPtr2});
   EXPECT_EQ(hist->GetNEntries(), 17);
}

TEST_P(RDFHist, Engine)
{
   RDataFrame df(10);
   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = std::make_shared<RHistEngine<double>>(axis);
   auto resPtr = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"}).Hist<double>(hist, {"x"});
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist->GetBinContent(index), 1.0);
   }
}

TEST_P(RDFHist, EngineJit)
{
   RDataFrame df(10);
   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = std::make_shared<RHistEngine<double>>(axis);
   auto resPtr = df.Define("x", "rdfentry_ + 5.5").Hist(hist, {"x"});
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist->GetBinContent(index), 1.0);
   }
}

TEST_P(RDFHist, EngineMultiDim)
{
   RDataFrame df(10);
   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto hist = std::make_shared<RHistEngine<double>>(regularAxis, variableBinAxis);
   auto resPtr = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                    .Define("y", [](ULong64_t e) { return 2 * e + 0.5; }, {"rdfentry_"})
                    .Hist<double, double>(hist, {"x", "y"});
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   for (auto x : regularAxis.GetNormalRange()) {
      for (auto y : variableBinAxis.GetNormalRange()) {
         if (2 * x.GetIndex() == y.GetIndex()) {
            EXPECT_EQ(hist->GetBinContent(x, y), 1.0);
         } else {
            EXPECT_EQ(hist->GetBinContent(x, y), 0.0);
         }
      }
   }
}

TEST_P(RDFHist, EngineMultiDimJit)
{
   RDataFrame df(10);
   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto hist = std::make_shared<RHistEngine<double>>(regularAxis, variableBinAxis);
   auto resPtr = df.Define("x", "rdfentry_ + 5.5").Define("y", "2 * rdfentry_ + 0.5").Hist(hist, {"x", "y"});
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   for (auto x : regularAxis.GetNormalRange()) {
      for (auto y : variableBinAxis.GetNormalRange()) {
         if (2 * x.GetIndex() == y.GetIndex()) {
            EXPECT_EQ(hist->GetBinContent(x, y), 1.0);
         } else {
            EXPECT_EQ(hist->GetBinContent(x, y), 0.0);
         }
      }
   }
}

TEST_P(RDFHist, InvalidNumberOfArguments)
{
   RDataFrame df(10);
   const RRegularAxis axis(10, {5.0, 15.0});
   auto dfX = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"});
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfX.Hist</*BinContentType=*/double, double, double>({axis}, {"x", "x"});
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &e) {
      // expected
   }

   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfX.Hist<double, double>(hist, {"x", "x"});
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &e) {
      // expected
   }

   auto engine = std::make_shared<RHistEngine<double>>(10, std::make_pair(5.0, 15.0));
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfX.Hist<double, double>(engine, {"x", "x"});
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &e) {
      // expected
   }
}

TEST_P(RDFHist, InvalidNumberOfArgumentsJit)
{
   RDataFrame df(10);
   const RRegularAxis axis(10, {5.0, 15.0});
   auto dfX = df.Define("x", "rdfentry_ + 5.5");
   EXPECT_THROW(dfX.Hist({axis}, {"x", "x"}), std::invalid_argument);

   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   EXPECT_THROW(dfX.Hist(hist, {"x", "x"}), std::invalid_argument);

   auto engine = std::make_shared<RHistEngine<double>>(10, std::make_pair(5.0, 15.0));
   EXPECT_THROW(dfX.Hist(engine, {"x", "x"}), std::invalid_argument);
}

TEST_P(RDFHist, Weight)
{
   RDataFrame df(10);
   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"})
                  .Hist</*BinContentType=*/RBinWithError, double, double>({axis}, {"x"}, "w");
   EXPECT_EQ(hist->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      auto &bin = hist->GetBinContent(index);
      double weight = 0.1 + index.GetIndex() * 0.03;
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
   }
}

TEST_P(RDFHist, WeightJit)
{
   RDataFrame df(10);
   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = df.Define("x", "rdfentry_ + 5.5").Define("w", "0.1 + rdfentry_ * 0.03").Hist({axis}, {"x"}, "w");
   EXPECT_EQ(hist->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      auto &bin = hist->GetBinContent(index);
      double weight = 0.1 + index.GetIndex() * 0.03;
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
   }
}

TEST_P(RDFHist, PtrWeight)
{
   RDataFrame df(10);
   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   auto resPtr = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                    .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"})
                    .Hist<double, double>(hist, {"x"}, "w");
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   EXPECT_EQ(hist->GetNEntries(), 10);
}

TEST_P(RDFHist, PtrWeightJit)
{
   RDataFrame df(10);
   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   auto resPtr = df.Define("x", "rdfentry_ + 5.5").Define("w", "0.1 + rdfentry_ * 0.03").Hist(hist, {"x"}, "w");
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   EXPECT_EQ(hist->GetNEntries(), 10);
}

TEST_P(RDFHist, EngineWeight)
{
   RDataFrame df(10);
   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = std::make_shared<RHistEngine<RBinWithError>>(10, std::make_pair(5.0, 15.0));
   auto resPtr = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                    .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"})
                    .Hist<double, double>(hist, {"x"}, "w");
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   for (auto index : axis.GetNormalRange()) {
      auto &bin = hist->GetBinContent(index);
      double weight = 0.1 + index.GetIndex() * 0.03;
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
   }
}

TEST_P(RDFHist, EngineWeightJit)
{
   RDataFrame df(10);
   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = std::make_shared<RHistEngine<RBinWithError>>(10, std::make_pair(5.0, 15.0));
   auto resPtr = df.Define("x", "rdfentry_ + 5.5").Define("w", "0.1 + rdfentry_ * 0.03").Hist(hist, {"x"}, "w");
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   for (auto index : axis.GetNormalRange()) {
      auto &bin = hist->GetBinContent(index);
      double weight = 0.1 + index.GetIndex() * 0.03;
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
   }
}

TEST_P(RDFHist, WeightInvalidNumberOfArguments)
{
   RDataFrame df(10);
   const RRegularAxis axis(10, {5.0, 15.0});
   auto dfXW = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"});
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfXW.Hist</*BinContentType=*/double, double, double, double>({axis}, {"x", "x"}, "w");
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &e) {
      // expected
   }

   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfXW.Hist<double, double, double>(hist, {"x", "x"}, "w");
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &e) {
      // expected
   }

   auto engine = std::make_shared<RHistEngine<double>>(10, std::make_pair(5.0, 15.0));
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfXW.Hist<double, double, double>(engine, {"x", "x"}, "w");
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &e) {
      // expected
   }
}

TEST_P(RDFHist, WeightInvalidNumberOfArgumentsJit)
{
   RDataFrame df(10);
   const RRegularAxis axis(10, {5.0, 15.0});
   auto dfXW = df.Define("x", "rdfentry_ + 5.5").Define("w", "0.1 + rdfentry_ * 0.03");
   EXPECT_THROW(dfXW.Hist({axis}, {"x", "x"}, "w"), std::invalid_argument);

   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   EXPECT_THROW(dfXW.Hist(hist, {"x", "x"}, "w"), std::invalid_argument);

   auto engine = std::make_shared<RHistEngine<double>>(10, std::make_pair(5.0, 15.0));
   EXPECT_THROW(dfXW.Hist(engine, {"x", "x"}, "w"), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(Seq, RDFHist, ::testing::Values(false));

#ifdef R__USE_IMT
INSTANTIATE_TEST_SUITE_P(MT, RDFHist, ::testing::Values(true));
#endif
