#include <gtest/gtest.h>

#include <ROOT/TestSupport.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RBinWithError.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RHistEngine.hxx>
#include <ROOT/RRegularAxis.hxx>
#include <ROOT/RVariableBinAxis.hxx>
#include <ROOT/RVec.hxx>

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
   auto dfX = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"});

   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = dfX.Hist</*BinContentType=*/double, double>({axis}, {"x"});
   EXPECT_EQ(hist->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist->GetBinContent(index), 1.0);
   }

   // The one-dimensional specialization returns the same type.
   hist = dfX.Hist</*BinContentType=*/double, double>(10, {5.0, 15.0}, "x");
   EXPECT_EQ(hist->GetNEntries(), 10);
}

TEST_P(RDFHist, RegularJit)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", "rdfentry_ + 5.5");

   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = dfX.Hist({axis}, {"x"});
   EXPECT_EQ(hist->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist->GetBinContent(index), 1.0);
   }

   // The user can template explicitly only the bin content type.
   hist = dfX.Hist</*BinContentType=*/double>({axis}, {"x"});
   EXPECT_EQ(hist->GetNEntries(), 10);

   // The one-dimensional specialization returns the same type.
   hist = dfX.Hist(10, {5.0, 15.0}, "x");
   EXPECT_EQ(hist->GetNEntries(), 10);
}

TEST_P(RDFHist, MultiDim)
{
   RDataFrame df(10);
   auto dfXY = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("y", [](ULong64_t e) { return 2 * e + 0.5; }, {"rdfentry_"});

   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto hist = dfXY.Hist</*BinContentType=*/double, double, double>({regularAxis, variableBinAxis}, {"x", "y"});
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
   auto dfXY = df.Define("x", "rdfentry_ + 5.5").Define("y", "2 * rdfentry_ + 0.5");

   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto hist = dfXY.Hist({regularAxis, variableBinAxis}, {"x", "y"});
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
   auto dfX = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"});

   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   auto resPtr = dfX.Hist<double>(hist, {"x"});
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   EXPECT_EQ(hist->GetNEntries(), 10);
}

TEST_P(RDFHist, PtrJit)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", "rdfentry_ + 5.5");

   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   auto resPtr = dfX.Hist(hist, {"x"});
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
   auto dfX = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"});

   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = std::make_shared<RHistEngine<double>>(axis);
   auto resPtr = dfX.Hist<double>(hist, {"x"});
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist->GetBinContent(index), 1.0);
   }
}

TEST_P(RDFHist, EngineJit)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", "rdfentry_ + 5.5");

   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = std::make_shared<RHistEngine<double>>(axis);
   auto resPtr = dfX.Hist(hist, {"x"});
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist->GetBinContent(index), 1.0);
   }
}

TEST_P(RDFHist, EngineMultiDim)
{
   RDataFrame df(10);
   auto dfXY = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("y", [](ULong64_t e) { return 2 * e + 0.5; }, {"rdfentry_"});

   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto hist = std::make_shared<RHistEngine<double>>(regularAxis, variableBinAxis);
   auto resPtr = dfXY.Hist<double, double>(hist, {"x", "y"});
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
   auto dfXY = df.Define("x", "rdfentry_ + 5.5").Define("y", "2 * rdfentry_ + 0.5");

   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto hist = std::make_shared<RHistEngine<double>>(regularAxis, variableBinAxis);
   auto resPtr = dfXY.Hist(hist, {"x", "y"});
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
   auto dfX = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"});

   const RRegularAxis axis(10, {5.0, 15.0});
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfX.Hist</*BinContentType=*/double, double, double>({axis}, {"x", "x"});
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &) {
      // expected
   }

   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfX.Hist<double, double>(hist, {"x", "x"});
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &) {
      // expected
   }

   auto engine = std::make_shared<RHistEngine<double>>(10, std::make_pair(5.0, 15.0));
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfX.Hist<double, double>(engine, {"x", "x"});
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &) {
      // expected
   }
}

TEST_P(RDFHist, InvalidNumberOfArgumentsJit)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", "rdfentry_ + 5.5");

   const RRegularAxis axis(10, {5.0, 15.0});
   EXPECT_THROW(dfX.Hist({axis}, {"x", "x"}), std::invalid_argument);

   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   EXPECT_THROW(dfX.Hist(hist, {"x", "x"}), std::invalid_argument);

   auto engine = std::make_shared<RHistEngine<double>>(10, std::make_pair(5.0, 15.0));
   EXPECT_THROW(dfX.Hist(engine, {"x", "x"}), std::invalid_argument);
}

TEST_P(RDFHist, Weight)
{
   RDataFrame df(10);
   auto dfXW = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"});

   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = dfXW.Hist</*BinContentType=*/RBinWithError, double, double>({axis}, {"x"}, "w");
   EXPECT_EQ(hist->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      auto &bin = hist->GetBinContent(index);
      double weight = 0.1 + index.GetIndex() * 0.03;
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
   }

   // The one-dimensional specialization returns the same type.
   hist = dfXW.Hist</*BinContentType=*/RBinWithError, double, double>(10, {5.0, 15.0}, "x", "w");
   EXPECT_EQ(hist->GetNEntries(), 10);
}

TEST_P(RDFHist, WeightJit)
{
   RDataFrame df(10);
   auto dfXW = df.Define("x", "rdfentry_ + 5.5").Define("w", "0.1 + rdfentry_ * 0.03");

   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = dfXW.Hist({axis}, {"x"}, "w");
   EXPECT_EQ(hist->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      auto &bin = hist->GetBinContent(index);
      double weight = 0.1 + index.GetIndex() * 0.03;
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
   }

   // The user can template explicitly only the bin content type.
   hist = dfXW.Hist</*BinContentType=*/RBinWithError>({axis}, {"x"}, "w");
   EXPECT_EQ(hist->GetNEntries(), 10);

   // The one-dimensional specialization returns the same type.
   hist = dfXW.Hist(10, {5.0, 15.0}, "x", "w");
   EXPECT_EQ(hist->GetNEntries(), 10);
}

TEST_P(RDFHist, PtrWeight)
{
   RDataFrame df(10);
   auto dfXW = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"});

   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   auto resPtr = dfXW.Hist<double, double>(hist, {"x"}, "w");
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   EXPECT_EQ(hist->GetNEntries(), 10);
}

TEST_P(RDFHist, PtrWeightJit)
{
   RDataFrame df(10);
   auto dfXW = df.Define("x", "rdfentry_ + 5.5").Define("w", "0.1 + rdfentry_ * 0.03");

   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   auto resPtr = dfXW.Hist(hist, {"x"}, "w");
   EXPECT_EQ(hist, resPtr.GetSharedPtr());
   EXPECT_EQ(hist->GetNEntries(), 10);
}

TEST_P(RDFHist, EngineWeight)
{
   RDataFrame df(10);
   auto dfXW = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"});

   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = std::make_shared<RHistEngine<RBinWithError>>(10, std::make_pair(5.0, 15.0));
   auto resPtr = dfXW.Hist<double, double>(hist, {"x"}, "w");
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
   auto dfXW = df.Define("x", "rdfentry_ + 5.5").Define("w", "0.1 + rdfentry_ * 0.03");

   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = std::make_shared<RHistEngine<RBinWithError>>(10, std::make_pair(5.0, 15.0));
   auto resPtr = dfXW.Hist(hist, {"x"}, "w");
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
   auto dfXW = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"});

   const RRegularAxis axis(10, {5.0, 15.0});
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfXW.Hist</*BinContentType=*/double, double, double, double>({axis}, {"x", "x"}, "w");
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &) {
      // expected
   }

   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfXW.Hist<double, double, double>(hist, {"x", "x"}, "w");
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &) {
      // expected
   }

   auto engine = std::make_shared<RHistEngine<double>>(10, std::make_pair(5.0, 15.0));
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfXW.Hist<double, double, double>(engine, {"x", "x"}, "w");
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &) {
      // expected
   }
}

TEST_P(RDFHist, WeightInvalidNumberOfArgumentsJit)
{
   RDataFrame df(10);
   auto dfXW = df.Define("x", "rdfentry_ + 5.5").Define("w", "0.1 + rdfentry_ * 0.03");

   const RRegularAxis axis(10, {5.0, 15.0});
   EXPECT_THROW(dfXW.Hist({axis}, {"x", "x"}, "w"), std::invalid_argument);

   auto hist = std::make_shared<RHist<double>>(10, std::make_pair(5.0, 15.0));
   EXPECT_THROW(dfXW.Hist(hist, {"x", "x"}, "w"), std::invalid_argument);

   auto engine = std::make_shared<RHistEngine<double>>(10, std::make_pair(5.0, 15.0));
   EXPECT_THROW(dfXW.Hist(engine, {"x", "x"}, "w"), std::invalid_argument);
}

TEST_P(RDFHist, Container)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", [](ULong64_t e) { return ROOT::RVecD{e + 5.5}; }, {"rdfentry_"});

   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = dfX.Hist</*BinContentType=*/double, ROOT::RVecD>({axis}, {"x"});
   EXPECT_EQ(hist->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist->GetBinContent(index), 1.0);
   }
}

TEST_P(RDFHist, ContainerEmpty)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", [] { return ROOT::RVecD{}; });

   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = dfX.Hist</*BinContentType=*/double, ROOT::RVecD>({axis}, {"x"});
   EXPECT_EQ(hist->GetNEntries(), 0);
}

TEST_P(RDFHist, ContainerElements)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", [](ULong64_t e) { return ROOT::RVecD{e + 5.5, e + 15.5}; }, {"rdfentry_"});

   const RRegularAxis axis(20, {5.0, 25.0});
   auto hist = dfX.Hist</*BinContentType=*/double, ROOT::RVecD>({axis}, {"x"});
   EXPECT_EQ(hist->GetNEntries(), 20);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist->GetBinContent(index), 1.0);
   }
}

TEST_P(RDFHist, ContainerNested)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", [](ULong64_t e) { return ROOT::RVec<ROOT::RVecD>{{e + 5.5}, {e + 15.5}}; }, {"rdfentry_"});

   const RRegularAxis axis(20, {5.0, 25.0});
   auto hist = dfX.Hist</*BinContentType=*/double, ROOT::RVec<ROOT::RVecD>>({axis}, {"x"});
   EXPECT_EQ(hist->GetNEntries(), 20);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist->GetBinContent(index), 1.0);
   }
}

TEST_P(RDFHist, ContainerJit)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", "ROOT::RVecD{rdfentry_ + 5.5}");

   const RRegularAxis axis(10, {5.0, 15.0});
   auto hist = dfX.Hist({axis}, {"x"});
   EXPECT_EQ(hist->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist->GetBinContent(index), 1.0);
   }
}

TEST_P(RDFHist, ContainerMultiDim)
{
   RDataFrame df(10);
   auto dfXY = df.Define("x", [](ULong64_t e) { return ROOT::RVecD{e + 5.5}; }, {"rdfentry_"})
                  .Define("y", [](ULong64_t e) { return ROOT::RVecD{2 * e + 0.5}; }, {"rdfentry_"});

   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto hist =
      dfXY.Hist</*BinContentType=*/double, ROOT::RVecD, ROOT::RVecD>({regularAxis, variableBinAxis}, {"x", "y"});
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

TEST_P(RDFHist, ContainerMultiDimBroadcast)
{
   RDataFrame df(10);
   auto dfXY = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("y", [](ULong64_t e) { return ROOT::RVecD{2 * e + 0.5}; }, {"rdfentry_"});

   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto hist = dfXY.Hist</*BinContentType=*/double, double, ROOT::RVecD>({regularAxis, variableBinAxis}, {"x", "y"});
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

TEST_P(RDFHist, ContainerMultiDimNestedBroadcast)
{
   RDataFrame df(10);
   auto dfXY = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("xV", [](ULong64_t e) { return ROOT::RVecD{e + 5.5}; }, {"rdfentry_"})
                  .Define("y", [](ULong64_t e) { return ROOT::RVec<ROOT::RVecD>{{2 * e + 0.5}}; }, {"rdfentry_"});

   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto hist =
      dfXY.Hist</*BinContentType=*/double, double, ROOT::RVec<ROOT::RVecD>>({regularAxis, variableBinAxis}, {"x", "y"});
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

   hist = dfXY.Hist</*BinContentType=*/double, ROOT::RVecD, ROOT::RVec<ROOT::RVecD>>({regularAxis, variableBinAxis},
                                                                                     {"xV", "y"});
   EXPECT_EQ(hist->GetNEntries(), 10);
}

TEST_P(RDFHist, ContainerInvalidElements)
{
   RDataFrame df(10);
   auto dfXY = df.Define("x0", [] { return ROOT::RVecD{}; })
                  .Define("x2", [](ULong64_t e) { return ROOT::RVecD{e + 5.5, e + 15.5}; }, {"rdfentry_"})
                  .Define("y1", [](ULong64_t e) { return ROOT::RVecD{2 * e + 0.5}; }, {"rdfentry_"});

   const RRegularAxis regularAxis(20, {5.0, 25.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   try {
      // Cannot use EXPECT_THROW because of template arguments...
      *dfXY.Hist</*BinContentType=*/double, ROOT::RVecD, ROOT::RVecD>({regularAxis, variableBinAxis}, {"x0", "y1"});
      FAIL() << "expected std::runtime_error";
   } catch (const std::runtime_error &) {
      // expected
   }

   try {
      // Cannot use EXPECT_THROW because of template arguments...
      *dfXY.Hist</*BinContentType=*/double, ROOT::RVecD, ROOT::RVecD>({regularAxis, variableBinAxis}, {"x2", "y1"});
      FAIL() << "expected std::runtime_error";
   } catch (const std::runtime_error &e) {
      // expected
   }
}

TEST_P(RDFHist, ContainerNestedInvalidElements)
{
   RDataFrame df(10);
   auto dfXY =
      df.Define("x11", [](ULong64_t e) { return ROOT::RVec<ROOT::RVecD>{{e + 5.5}}; }, {"rdfentry_"})
         .Define("y12", [](ULong64_t e) { return ROOT::RVec<ROOT::RVecD>{{2 * e + 0.5, 2 * e + 0.5}}; }, {"rdfentry_"});

   const RRegularAxis regularAxis(20, {5.0, 25.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   try {
      // Cannot use EXPECT_THROW because of template arguments...
      *dfXY.Hist</*BinContentType=*/double, ROOT::RVec<ROOT::RVecD>, ROOT::RVec<ROOT::RVecD>>(
         {regularAxis, variableBinAxis}, {"x11", "y12"});
      FAIL() << "expected std::runtime_error";
   } catch (const std::runtime_error &e) {
      // expected
   }
}

TEST_P(RDFHist, WeightContainer)
{
   RDataFrame df(10);
   auto dfXW =
      df.Define("xV", [](ULong64_t e) { return ROOT::RVecD{e + 5.5, e + 15.5}; }, {"rdfentry_"})
         .Define("wV", [](ULong64_t e) { return ROOT::RVecD{0.1 + e * 0.03, 0.2 + e * e * 0.06}; }, {"rdfentry_"});

   const RRegularAxis axis(20, {5.0, 25.0});
   auto hist = dfXW.Hist</*BinContentType=*/RBinWithError, ROOT::RVecD, ROOT::RVecD>({axis}, {"xV"}, "wV");
   EXPECT_EQ(hist->GetNEntries(), 20);
   for (auto index : axis.GetNormalRange()) {
      auto &bin = hist->GetBinContent(index);
      auto i = index.GetIndex();
      double weight = i >= 10 ? (0.2 + (i - 10) * (i - 10) * 0.06) : (0.1 + i * 0.03);
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
   }
}

TEST_P(RDFHist, WeightContainerJit)
{
   RDataFrame df(10);
   auto dfXW = df.Define("xV", "ROOT::RVecD{rdfentry_ + 5.5, rdfentry_ + 15.5}")
                  .Define("wV", "ROOT::RVecD{0.1 + rdfentry_ * 0.03, 0.2 + rdfentry_ * rdfentry_ * 0.06}");

   const RRegularAxis axis(20, {5.0, 25.0});
   auto hist = dfXW.Hist({axis}, {"xV"}, "wV");
   EXPECT_EQ(hist->GetNEntries(), 20);
   for (auto index : axis.GetNormalRange()) {
      auto &bin = hist->GetBinContent(index);
      auto i = index.GetIndex();
      double weight = i >= 10 ? (0.2 + (i - 10) * (i - 10) * 0.06) : (0.1 + i * 0.03);
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
   }
}

TEST_P(RDFHist, WeightContainerBroadcast)
{
   RDataFrame df(10);
   auto dfXW =
      df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
         .Define("xV", [](ULong64_t e) { return ROOT::RVecD{e + 5.5, e + 15.5}; }, {"rdfentry_"})
         .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"})
         .Define("wV", [](ULong64_t e) { return ROOT::RVecD{0.1 + e * 0.03, 0.2 + e * e * 0.06}; }, {"rdfentry_"});

   const RRegularAxis axis(20, {5.0, 25.0});
   auto hist = dfXW.Hist</*BinContentType=*/RBinWithError, ROOT::RVecD, double>({axis}, {"xV"}, "w");
   EXPECT_EQ(hist->GetNEntries(), 20);
   for (auto index : axis.GetNormalRange()) {
      auto &bin = hist->GetBinContent(index);
      auto i = index.GetIndex();
      double weight = 0.1 + (i >= 10 ? i - 10 : i) * 0.03;
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
   }

   hist = dfXW.Hist</*BinContentType=*/RBinWithError, double, ROOT::RVecD>({axis}, {"x"}, "wV");
   EXPECT_EQ(hist->GetNEntries(), 20);
   EXPECT_EQ(hist->GetBinContent(2).fSum, 0.1 + 2 * 0.03 + 0.2 + 2 * 2 * 0.06);
}

INSTANTIATE_TEST_SUITE_P(Seq, RDFHist, ::testing::Values(false));

#ifdef R__USE_IMT
INSTANTIATE_TEST_SUITE_P(MT, RDFHist, ::testing::Values(true));
#endif
