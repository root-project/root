#include <gtest/gtest.h>

#include <ROOT/TestSupport.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RProfile.hxx>
#include <ROOT/RRegularAxis.hxx>
#include <ROOT/RVariableBinAxis.hxx>
#include <ROOT/RVec.hxx>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using ROOT::RDataFrame;
using ROOT::Experimental::RProfile;
using ROOT::Experimental::RRegularAxis;
using ROOT::Experimental::RVariableBinAxis;
using ROOT::RDF::RunGraphs;

// Fixture for all tests in this file to optionally run with multi-threading.
class RDFProfile : public ::testing::TestWithParam<bool> {
   ROOT::TestSupport::CheckDiagsRAII fDiag;

public:
   RDFProfile()
   {
      fDiag.optionalDiag(kWarning, "", "Filling RProfile is experimental", /*matchFullMessage=*/false);
      fDiag.optionalDiag(kWarning, "cling",
                         "expected alignment (8 bytes) exceeds the actual alignment (4 bytes) [-Watomic-alignment]",
                         false);
      if (GetParam())
         ROOT::EnableImplicitMT(4);
   }
   ~RDFProfile() override
   {
      if (GetParam())
         ROOT::DisableImplicitMT();
   }
};

TEST_P(RDFProfile, Regular)
{
   RDataFrame df(10);
   auto dfXY = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("y", [](ULong64_t e) { return 2 * e + 0.5; }, {"rdfentry_"});

   const RRegularAxis axis(10, {5.0, 15.0});
   auto profile = dfXY.Profile<double, double>({axis}, {"x"}, "y");
   EXPECT_EQ(profile->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(profile->GetBinContent(index).fSum, 1.0);
      EXPECT_EQ(profile->GetBinContent(index).fSumValues, 2 * index.GetIndex() + 0.5);
   }

   // The one-dimensional specialization returns the same type.
   profile = dfXY.Profile<double, double>(10, {5.0, 15.0}, "x", "y");
   EXPECT_EQ(profile->GetNEntries(), 10);
}

TEST_P(RDFProfile, RegularJit)
{
   RDataFrame df(10);
   auto dfXY = df.Define("x", "rdfentry_ + 5.5").Define("y", "2 * rdfentry_ + 0.5");

   const RRegularAxis axis(10, {5.0, 15.0});
   auto profile = dfXY.Profile({axis}, {"x"}, "y");
   EXPECT_EQ(profile->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(profile->GetBinContent(index).fSum, 1.0);
      EXPECT_EQ(profile->GetBinContent(index).fSumValues, 2 * index.GetIndex() + 0.5);
   }

   // The one-dimensional specialization returns the same type.
   profile = dfXY.Profile(10, {5.0, 15.0}, "x", "y");
   EXPECT_EQ(profile->GetNEntries(), 10);
}

TEST_P(RDFProfile, MultiDim)
{
   RDataFrame df(10);
   auto dfXYZ = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                   .Define("y", [](ULong64_t e) { return 2 * e + 0.5; }, {"rdfentry_"})
                   .Define("z", [](ULong64_t e) { return 3 * e + 10.5; }, {"rdfentry_"});

   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto profile = dfXYZ.Profile<double, double, double>({regularAxis, variableBinAxis}, {"x", "y"}, "z");
   EXPECT_EQ(profile->GetNEntries(), 10);
   for (auto x : regularAxis.GetNormalRange()) {
      for (auto y : variableBinAxis.GetNormalRange()) {
         if (2 * x.GetIndex() == y.GetIndex()) {
            EXPECT_EQ(profile->GetBinContent(x, y).fSum, 1.0);
            EXPECT_EQ(profile->GetBinContent(x, y).fSumValues, 3 * x.GetIndex() + 10.5);
         } else {
            EXPECT_EQ(profile->GetBinContent(x, y).fSum, 0.0);
         }
      }
   }
}

TEST_P(RDFProfile, MultiDimJit)
{
   RDataFrame df(10);
   auto dfXYZ =
      df.Define("x", "rdfentry_ + 5.5").Define("y", "2 * rdfentry_ + 0.5").Define("z", "3 * rdfentry_ + 10.5");

   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto profile = dfXYZ.Profile({regularAxis, variableBinAxis}, {"x", "y"}, "z");
   EXPECT_EQ(profile->GetNEntries(), 10);
   for (auto x : regularAxis.GetNormalRange()) {
      for (auto y : variableBinAxis.GetNormalRange()) {
         if (2 * x.GetIndex() == y.GetIndex()) {
            EXPECT_EQ(profile->GetBinContent(x, y).fSum, 1.0);
            EXPECT_EQ(profile->GetBinContent(x, y).fSumValues, 3 * x.GetIndex() + 10.5);
         } else {
            EXPECT_EQ(profile->GetBinContent(x, y).fSum, 0.0);
         }
      }
   }
}

TEST_P(RDFProfile, Ptr)
{
   RDataFrame df(10);
   auto dfXY = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("y", [](ULong64_t e) { return 2 * e + 0.5; }, {"rdfentry_"});

   auto profile = std::make_shared<RProfile>(10, std::make_pair(5.0, 15.0));
   auto resPtr = dfXY.Profile<double, double>(profile, {"x"}, "y");
   EXPECT_EQ(profile, resPtr.GetSharedPtr());
   EXPECT_EQ(profile->GetNEntries(), 10);
}

TEST_P(RDFProfile, PtrJit)
{
   RDataFrame df(10);
   auto dfXY = df.Define("x", "rdfentry_ + 5.5").Define("y", "2 * rdfentry_ + 0.5");

   auto profile = std::make_shared<RProfile>(10, std::make_pair(5.0, 15.0));
   auto resPtr = dfXY.Profile(profile, {"x"}, "y");
   EXPECT_EQ(profile, resPtr.GetSharedPtr());
   EXPECT_EQ(profile->GetNEntries(), 10);
}

TEST_P(RDFProfile, InvalidNumberOfArguments)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"});

   const RRegularAxis axis(10, {5.0, 15.0});
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfX.Profile<double, double, double>({axis}, {"x", "x"}, "x");
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &) {
      // expected
   }

   auto profile = std::make_shared<RProfile>(axis);
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfX.Profile<double, double, double>(profile, {"x", "x"}, "x");
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &) {
      // expected
   }
}

TEST_P(RDFProfile, InvalidNumberOfArgumentsJit)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", "rdfentry_ + 5.5");

   const RRegularAxis axis(10, {5.0, 15.0});
   EXPECT_THROW(dfX.Profile({axis}, {"x", "x"}, "x"), std::invalid_argument);

   auto profile = std::make_shared<RProfile>(axis);
   EXPECT_THROW(dfX.Profile(profile, {"x", "x"}, "x"), std::invalid_argument);
}

TEST_P(RDFProfile, Weight)
{
   RDataFrame df(10);
   auto dfXYW = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                   .Define("y", [](ULong64_t e) { return 2 * e + 0.5; }, {"rdfentry_"})
                   .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"});

   const RRegularAxis axis(10, {5.0, 15.0});
   auto profile = dfXYW.Profile<double, double, double>({axis}, {"x"}, "y", "w");
   EXPECT_EQ(profile->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      auto &bin = profile->GetBinContent(index);
      double value = 2 * index.GetIndex() + 0.5;
      double weight = 0.1 + index.GetIndex() * 0.03;
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
      EXPECT_FLOAT_EQ(bin.fSumValues, weight * value);
      EXPECT_FLOAT_EQ(bin.fSumValues2, weight * value * value);
   }

   // The one-dimensional specialization returns the same type.
   profile = dfXYW.Profile<double, double, double>(10, {5.0, 15.0}, "x", "y", "w");
   EXPECT_EQ(profile->GetNEntries(), 10);
}

TEST_P(RDFProfile, WeightJit)
{
   RDataFrame df(10);
   auto dfXYW =
      df.Define("x", "rdfentry_ + 5.5").Define("y", "2 * rdfentry_ + 0.5").Define("w", "0.1 + rdfentry_ * 0.03");

   const RRegularAxis axis(10, {5.0, 15.0});
   auto profile = dfXYW.Profile({axis}, {"x"}, "y", "w");
   EXPECT_EQ(profile->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      auto &bin = profile->GetBinContent(index);
      double value = 2 * index.GetIndex() + 0.5;
      double weight = 0.1 + index.GetIndex() * 0.03;
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
      EXPECT_FLOAT_EQ(bin.fSumValues, weight * value);
      EXPECT_FLOAT_EQ(bin.fSumValues2, weight * value * value);
   }

   // The one-dimensional specialization returns the same type.
   profile = dfXYW.Profile(10, {5.0, 15.0}, "x", "y", "w");
   EXPECT_EQ(profile->GetNEntries(), 10);
}

TEST_P(RDFProfile, PtrWeight)
{
   RDataFrame df(10);
   auto dfXYW = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                   .Define("y", [](ULong64_t e) { return 2 * e + 0.5; }, {"rdfentry_"})
                   .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"});

   auto profile = std::make_shared<RProfile>(10, std::make_pair(5.0, 15.0));
   auto resPtr = dfXYW.Profile<double, double, double>(profile, {"x"}, "y", "w");
   EXPECT_EQ(profile, resPtr.GetSharedPtr());
   EXPECT_EQ(profile->GetNEntries(), 10);
}

TEST_P(RDFProfile, PtrWeightJit)
{
   RDataFrame df(10);
   auto dfXYW =
      df.Define("x", "rdfentry_ + 5.5").Define("y", "2 * rdfentry_ + 0.5").Define("w", "0.1 + rdfentry_ * 0.03");

   auto profile = std::make_shared<RProfile>(10, std::make_pair(5.0, 15.0));
   auto resPtr = dfXYW.Profile(profile, {"x"}, "y", "w");
   EXPECT_EQ(profile, resPtr.GetSharedPtr());
   EXPECT_EQ(profile->GetNEntries(), 10);
}

TEST_P(RDFProfile, WeightInvalidNumberOfArguments)
{
   RDataFrame df(10);
   auto dfXW = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"});

   const RRegularAxis axis(10, {5.0, 15.0});
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfXW.Profile<double, double, double, double>({axis}, {"x", "x"}, "x", "w");
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &) {
      // expected
   }

   auto profile = std::make_shared<RProfile>(axis);
   try {
      // Cannot use EXPECT_THROW because of template arguments...
      dfXW.Profile<double, double, double, double>(profile, {"x", "x"}, "x", "w");
      FAIL() << "expected std::invalid_argument";
   } catch (const std::invalid_argument &) {
      // expected
   }
}

TEST_P(RDFProfile, WeightInvalidNumberOfArgumentsJit)
{
   RDataFrame df(10);
   auto dfXW = df.Define("x", "rdfentry_ + 5.5").Define("w", "0.1 + rdfentry_ * 0.03");

   const RRegularAxis axis(10, {5.0, 15.0});
   EXPECT_THROW(dfXW.Profile({axis}, {"x", "x"}, "x", "w"), std::invalid_argument);

   auto profile = std::make_shared<RProfile>(axis);
   EXPECT_THROW(dfXW.Profile(profile, {"x", "x"}, "x", "w"), std::invalid_argument);
}

TEST_P(RDFProfile, Container)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", [](ULong64_t e) { return ROOT::RVecD{e + 5.5, e + 15.5}; }, {"rdfentry_"});

   const RRegularAxis axis(20, {5.0, 25.0});
   auto profile = dfX.Profile<ROOT::RVecD, ROOT::RVecD>({axis}, {"x"}, "x");
   EXPECT_EQ(profile->GetNEntries(), 20);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(profile->GetBinContent(index).fSum, 1.0);
   }
}

TEST_P(RDFProfile, ContainerNested)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", [](ULong64_t e) { return ROOT::RVec<ROOT::RVecD>{{e + 5.5}, {e + 15.5}}; }, {"rdfentry_"});

   const RRegularAxis axis(20, {5.0, 25.0});
   auto profile = dfX.Profile<ROOT::RVec<ROOT::RVecD>, ROOT::RVec<ROOT::RVecD>>({axis}, {"x"}, "x");
   EXPECT_EQ(profile->GetNEntries(), 20);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(profile->GetBinContent(index).fSum, 1.0);
   }
}

TEST_P(RDFProfile, ContainerJit)
{
   RDataFrame df(10);
   auto dfX = df.Define("x", "ROOT::RVecD{rdfentry_ + 5.5}");

   const RRegularAxis axis(10, {5.0, 15.0});
   auto profile = dfX.Profile({axis}, {"x"}, "x");
   EXPECT_EQ(profile->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(profile->GetBinContent(index).fSum, 1.0);
   }
}

TEST_P(RDFProfile, ContainerMultiDim)
{
   RDataFrame df(10);
   auto dfXYZ = df.Define("x", [](ULong64_t e) { return ROOT::RVecD{e + 5.5}; }, {"rdfentry_"})
                   .Define("y", [](ULong64_t e) { return ROOT::RVecD{2 * e + 0.5}; }, {"rdfentry_"})
                   .Define("z", [](ULong64_t e) { return ROOT::RVecD{3 * e + 10.5}; }, {"rdfentry_"});

   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto profile = dfXYZ.Profile<ROOT::RVecD, ROOT::RVecD, ROOT::RVecD>({regularAxis, variableBinAxis}, {"x", "y"}, "z");
   EXPECT_EQ(profile->GetNEntries(), 10);
   for (auto x : regularAxis.GetNormalRange()) {
      for (auto y : variableBinAxis.GetNormalRange()) {
         if (2 * x.GetIndex() == y.GetIndex()) {
            EXPECT_EQ(profile->GetBinContent(x, y).fSum, 1.0);
            EXPECT_EQ(profile->GetBinContent(x, y).fSumValues, 3 * x.GetIndex() + 10.5);
         } else {
            EXPECT_EQ(profile->GetBinContent(x, y).fSum, 0.0);
         }
      }
   }
}

TEST_P(RDFProfile, ContainerMultiDimBroadcast)
{
   RDataFrame df(10);
   auto dfXYZ = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                   .Define("y", [](ULong64_t e) { return ROOT::RVecD{2 * e + 0.5}; }, {"rdfentry_"})
                   .Define("z", [](ULong64_t e) { return ROOT::RVecD{3 * e + 10.5}; }, {"rdfentry_"});

   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto profile = dfXYZ.Profile<double, ROOT::RVecD, ROOT::RVecD>({regularAxis, variableBinAxis}, {"x", "y"}, "z");
   EXPECT_EQ(profile->GetNEntries(), 10);
   for (auto x : regularAxis.GetNormalRange()) {
      for (auto y : variableBinAxis.GetNormalRange()) {
         if (2 * x.GetIndex() == y.GetIndex()) {
            EXPECT_EQ(profile->GetBinContent(x, y).fSum, 1.0);
            EXPECT_EQ(profile->GetBinContent(x, y).fSumValues, 3 * x.GetIndex() + 10.5);
         } else {
            EXPECT_EQ(profile->GetBinContent(x, y).fSum, 0.0);
         }
      }
   }
}

TEST_P(RDFProfile, ContainerMultiDimNestedBroadcast)
{
   RDataFrame df(10);
   auto dfXYZ = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                   .Define("xV", [](ULong64_t e) { return ROOT::RVecD{e + 5.5}; }, {"rdfentry_"})
                   .Define("y", [](ULong64_t e) { return ROOT::RVec<ROOT::RVecD>{{2 * e + 0.5}}; }, {"rdfentry_"})
                   .Define("z", [](ULong64_t e) { return ROOT::RVecD{3 * e + 10.5}; }, {"rdfentry_"});

   const RRegularAxis regularAxis(10, {5.0, 15.0});
   static constexpr std::size_t BinsY = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   auto profile =
      dfXYZ.Profile<double, ROOT::RVec<ROOT::RVecD>, ROOT::RVecD>({regularAxis, variableBinAxis}, {"x", "y"}, "z");
   EXPECT_EQ(profile->GetNEntries(), 10);
   for (auto x : regularAxis.GetNormalRange()) {
      for (auto y : variableBinAxis.GetNormalRange()) {
         if (2 * x.GetIndex() == y.GetIndex()) {
            EXPECT_EQ(profile->GetBinContent(x, y).fSum, 1.0);
            EXPECT_EQ(profile->GetBinContent(x, y).fSumValues, 3 * x.GetIndex() + 10.5);
         } else {
            EXPECT_EQ(profile->GetBinContent(x, y).fSum, 0.0);
         }
      }
   }

   profile = dfXYZ.Profile<ROOT::RVecD, ROOT::RVec<ROOT::RVecD>, ROOT::RVecD>({regularAxis, variableBinAxis},
                                                                              {"xV", "y"}, "z");
   EXPECT_EQ(profile->GetNEntries(), 10);
}

TEST_P(RDFProfile, ContainerInvalidElements)
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
      *dfXY.Profile<ROOT::RVecD, ROOT::RVecD, ROOT::RVecD>({regularAxis, variableBinAxis}, {"x0", "y1"}, "y1");
      FAIL() << "expected std::runtime_error";
   } catch (const std::runtime_error &) {
      // expected
   }

   try {
      // Cannot use EXPECT_THROW because of template arguments...
      *dfXY.Profile<ROOT::RVecD, ROOT::RVecD, ROOT::RVecD>({regularAxis, variableBinAxis}, {"x2", "y1"}, "x2");
      FAIL() << "expected std::runtime_error";
   } catch (const std::runtime_error &e) {
      // expected
   }
}

TEST_P(RDFProfile, WeightContainer)
{
   RDataFrame df(10);
   auto dfXW =
      df.Define("xV", [](ULong64_t e) { return ROOT::RVecD{e + 5.5, e + 15.5}; }, {"rdfentry_"})
         .Define("yV", [](ULong64_t e) { return ROOT::RVecD{2 * e + 0.5, 2 * e + 2.5}; }, {"rdfentry_"})
         .Define("wV", [](ULong64_t e) { return ROOT::RVecD{0.1 + e * 0.03, 0.2 + e * e * 0.06}; }, {"rdfentry_"});

   const RRegularAxis axis(20, {5.0, 25.0});
   auto profile = dfXW.Profile<ROOT::RVecD, ROOT::RVecD, ROOT::RVecD>({axis}, {"xV"}, "yV", "wV");
   EXPECT_EQ(profile->GetNEntries(), 20);
   for (auto index : axis.GetNormalRange()) {
      auto &bin = profile->GetBinContent(index);
      auto i = index.GetIndex();
      double value = i >= 10 ? (2 * (i - 10) + 2.5) : (2 * i + 0.5);
      double weight = i >= 10 ? (0.2 + (i - 10) * (i - 10) * 0.06) : (0.1 + i * 0.03);
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
      EXPECT_FLOAT_EQ(bin.fSumValues, weight * value);
      EXPECT_FLOAT_EQ(bin.fSumValues2, weight * value * value);
   }
}

TEST_P(RDFProfile, WeightContainerJit)
{
   RDataFrame df(10);
   auto dfXYW = df.Define("xV", "ROOT::RVecD{rdfentry_ + 5.5, rdfentry_ + 15.5}")
                   .Define("yV", "ROOT::RVecD{2 * rdfentry_ + 0.5, 2 * rdfentry_ + 2.5}")
                   .Define("wV", "ROOT::RVecD{0.1 + rdfentry_ * 0.03, 0.2 + rdfentry_ * rdfentry_ * 0.06}");

   const RRegularAxis axis(20, {5.0, 25.0});
   auto profile = dfXYW.Profile({axis}, {"xV"}, "yV", "wV");
   EXPECT_EQ(profile->GetNEntries(), 20);
   for (auto index : axis.GetNormalRange()) {
      auto &bin = profile->GetBinContent(index);
      auto i = index.GetIndex();
      double value = i >= 10 ? (2 * (i - 10) + 2.5) : (2 * i + 0.5);
      double weight = i >= 10 ? (0.2 + (i - 10) * (i - 10) * 0.06) : (0.1 + i * 0.03);
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
      EXPECT_FLOAT_EQ(bin.fSumValues, weight * value);
      EXPECT_FLOAT_EQ(bin.fSumValues2, weight * value * value);
   }
}

TEST_P(RDFProfile, WeightContainerBroadcast)
{
   RDataFrame df(10);
   auto dfXYW =
      df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
         .Define("xV", [](ULong64_t e) { return ROOT::RVecD{e + 5.5, e + 15.5}; }, {"rdfentry_"})
         .Define("yV", [](ULong64_t e) { return ROOT::RVecD{2 * e + 0.5, 2 * e + 2.5}; }, {"rdfentry_"})
         .Define("w", [](ULong64_t e) { return 0.1 + e * 0.03; }, {"rdfentry_"})
         .Define("wV", [](ULong64_t e) { return ROOT::RVecD{0.1 + e * 0.03, 0.2 + e * e * 0.06}; }, {"rdfentry_"});

   const RRegularAxis axis(20, {5.0, 25.0});
   auto profile = dfXYW.Profile<ROOT::RVecD, ROOT::RVecD, double>({axis}, {"xV"}, "yV", "w");
   EXPECT_EQ(profile->GetNEntries(), 20);
   for (auto index : axis.GetNormalRange()) {
      auto &bin = profile->GetBinContent(index);
      auto i = index.GetIndex();
      double value = i >= 10 ? (2 * (i - 10) + 2.5) : (2 * i + 0.5);
      double weight = 0.1 + (i >= 10 ? i - 10 : i) * 0.03;
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
      EXPECT_FLOAT_EQ(bin.fSumValues, weight * value);
      EXPECT_FLOAT_EQ(bin.fSumValues2, weight * value * value);
   }

   profile = dfXYW.Profile<double, ROOT::RVecD, ROOT::RVecD>({axis}, {"x"}, "yV", "wV");
   EXPECT_EQ(profile->GetNEntries(), 20);
   EXPECT_EQ(profile->GetBinContent(2).fSum, 0.1 + 2 * 0.03 + 0.2 + 2 * 2 * 0.06);
}

TEST_P(RDFProfile, Variations)
{
   RDataFrame df(10);
   auto dfXY = df.Define("x", [](ULong64_t e) { return e + 5.5; }, {"rdfentry_"})
                  .Vary(
                     "x", [](double x) { return ROOT::RVecD{x - 1.0, x + 1.0}; }, {"x"}, 2)
                  .Define("y", [](ULong64_t e) { return 2 * e + 0.5; }, {"rdfentry_"});

   const RRegularAxis axis(10, {5.0, 15.0});
   auto profile = dfXY.Profile<double, double>({axis}, {"x"}, "y");
   auto vars = ROOT::RDF::Experimental::VariationsFor(profile);

   // Check nominal
   EXPECT_EQ(profile->GetNEntries(), 10);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(profile->GetBinContent(index).fSum, 1.0);
   }

   // Check variations
   EXPECT_EQ(vars.GetKeys().size(), 3); // nominal + 2 variations
   for (const auto &key : vars.GetKeys()) {
      auto &varHist = vars[key];
      EXPECT_EQ(varHist.GetNEntries(), 10);
      for (auto index : axis.GetNormalRange()) {
         if (key == "nominal") {
            EXPECT_EQ(varHist.GetBinContent(index).fSum, 1.0);
         } else if (key == "x:0") {
            EXPECT_EQ(varHist.GetBinContent(index).fSum, (index.GetIndex() == 9 ? 0.0 : 1.0));
         } else if (key == "x:1") {
            EXPECT_EQ(varHist.GetBinContent(index).fSum, (index.GetIndex() == 0 ? 0.0 : 1.0));
         } else {
            FAIL() << "Unexpected variation key: " << key;
         }
      }
   }
}

INSTANTIATE_TEST_SUITE_P(Seq, RDFProfile, ::testing::Values(false));

#ifdef R__USE_IMT
INSTANTIATE_TEST_SUITE_P(MT, RDFProfile, ::testing::Values(true));
#endif
