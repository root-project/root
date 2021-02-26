// Tests for the RooDataHist
// Authors: Stephan Hageboeck, CERN  01/2019

#include "RooDataHist.h"
#include "RooGlobalFunc.h"
#include "RooRealVar.h"
#include "RooHelpers.h"
#include "RooCategory.h"
#include "RunContext.h"

#include "TH1D.h"
#include "TH2D.h"
#include "TMath.h"
#include "TFile.h"
#include "ROOT/RMakeUnique.hxx"

#include "gtest/gtest.h"

/// ROOT-8163
/// The RooDataHist warns that it has to adjust the binning of x to the next bin boundary
/// although the boundaries match perfectly.
TEST(RooDataHist, BinningRangeCheck_8163)
{
  RooHelpers::HijackMessageStream hijack(RooFit::INFO, RooFit::DataHandling, "dataHist");

  RooRealVar x("x", "x", 0., 1.);
  TH1D hist("hist", "", 10, 0., 1.);

  RooDataHist dataHist("dataHist", "", RooArgList(x), &hist);
  EXPECT_TRUE(hijack.str().empty()) << "Messages issued were: " << hijack.str();
}



double computePoissonUpper(double weight) {
  double upperLimit = weight;
  double CL;
  do {
    upperLimit += 0.001;
    CL = 0.;
    for (unsigned int i = 0; i <= (unsigned int)weight; ++i) {
      CL += TMath::PoissonI(i, upperLimit);
    }
//    std::cout << "Upper=" << upperLimit << "\tCL=" << CL << std::endl;
  } while (CL > 0.16);

  return upperLimit;
}

double computePoissonLower(double weight) {
  double lowerLimit = weight;
  double CL;
  do {
    CL = 0.;
    lowerLimit -= 0.001;
    for (unsigned int i = 0; i < (unsigned int)weight; ++i) {
      CL += TMath::PoissonI(i, lowerLimit);
    }
  } while (CL < 1. - 0.16);

  return lowerLimit;
}

TEST(RooDataHist, UnWeightedEntries)
{
  RooRealVar x("x", "x", -10., 10.);
  x.setBins(20);
  RooRealVar w("w", "w", 0., 10.);
  RooArgSet coordinates(x);

  constexpr double targetBinContent = 10.;
  RooDataHist dataHist("dataHist", "", RooArgList(x));
  for (unsigned int i=0; i < 200; ++i) {
    x.setVal(-10. + 20./200. * i);
    dataHist.add(coordinates);
  }

  EXPECT_EQ(dataHist.numEntries(), 20);
  ASSERT_EQ(dataHist.sumEntries(), 20 * targetBinContent);

  x.setVal(0.);
  RooArgSet* coordsAtZero = dataHist.get(coordinates)->snapshot();
  x.setVal(0.9);
  RooArgSet* coordsAtPoint9 = dataHist.get(coordinates)->snapshot();
  EXPECT_EQ(static_cast<RooRealVar*>(coordsAtZero->find(x))->getVal(),
            static_cast<RooRealVar*>(coordsAtPoint9->find(x))->getVal());

  const double weight = dataHist.weight();
  EXPECT_EQ(weight, targetBinContent);

  EXPECT_NEAR(dataHist.weightError(RooAbsData::Poisson),
      sqrt(targetBinContent), 0.7); // TODO What is the actual value?

  {
    double lo, hi;
    dataHist.weightError(lo, hi, RooAbsData::Poisson);
    EXPECT_LT(lo, hi);
    EXPECT_NEAR(lo, weight - computePoissonLower(weight), 3.E-2);
    EXPECT_NEAR(hi, computePoissonUpper(weight) - weight, 3.E-2);
  }

  EXPECT_NEAR(dataHist.weightError(RooAbsData::SumW2),
      sqrt(targetBinContent), 1.E-14);

  {
    double lo, hi;
    dataHist.weightError(lo, hi, RooAbsData::SumW2);
    EXPECT_NEAR(lo, sqrt(targetBinContent), 1.E-14);
    EXPECT_NEAR(lo, sqrt(targetBinContent), 1.E-14);
    EXPECT_EQ(lo, hi);
  }


  RooArgSet* coordsAt10 = dataHist.get(10)->snapshot();
  const double weightBin10 = dataHist.weight();

  EXPECT_NEAR(static_cast<RooRealVar*>(coordsAt10->find(x))->getVal(), 0.5, 1.E-1);
  EXPECT_EQ(weight, weightBin10);
}


TEST(RooDataHist, WeightedEntries)
{
  RooRealVar x("x", "x", -10., 10.);
  x.setBins(20);
  RooRealVar w("w", "w", 0., 10.);
  RooArgSet coordinates(x);

  constexpr double targetBinContent = 20.;
  RooDataHist dataHist("dataHist", "", RooArgList(x));
  for (unsigned int i=0; i < 200; ++i) {
    x.setVal(-10. + 20./200. * i);
    dataHist.add(coordinates, 2.);
  }


  EXPECT_EQ(dataHist.numEntries(), 20);
  EXPECT_EQ(dataHist.sumEntries(), 20 * targetBinContent);

  x.setVal(0.);
  dataHist.get(coordinates)->snapshot();
  const double weight = dataHist.weight();
  ASSERT_EQ(weight, targetBinContent);


  const double targetError = sqrt(10*4.);

  EXPECT_NEAR(dataHist.weightError(RooAbsData::Poisson),
      targetError, 1.5); // TODO What is the actual value?

  {
    double lo, hi;
    dataHist.weightError(lo, hi, RooAbsData::Poisson);
    EXPECT_LT(lo, hi);
    EXPECT_NEAR(lo, weight - computePoissonLower(weight), 3.E-2);
    EXPECT_NEAR(hi, computePoissonUpper(weight) - weight, 3.E-2);
  }

  EXPECT_NEAR(dataHist.weightError(RooAbsData::SumW2),
      targetError, 1.E-14);

  {
    double lo, hi;
    dataHist.weightError(lo, hi, RooAbsData::SumW2);
    EXPECT_NEAR(lo, targetError, 1.E-14);
    EXPECT_NEAR(lo, targetError, 1.E-14);
    EXPECT_EQ(lo, hi);
  }


  RooArgSet* coordsAt10 = dataHist.get(10)->snapshot();
  const double weightBin10 = dataHist.weight();

  EXPECT_NEAR(static_cast<RooRealVar*>(coordsAt10->find(x))->getVal(), 0.5, 1.E-1);
  EXPECT_EQ(weight, weightBin10);
}

TEST(RooDataHist, ReadV4Legacy)
{
  TFile v4Ref("dataHistv4_ref.root");
  ASSERT_TRUE(v4Ref.IsOpen());

  RooDataHist* legacy = nullptr;
  v4Ref.GetObject("dataHist", legacy);
  ASSERT_NE(legacy, nullptr);

  RooDataHist& dataHist = *legacy;

  constexpr double targetBinContent = 20.;

  RooArgSet* legacyVals = dataHist.get(10)->snapshot();
  EXPECT_EQ(static_cast<RooAbsReal*>(legacyVals->find("x"))->getVal(),
      static_cast<RooAbsReal*>(dataHist.get(10)->find("x"))->getVal());


  EXPECT_EQ(dataHist.numEntries(), 20);
  EXPECT_EQ(dataHist.sumEntries(), 20 * targetBinContent);

  static_cast<RooRealVar*>(legacyVals->find("x"))->setVal(0.);
  dataHist.get(*legacyVals); // trigger side effect for weight below.
  const double weight = dataHist.weight();
  ASSERT_EQ(weight, targetBinContent);


  const double targetError = sqrt(10*4.);

  EXPECT_NEAR(dataHist.weightError(RooAbsData::Poisson), targetError, 1.5); // TODO What is the actual value?

  {
    double lo, hi;
    dataHist.weightError(lo, hi, RooAbsData::Poisson);
    EXPECT_LT(lo, hi);
    EXPECT_NEAR(lo, weight - computePoissonLower(weight), 3.E-2);
    EXPECT_NEAR(hi, computePoissonUpper(weight) - weight, 3.E-2);
  }

  EXPECT_NEAR(dataHist.weightError(RooAbsData::SumW2),
      targetError, 1.E-14);

  {
    double lo, hi;
    dataHist.weightError(lo, hi, RooAbsData::SumW2);
    EXPECT_NEAR(lo, targetError, 1.E-14);
    EXPECT_NEAR(lo, targetError, 1.E-14);
    EXPECT_EQ(lo, hi);
  }


  RooArgSet* coordsAt10 = dataHist.get(10)->snapshot();
  const double weightBin10 = dataHist.weight();

  EXPECT_NEAR(static_cast<RooRealVar*>(coordsAt10->find("x"))->getVal(), 0.5, 1.E-1);
  EXPECT_EQ(weight, weightBin10);
}


void fillHist(TH2D* histo, double content) {
  for (int i=0; i < histo->GetNbinsX()+2; ++i) {
    for (int j=0; j < histo->GetNbinsY()+2; ++j) {
      histo->SetBinContent(i, j, content + (i-1)*100 + (j-1));
    }
  }
}

TEST(RooDataHist, BatchDataAccess) {
  RooRealVar x("x", "x", 0, -10, 10);
  RooRealVar y("y", "y", 1, 0, 20);

  auto histo = std::make_unique<TH2D>("xyHist", "xyHist", 20, -10, 10, 10, 0, 10);
  fillHist(histo.get(), 0.1);

  RooDataHist dataHist("dataHist", "Data histogram with batch access",
      RooArgSet(x, y), RooFit::Import(*histo));

  const std::size_t numEntries = static_cast<std::size_t>(dataHist.numEntries());
  ASSERT_EQ(numEntries, 200ul);

  const RooArgSet& vars = *dataHist.get();
  auto xp = dynamic_cast<RooRealVar*>(vars[0ul]);
  auto yp = dynamic_cast<RooRealVar*>(vars[1]);
  ASSERT_STREQ(xp->GetName(), "x");
  ASSERT_STREQ(yp->GetName(), "y");

  RooBatchCompute::RunContext evalDataShort{};
  dataHist.getBatches(evalDataShort, 0, 100);
  RooBatchCompute::RunContext evalData{};
  dataHist.getBatches(evalData, 0, numEntries);

  auto xBatchShort = xp->getValues(evalDataShort);
  auto xBatch = xp->getValues(evalData);
  auto yBatch = yp->getValues(evalData);

  ASSERT_FALSE(xBatchShort.empty());
  ASSERT_FALSE(xBatch.empty());
  ASSERT_FALSE(yBatch.empty());

  EXPECT_EQ(xBatchShort.size(), 100ul);
  EXPECT_EQ(xBatch.size(), numEntries);
  EXPECT_EQ(yBatch.size(), numEntries);

  EXPECT_EQ(xBatch[5 * 10], histo->GetXaxis()->GetBinCenter(5+1));
  EXPECT_EQ(yBatch[5], histo->GetYaxis()->GetBinCenter(5+1));

  auto weights = dataHist.getWeightBatch(0, numEntries);
  ASSERT_FALSE(weights.empty());
  ASSERT_EQ(weights.size(), numEntries);
  EXPECT_EQ(weights[2], 0.1 + 2.);
  EXPECT_EQ(weights[4*10 + 7], 0.1 + 400. + 7.);
}


void fillHist(TH1D* histo, double content) {
  for (int i=0; i < histo->GetNbinsX()+2; ++i) {
    histo->SetBinContent(i, content + i-1);
  }
}

TEST(RooDataHist, BatchDataAccessWithCategories) {
  RooRealVar x("x", "x", 0, -10, 10);
  RooCategory cat("cat", "category");

  auto histoX = std::make_unique<TH1D>("xHist", "xHist", 20, -10., 10.);
  auto histoY = std::make_unique<TH1D>("yHist", "yHist", 20, -10., 10.);
  fillHist(histoX.get(), 0.2);
  fillHist(histoY.get(), 0.3);

  RooDataHist dataHist("dataHist", "Data histogram with batch access",
      RooArgSet(x), RooFit::Index(cat), RooFit::Import("catX", *histoX), RooFit::Import("catY", *histoY));

  const std::size_t numEntries = (std::size_t)dataHist.numEntries();
  ASSERT_EQ(numEntries, 40ul);

  const RooArgSet& vars = *dataHist.get();
  auto catp = dynamic_cast<RooCategory*>(vars[0ul]);
  auto xp = dynamic_cast<RooRealVar*>(vars[1]);
  ASSERT_NE(catp, nullptr);
  ASSERT_NE(xp, nullptr);
  ASSERT_STREQ(catp->GetName(), "cat");
  ASSERT_STREQ(xp->GetName(), "x");

  RooBatchCompute::RunContext evalDataShort{};
  dataHist.getBatches(evalDataShort, 0, 10);
  RooBatchCompute::RunContext evalData{};
  dataHist.getBatches(evalData, 0, numEntries);

  auto xBatchShort = xp->getValues(evalDataShort);
  auto xBatch      = xp->getValues(evalData);

  ASSERT_FALSE(xBatchShort.empty());
  ASSERT_FALSE(xBatch.empty());

  EXPECT_EQ(xBatchShort.size(), 10ul);
  EXPECT_EQ(xBatch.size(), numEntries);

  EXPECT_EQ(xBatch[15], histoX->GetXaxis()->GetBinCenter(15+1));
  EXPECT_EQ(xBatch[35], histoX->GetXaxis()->GetBinCenter(15+1));

  auto weights = dataHist.getWeightBatch(0, numEntries);
  ASSERT_FALSE(weights.empty());
  ASSERT_EQ(weights.size(), numEntries);
  EXPECT_EQ(weights[2], 0.2 + 2.);
  EXPECT_EQ(weights[23], 0.3 + 3.);
}


TEST(RooDataHist, BatchDataAccessWithCategoriesAndFitRange) {
  RooRealVar x("x", "x", 0, -10, 10);
  RooCategory cat("cat", "category");
  x.setRange(-8., 5);

  auto histoX = std::make_unique<TH1D>("xHist", "xHist", 20, -10., 10.);
  auto histoY = std::make_unique<TH1D>("yHist", "yHist", 20, -10., 10.);
  fillHist(histoX.get(), 0.2);
  fillHist(histoY.get(), 0.3);

  RooDataHist dataHist("dataHist", "Data histogram with batch access",
      RooArgSet(x), RooFit::Index(cat), RooFit::Import("catX", *histoX), RooFit::Import("catY", *histoY));

  dataHist.cacheValidEntries();

  const std::size_t numEntries = (std::size_t)dataHist.numEntries();
  ASSERT_EQ(numEntries, 26ul);

  const RooArgSet& vars = *dataHist.get();
  auto catp = dynamic_cast<RooCategory*>(vars[0ul]);
  auto xp = dynamic_cast<RooRealVar*>(vars[1]);
  ASSERT_NE(catp, nullptr);
  ASSERT_NE(xp, nullptr);
  ASSERT_STREQ(catp->GetName(), "cat");
  ASSERT_STREQ(xp->GetName(), "x");

  RooBatchCompute::RunContext evalDataShort{};
  dataHist.getBatches(evalDataShort, 0, 10);
  RooBatchCompute::RunContext evalData{};
  dataHist.getBatches(evalData, 0, numEntries);

  auto xBatchShort = xp->getValues(evalDataShort);
  auto xBatch      = xp->getValues(evalData);

  ASSERT_FALSE(xBatchShort.empty());
  ASSERT_FALSE(xBatch.empty());

  EXPECT_EQ(xBatchShort.size(), 10ul);
  EXPECT_EQ(xBatch.size(), numEntries);

  EXPECT_TRUE(std::all_of(xBatch.begin(), xBatch.end(), [](double arg){return -8. < arg && arg < 5.;}));

  EXPECT_EQ(xBatch[ 0],  -7.5);
  EXPECT_EQ(xBatch[12],  4.5);
  EXPECT_EQ(xBatch[13], -7.5);
  EXPECT_EQ(xBatch[25],  4.5);

  auto weights = dataHist.getWeightBatch(0, numEntries);
  ASSERT_FALSE(weights.empty());
  ASSERT_EQ(weights.size(), numEntries);
  EXPECT_TRUE(std::none_of(weights.begin(), weights.end(), [](double arg){return arg == 0.;}));
}


TEST(RooDataHist, BatchDataAccessWithCategoriesAndFitRangeWithMasking) {
  RooRealVar x("x", "x", 0, -10, 10);
  RooCategory cat("cat", "category");

  auto histoX = std::make_unique<TH1D>("xHist", "xHist", 20, -10., 10.);
  auto histoY = std::make_unique<TH1D>("yHist", "yHist", 20, -10., 10.);
  fillHist(histoX.get(), 0.2);
  fillHist(histoY.get(), 0.3);

  RooDataHist dataHist("dataHist", "Data histogram with batch access",
      RooArgSet(x), RooFit::Index(cat), RooFit::Import("catX", *histoX), RooFit::Import("catY", *histoY));

  const RooArgSet& vars = *dataHist.get();
  auto catp = dynamic_cast<RooCategory*>(vars[0ul]);
  auto xp = dynamic_cast<RooRealVar*>(vars[1]);
  ASSERT_NE(catp, nullptr);
  ASSERT_NE(xp, nullptr);
  ASSERT_STREQ(catp->GetName(), "cat");
  ASSERT_STREQ(xp->GetName(), "x");

  xp->setRange(-8., 5.);
  dataHist.cacheValidEntries();

  const std::size_t numEntries = (std::size_t)dataHist.numEntries();
  ASSERT_EQ(numEntries, 40ul);

  RooBatchCompute::RunContext evalDataShort{};
  dataHist.getBatches(evalDataShort, 0, 10);
  RooBatchCompute::RunContext evalData{};
  dataHist.getBatches(evalData, 0, numEntries);

  auto xBatchShort = xp->getValues(evalDataShort);
  auto xBatch      = xp->getValues(evalData);

  ASSERT_FALSE(xBatchShort.empty());
  ASSERT_FALSE(xBatch.empty());

  EXPECT_EQ(xBatchShort.size(), 10ul);
  EXPECT_EQ(xBatch.size(), numEntries);

  EXPECT_EQ(xBatch[15], histoX->GetXaxis()->GetBinCenter(15+1));
  EXPECT_EQ(xBatch[35], histoX->GetXaxis()->GetBinCenter(15+1));

  auto weights = dataHist.getWeightBatch(0, numEntries);
  ASSERT_FALSE(weights.empty());
  ASSERT_EQ(weights.size(), numEntries);
  EXPECT_TRUE(std::any_of(weights.begin(), weights.end(), [](double arg){return arg == 0.;}));

  for (unsigned int i=0; i < numEntries; ++i) {
    EXPECT_TRUE((-8. < xBatch[i] && xBatch[i] < 5.) || weights[i] == 0.);
  }
}
