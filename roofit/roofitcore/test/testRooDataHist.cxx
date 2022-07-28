// Tests for the RooDataHist
// Authors: Stephan Hageboeck, CERN  01/2019
//          Jonas Rembser, CERN 02/2020

#include "RooDataHist.h"
#include "RooGlobalFunc.h"
#include "RooRealVar.h"
#include "RooHelpers.h"
#include "RooCategory.h"
#include "RooHistFunc.h"
#include "RooHistPdf.h"

#include "TH1D.h"
#include "TH2D.h"
#include "TMath.h"
#include "TFile.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>

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

class RooDataHistIO : public testing::TestWithParam<const char*> {
public:
  void SetUp() override {
    TFile file(GetParam(), "READ");
    ASSERT_TRUE(file.IsOpen());

    file.GetObject("dataHist", legacy);
    ASSERT_NE(legacy, nullptr);
  }

  void TearDown() override {
    delete legacy;
  }

protected:
  RooDataHist* legacy{nullptr};
};

TEST_P(RooDataHistIO, ReadLegacy) {

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


INSTANTIATE_TEST_SUITE_P(IO_SchemaEvol, RooDataHistIO,
    testing::Values("dataHistv4_ref.root", "dataHistv5_ref.root", "dataHistv6_ref.root"));


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

  auto evalData = dataHist.getBatches(0, numEntries);

  auto xBatchShort = dataHist.getBatches(0, 100)[xp];
  auto xBatch = evalData[xp];
  auto yBatch = evalData[yp];

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

  auto xBatchShort = dataHist.getBatches(0, 10)[xp];
  auto xBatch      = dataHist.getBatches(0, numEntries)[xp];

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

  auto xBatchShort = dataHist.getBatches(0, 10)[xp];
  auto xBatch      = dataHist.getBatches(0, numEntries)[xp];

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

  auto xBatchShort = dataHist.getBatches(0, 10)[xp];
  auto xBatch      = dataHist.getBatches(0, numEntries)[xp];

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

double integrate(RooAbsReal &absReal, const RooArgSet &iset, const RooArgSet &nset, const char *rangeName = 0)
{
   return absReal.createIntegral(iset, nset, rangeName)->getVal();
}

double integrate(RooAbsReal &absReal, const RooArgSet &iset, const char *rangeName = 0)
{
   return absReal.createIntegral(iset, rangeName)->getVal();
}

double integrateTH1DAsFunction(TH1D &hist)
{
   double result = 0;
   for (int i = 1; i < hist.GetNbinsX() + 1; ++i) {
      result += hist.GetBinContent(i) * hist.GetBinWidth(i);
   }
   return result;
}

TEST(RooDataHist, AnalyticalIntegration)
{
   // The RooDataHit can be analytically integrated with the RooDataHist::sum()
   // member functions.  This functionality is used in the analytical
   // integration capabilities of RooHistPdf and RooHistFunc.  Therefore, to
   // test these two classes at the same time, this tests validated
   // RooDataHist::sum() via the RooHistPdf and RooHistFunc interfaces.

   // The histograms for this example are kept simple such that one can easily
   // calculate the expected values with a calculator.

   // We first create an easy non-uniform histogram for the x variable so we
   // can know what we expect as test results analytically.  The histogram will
   // have the following bins with contents:

   //  - bin1 [0.0, 1.0]: 3 counts (bin volume x counts = 3)
   //  - bin2 [1.0, 3.0]: 1 count  (bin volume x counts = 2)
   //  - bin3 [3.0, 3.5]: 8 counts (bin volume x counts = 4)

   RooRealVar x("x", "x", 0, 3.5);
   x.setRange("R1", 1.0, 3.0);  // subrange that respects the bin edges
   x.setRange("R2", 0.5, 3.25); // subrange that slices throught the bins

   RooRealVar y("y", "y", 0.5, 0, 3.5);
   y.setRange("R1", 0, 2.5); // subrange that respects the bin edges
   y.setRange("R2", 0, 3.3); // subrange that slices throught the bins

   RooArgSet bothXandY{x, y};

   // histogram bin and content information
   std::vector<double> xEdges{0.0, 1.0, 3.0, 3.5};
   std::vector<double> yEdges{0.0, 2.5, 3.5};
   std::vector<double> yEdgesTrivial{0.0, 3.5};

   std::vector<double> xVals{0.5, 2.0, 3.1};
   std::vector<double> xWeights{3, 1, 8};
   std::vector<double> xValsInSecondYBin{0.5, 2.0, 3.1};
   std::vector<double> xWeightsInSecondYBin{1, 2, 1};

   TH1D h1{"h1", "h1", 3, xEdges.data()};
   for (size_t i = 0; i < xVals.size(); ++i) {
      h1.Fill(xVals[i], xWeights[i]);
   }

   RooDataHist dh1("dh1", "dh1", x, &h1);

   // test the RooHistFunc

   RooHistFunc hf1("hf1", "hf1", x, dh1);

   // RooHistFunc integrals are unnormalized, so no need to pass normalization sets
   EXPECT_FLOAT_EQ(integrate(hf1, x), integrateTH1DAsFunction(h1));
   EXPECT_FLOAT_EQ(integrate(hf1, x, "R1"), 2.0);
   EXPECT_FLOAT_EQ(integrate(hf1, x, "R2"), 5.5);

   // test the RooHistPdf
   RooHistPdf hpdf1("hpdf1", "hpdf1", x, dh1);

   EXPECT_FLOAT_EQ(integrate(hpdf1, x, x), 1.0);
   EXPECT_FLOAT_EQ(integrate(hpdf1, x, x, "R1"), h1.Integral(2, 2) / h1.Integral());
   EXPECT_FLOAT_EQ(integrate(hpdf1, x, x, "R2"), 6.5 / 12);

   {

      // Now test the simple 2D case where there just an additional dummy variable y that is always in the first bin
      // This should consistently give the same results as the 1D case.

      TH2D h2trivial{"h2trivial", "h2trivial", 3, xEdges.data(), 1, yEdgesTrivial.data()};
      for (size_t i = 0; i < xVals.size(); ++i) {
         h2trivial.Fill(xVals[i], 0.5, xWeights[i]);
      }

      RooDataHist dh2trivial("dh2trivial", "dh2trivial", bothXandY, &h2trivial);

      // test RooHistFunc
      RooHistFunc hf2trivial("hf2trivial", "hf2trivial", bothXandY, dh2trivial);

      EXPECT_FLOAT_EQ(integrate(hf2trivial, x), integrate(hf1, x));
      EXPECT_FLOAT_EQ(integrate(hf2trivial, x, "R1"), integrate(hf1, x, "R1"));
      EXPECT_FLOAT_EQ(integrate(hf2trivial, x, "R2"), integrate(hf1, x, "R2"));

      // test RooHistPdf
      RooHistPdf hpdf2trivial("hpdf2trivial", "hpdf2trivial", bothXandY, dh2trivial);

      EXPECT_FLOAT_EQ(integrate(hpdf2trivial, x, x), integrate(hpdf1, x, x));
      EXPECT_FLOAT_EQ(integrate(hpdf2trivial, x, x, "R1"), integrate(hpdf1, x, x, "R1"));
      EXPECT_FLOAT_EQ(integrate(hpdf2trivial, x, x, "R2"), integrate(hpdf1, x, x, "R2"));
   }

   // Now test the complete 2D case where the y variable is also distributed
   // with non-uniform binning.  To make things simple, the histogram has only
   // 2 bins.
   //
   // The x-histogram will have the following content for the entries where y
   // is in the [0, 2.5] bin (12 entries):
   //
   //  - bin1 [0.0, 1.0]: 3 counts (bin volume x counts = 3)
   //  - bin2 [1.0, 3.0]: 1 count  (bin volume x counts = 2)
   //  - bin3 [3.0, 3.5]: 8 counts (bin volume x counts = 4)
   //
   // Then, there are some more entries with y in the [2.5, 3.5] bin (4 entries);
   //
   //  - bin6 [0.0, 1.0]: 1 counts (bin volume x counts = 1)
   //  - bin7 [1.0, 3.0]: 2 counts  (bin volume x counts = 4)
   //  - bin8 [3.0, 3.5]: 1 counts (bin volume x counts = 0.5)

   TH2D h2{"h2", "h2", 3, xEdges.data(), 2, yEdges.data()};
   for (size_t i = 0; i < xVals.size(); ++i) {
      h2.Fill(xVals[i], 0.5, xWeights[i]);
      h2.Fill(xValsInSecondYBin[i], 3.0, xWeightsInSecondYBin[i]);
   }

   RooDataHist dh2("dh2", "dh2", bothXandY, &h2);

   // test RooHistFunc
   RooHistFunc hf2("hf2", "hf2", bothXandY, dh2);

   y.setVal(2.0);
   EXPECT_FLOAT_EQ(integrate(hf2, bothXandY), 28);
   EXPECT_FLOAT_EQ(integrate(hf2, bothXandY, "R1"), 5.);
   EXPECT_FLOAT_EQ(integrate(hf2, bothXandY, "R2"), 17.55);

   y.setVal(0.5);
   EXPECT_FLOAT_EQ(integrate(hf2, x), integrate(hf1, x));
   EXPECT_FLOAT_EQ(integrate(hf2, x, "R1"), integrate(hf1, x, "R1"));
   EXPECT_FLOAT_EQ(integrate(hf2, x, "R2"), integrate(hf1, x, "R2"));

   x.setVal(0.5);
   EXPECT_FLOAT_EQ(integrate(hf2, y), 8.5);
   EXPECT_FLOAT_EQ(integrate(hf2, y, "R1"), 7.5);
   EXPECT_FLOAT_EQ(integrate(hf2, y, "R2"), 8.3);

   // test RooHistPdf
   RooHistPdf hpdf2("hpdf2", "hpdf2", bothXandY, dh2);

   EXPECT_FLOAT_EQ(integrate(hpdf2, x, x), 1.);
   EXPECT_FLOAT_EQ(integrate(hpdf2, x, x, "R1"), 1. / 12.);
   EXPECT_FLOAT_EQ(integrate(hpdf2, x, x, "R2"), 6.5 / 12.);

   // Here one should not forget to divide by the bin volume of the slice set,
   // which is y = 0.5 in the first bin with width 2.5
   const double normXoverXY = h2.Integral(1, 3, 1, 1) / (2.5 * h2.Integral());
   EXPECT_FLOAT_EQ(integrate(hpdf2, x, bothXandY), normXoverXY);
   EXPECT_FLOAT_EQ(integrate(hpdf2, x, bothXandY, "R1"), 1. / 12. * normXoverXY);
   EXPECT_FLOAT_EQ(integrate(hpdf2, x, bothXandY, "R2"), 6.5 / 12. * normXoverXY);

   EXPECT_FLOAT_EQ(integrate(hpdf2, y, y), 1.);
   EXPECT_FLOAT_EQ(integrate(hpdf2, y, y, "R1"), 3.0 / 4.);
   EXPECT_FLOAT_EQ(integrate(hpdf2, y, y, "R2"), 3.8 / 4.);

   // Here one should not forget to divide by the bin volume of the slice set,
   // which is x = 0.5 in the first bin with width 1.0
   const double normYoverXY = h2.Integral(1, 1, 1, 2) / (1.0 * h2.Integral());
   EXPECT_FLOAT_EQ(integrate(hpdf2, y, bothXandY), 1 * normYoverXY);
   EXPECT_FLOAT_EQ(integrate(hpdf2, y, bothXandY, "R1"), 3.0 / 4. * normYoverXY);
   EXPECT_FLOAT_EQ(integrate(hpdf2, y, bothXandY, "R2"), 3.8 / 4. * normYoverXY);
}


TEST(RooDataHist, Interpolation2DSimple)
{
   TH2D hist{"hist", "hist", 2, 0, 2, 2, 0, 2};

   double a0 = 1;
   double b0 = 2;
   double a1 = 3;
   double b1 = 4;

   for(int i = 0; i < a0; ++i) {
      hist.Fill(0.5, 0.5);
   }
   for(int i = 0; i < b0; ++i) {
      hist.Fill(1.5, 0.5);
   }
   for(int i = 0; i < a1; ++i) {
      hist.Fill(0.5, 1.5);
   }
   for(int i = 0; i < b1; ++i) {
      hist.Fill(1.5, 1.5);
   }

   RooRealVar x("x", "x", 0, 2);
   RooRealVar y("y", "y", 0, 2);

   RooDataHist dataHist("data", "dataset with (x,y)", RooArgList(x, y), &hist);

   std::vector<double> values;
   int n = 5;
   for (int i = 0; i <= n; ++i) {
      values.push_back((i * 2.0) / n);
   }

   auto clamp = [](double v, double a, double b) {
      return std::min(std::max(v, a), b);
   };

   auto getTrueWeight = [&]() {
      double xVal = x.getVal();
      double yVal = y.getVal();

      double mix0 = (clamp(xVal, 0.5, 1.5) - 0.5) * (b0 - a0) + a0;
      double mix1 = (clamp(xVal, 0.5, 1.5) - 0.5) * (b1 - a1) + a1;
      return (clamp(yVal, 0.5, 1.5) - 0.5) * (mix1 - mix0) + mix0;
   };

   for (auto xVal : values) {
      for (auto yVal : values) {
         x.setVal(xVal);
         y.setVal(yVal);

         EXPECT_FLOAT_EQ(dataHist.weight({x, y}, 1), getTrueWeight());
      }
   }
}
