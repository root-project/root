// Tests for RooAbsReal
// Authors: Stephan Hageboeck, CERN 05/2020
//          Jonas Rembser, CERN 09/2022

#include <RooAbsPdf.h>
#include <RooAddPdf.h>
#include <RooAddition.h>
#include <RooBinning.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooFormulaVar.h>
#include <RooGaussian.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooPolyVar.h>
#include <RooProduct.h>
#include <RooRealVar.h>
#include <RooUniform.h>
#include <RooWorkspace.h>

#include <TFile.h>
#include <TH1.h>
#include <TMath.h>
#include <TTree.h>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

// ROOT-6882: Cannot read from ULong64_t branches.
TEST(RooAbsReal, ReadFromTree)
{
   RooRealVar x("ULong64Branch", "xx", 0, 0, 10);
   RooRealVar y("FloatBranch", "yy", 2, 0, 10);
   RooRealVar z("IntBranch", "zz", 2, 0, 10);
   RooRealVar a("DoubleBranch", "aa", 2, 0, 10);

   TFile theFile("testRooAbsReal_1.root", "READ");
   ASSERT_TRUE(theFile.IsOpen());

   TTree *tree = nullptr;
   theFile.GetObject("tree", tree);
   ASSERT_NE(tree, nullptr);

   tree->AddFriend("tree2", "testRooAbsReal_2.root");

   RooHelpers::HijackMessageStream hijack(RooFit::INFO, RooFit::DataHandling);

   RooDataSet data("data", "data", {x, y, z, a}, RooFit::Import(*tree));
   ASSERT_EQ(data.numEntries(), 4);

   EXPECT_DOUBLE_EQ(data.moment(x, 1), (1. + 3. + 3. + 7.) / 4.) << "ULong64Branch should contain {1, 3, 3, 7}";
   EXPECT_FLOAT_EQ(data.moment(y, 1), (1.f + 0.3f + 0.03f + 0.007f) / 4.f)
      << "FloatBranch should contain {1., 0.3, 0.03, 0.007}";
   EXPECT_DOUBLE_EQ(data.moment(z, 1), (1. + 3. + 3. + 7.) / 4.) << "IntBranch should contain {1, 3, 3, 7}";
   EXPECT_DOUBLE_EQ(data.moment(a, 1), (1. + 0.3 + 0.03 + 0.007) / 4.) << "DoubleBranch should contain {1, 3, 3, 7}";

   const std::string &msgs = hijack.str();
   const char *targetMsg = " will be converted to double precision";
   EXPECT_NE(msgs.find(std::string(x.GetName()) + targetMsg), std::string::npos)
      << "Expect to see INFO messages for conversion of integer branch to double.";
   EXPECT_NE(msgs.find(std::string(y.GetName()) + targetMsg), std::string::npos)
      << "Expect to see INFO messages for conversion of float branch to double.";
   EXPECT_NE(msgs.find(std::string(z.GetName()) + targetMsg), std::string::npos)
      << "Expect to see INFO messages for conversion of integer branch to double.";
   EXPECT_EQ(msgs.find(std::string(a.GetName()) + targetMsg), std::string::npos)
      << "Expect not to see INFO messages for conversion of double branch to double.";
}

/// Check that RooAbsReal::getPropagatedError() works as expected in some
/// corner cases.
TEST(RooAbsReal, ErrorPropagation)
{
   using namespace RooFit;

   RooWorkspace ws;

   ws.factory("Gaussian::gauss1(x[5, 0, 10], mu[5, 0, 10], sigma1[2, 0.1, 10.])");
   ws.factory("Gaussian::gauss2(x[5, 0, 10], mu[5, 0, 10], sigma2[2, 0.1, 10.])");

   RooRealVar &x = *ws.var("x");
   RooRealVar &mu = *ws.var("mu");
   RooAbsPdf &gauss1 = *ws.pdf("gauss1");
   RooAbsPdf &gauss2 = *ws.pdf("gauss2");

   std::unique_ptr<RooDataSet> data{gauss1.generate(x, 1000)};

   std::unique_ptr<RooFitResult> res1{gauss1.fitTo(*data, Save(), PrintLevel(-1))};

   RooArgSet normSet{x};

   x.setError(1.0);
   EXPECT_FLOAT_EQ(x.getPropagatedError(*res1, normSet), 0.0)
      << "Propagating the uncertainties to an unrelated variable should give no uncertainty, even if that variable has "
         "an intrinsic error";

   EXPECT_FLOAT_EQ(mu.getPropagatedError(*res1, normSet), mu.getError())
      << "Propagating the uncertainties to an parameter in the fit result should result in that parameters uncertainty";

   const double err1 = gauss1.getPropagatedError(*res1, normSet);
   const double err2 = gauss2.getPropagatedError(*res1, normSet);
   EXPECT_LE(err2, err1)
      << "When propagating uncertainties to another PDF that is identical to the fit model except for one parameter, "
         "the uncertainty is expected to be smaller because that parameters uncertainty is not considered";
}

// Test that we can get the yield histogram from an extended pdf with
// RooAbsReal::createHistogram() as described in the documentation.
TEST(RooAbsReal, YieldsHistogram)
{
   using namespace RooFit;

   // Define non-uniform boundaries and total yield.
   // Second bin is three times larger than the first
   double totalYield = 150.;
   std::vector<double> boundaries{0., 2.5, 10.};
   int nBins = boundaries.size() - 1;

   // Define variable and binning
   RooRealVar x{"x", "x", boundaries.front(), boundaries.back()};
   RooBinning binning{nBins, boundaries.data()};
   x.setBinning(binning);

   // Create uniform extended pdf (shape doesn't matter for this test)
   RooRealVar totalYieldVar{"total_yield", "", totalYield};
   RooUniform pdf{"pdf", "", x};
   RooAddPdf pdfExt{"pdf_ext", "", pdf, totalYieldVar};

   std::unique_ptr<TH1> yields{pdfExt.createHistogram("hist", x)};

   double c1 = yields->GetBinContent(1);
   double c2 = yields->GetBinContent(2);

   // since the underlying pdf is uniform, the yields are just proportional to
   // the bin widths.
   EXPECT_FLOAT_EQ(c1 * x.getBinWidth(1), c2 * x.getBinWidth(0)) << "relative yield is wrong";
   EXPECT_FLOAT_EQ(c1 + c2, totalYield) << "total yield is wrong";
}

/// The sum and product utility classes (RooFormulaVar, RooPolyVar,
/// RooAddition, RooProduct) used to tailor the parameters of a pdf, checked
/// against directly computed function values. Replaces the former stressRooFit
/// test based on the rf302 tutorial, which compared against stored reference
/// histograms.
TEST(RooAbsReal, UtilityFunctionsToTailorPdfs)
{
   RooRealVar x("x", "x", -5, 5);
   RooRealVar y("y", "y", -5, 5);

   RooRealVar a0("a0", "a0", -1.5, -5, 5);
   RooRealVar a1("a1", "a1", -0.5, -1, 1);
   RooRealVar sigma("sigma", "width of gaussian", 0.5);

   RooFormulaVar fy1("fy_1", "a0-a1*sqrt(10*abs(y))", RooArgSet(y, a0, a1));
   RooPolyVar fy2("fy_2", "fy_2", y, RooArgSet(a0, a1));
   RooAddition fy3("fy_3", "a0+y", RooArgSet(a0, y));
   RooProduct fy4("fy_4", "a1*y", RooArgSet(a1, y));

   RooGaussian model1("model_1", "Gaussian with shifting mean", x, fy1, sigma);
   RooGaussian model2("model_2", "Gaussian with shifting mean", x, fy2, sigma);
   RooGaussian model3("model_3", "Gaussian with shifting mean", x, fy3, sigma);
   RooGaussian model4("model_4", "Gaussian with shifting mean", x, fy4, sigma);

   // Normalized value of a Gaussian pdf in x with mean "mu", truncated to the
   // range of x.
   auto gaussPdfVal = [&](double xVal, double mu) {
      const double sqrt2 = std::sqrt(2.0);
      const double norm = 0.5 * std::sqrt(2. * TMath::Pi()) * sigma.getVal() *
                          (std::erf((x.getMax() - mu) / (sqrt2 * sigma.getVal())) -
                           std::erf((x.getMin() - mu) / (sqrt2 * sigma.getVal())));
      return std::exp(-0.5 * std::pow((xVal - mu) / sigma.getVal(), 2)) / norm;
   };

   RooArgSet normSet{x};

   for (double yVal : {-4.5, -1.2, 0., 0.7, 3.3}) {
      y.setVal(yVal);

      const double mu1 = a0.getVal() - a1.getVal() * std::sqrt(10. * std::abs(yVal));
      const double mu2 = a0.getVal() + a1.getVal() * yVal;
      const double mu3 = a0.getVal() + yVal;
      const double mu4 = a1.getVal() * yVal;

      EXPECT_NEAR(fy1.getVal(), mu1, 1e-12) << "y = " << yVal;
      EXPECT_NEAR(fy2.getVal(), mu2, 1e-12) << "y = " << yVal;
      EXPECT_NEAR(fy3.getVal(), mu3, 1e-12) << "y = " << yVal;
      EXPECT_NEAR(fy4.getVal(), mu4, 1e-12) << "y = " << yVal;

      for (double xVal : {-3., 0., 1.5}) {
         x.setVal(xVal);
         EXPECT_NEAR(model1.getVal(&normSet), gaussPdfVal(xVal, mu1), 1e-9) << "(x, y) = (" << xVal << ", " << yVal << ")";
         EXPECT_NEAR(model2.getVal(&normSet), gaussPdfVal(xVal, mu2), 1e-9) << "(x, y) = (" << xVal << ", " << yVal << ")";
         EXPECT_NEAR(model3.getVal(&normSet), gaussPdfVal(xVal, mu3), 1e-9) << "(x, y) = (" << xVal << ", " << yVal << ")";
         EXPECT_NEAR(model4.getVal(&normSet), gaussPdfVal(xVal, mu4), 1e-9) << "(x, y) = (" << xVal << ", " << yVal << ")";
      }
   }
}
