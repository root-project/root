#include <RooAddPdf.h>
#include <RooArgSet.h>
#include <RooDataSet.h>
#include <RooExponential.h>
#include <RooGaussian.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooStats/BayesianCalculator.h>
#include <RooStats/SimpleInterval.h>
#include <RooUniform.h>

#include <gtest/gtest.h>

/// Regression test for https://github.com/root-project/root/issues/17567

TEST(BayesianCalculator, UpperLimitNoNuisance_Issue17567)
{
   RooRandom::randomGenerator()->SetSeed(12345);

   // Background-only generation model.
   RooRealVar mass("mass", "mass", 6500, 7500);
   RooRealVar slopeGen("slopeGen", "slopeGen", -8e-4);
   RooExponential expoGen("expoGen", "expoGen", mass, slopeGen);
   std::unique_ptr<RooDataSet> data{expoGen.generate(mass, 1000)};

   // Fit model: Gaussian signal on top of exponential background.
   // All parameters except the POI ngauss are fixed so there are no nuisances.
   RooRealVar mean("mean", "mean", 7024);
   RooRealVar sigma("sigma", "sigma", 44);
   RooGaussian gauss("gauss", "gauss", mass, mean, sigma);
   RooRealVar slope("slope", "slope", -8e-4);
   RooExponential expo("expo", "expo", mass, slope);
   RooRealVar ngauss("ngauss", "ngauss", 10, 0, 50);
   RooRealVar nexpo("nexpo", "nexpo", 1e3);
   RooAddPdf pdftot("pdftot", "pdftot", RooArgList(gauss, expo), RooArgList(ngauss, nexpo));

   RooUniform prior("prior", "prior", ngauss);

   // Run with the default interval method (numerical integration instead of
   // brute-force scan).
   RooStats::BayesianCalculator bc(*data, pdftot, RooArgSet(ngauss), prior);
   bc.SetConfidenceLevel(0.95);
   bc.SetLeftSideTailFraction(0.);
   std::unique_ptr<RooStats::SimpleInterval> bcInterval{bc.GetInterval()};
   ASSERT_NE(bcInterval, nullptr);
   const double limitDefault = bcInterval->UpperLimit();
   EXPECT_EQ(bcInterval->LowerLimit(), 0.0);

   // Reference: explicit scan with many bins gives the same answer up to
   // numerical integration and scan discretization errors.
   RooStats::BayesianCalculator bcScan(*data, pdftot, RooArgSet(ngauss), prior);
   bcScan.SetConfidenceLevel(0.95);
   bcScan.SetLeftSideTailFraction(0.);
   bcScan.SetScanOfPosterior(200);
   std::unique_ptr<RooStats::SimpleInterval> scanInterval{bcScan.GetInterval()};
   ASSERT_NE(scanInterval, nullptr);
   const double limitScan = scanInterval->UpperLimit();

   // Both limits should sit well inside the POI range (if the old bug were
   // present the limit would be at or near poi->getMax() = 50, or pulled down
   // by a broken CDF root) and must agree within 1%.
   EXPECT_GT(limitDefault, 20.0);
   EXPECT_LT(limitDefault, 35.0);
   EXPECT_NEAR(limitDefault, limitScan, 0.01 * limitScan);
}
