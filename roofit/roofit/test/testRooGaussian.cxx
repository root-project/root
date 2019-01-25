// Tests for the RooGaussian
// Authors: Stephan Hageboeck, CERN  01/2019

#include "RooRealVar.h"
#include "RooGaussian.h"

#include "gtest/gtest.h"


TEST(RooGaussian, AnalyticalIntegral)
{ 
  //The last working implementation was accurate up to ~ 1.E-9
  constexpr double prec = 1.E-8;
  constexpr double oneSig = 0.682689492137;
  constexpr double twoSig = 0.954499736104;
  constexpr double thrSig = 0.997300203937;

  RooRealVar x("x", "x", 0.);
  RooRealVar mean("mean", "mean", 0.);
  RooRealVar sig("sig", "sig", 1.);
  RooGaussian gaus("gaus", "gaus", x, mean, sig);

  x.setRange("range", -1.E30, 1.E30);

  std::unique_ptr<RooAbsReal> integral(gaus.createIntegral(x,
      RooFit::NormSet(x), RooFit::Range("range")));

  EXPECT_DOUBLE_EQ(integral->getVal(), 1.);

  //Scale the range such that different sigmas are used:
  for (double sigVal : {1., 0.5, 1.E-3, 1.E3}) {
    sig.setVal(sigVal);

    //Shift the mean value of the Gaussian:
    for (double meanVal : {0., 1., -5., 20., -1.E3}) {
      mean.setVal(meanVal);

      auto runTest = [=, &x, &integral](double min, double max, double target) {
        x.setRange("range", meanVal + min*sigVal, meanVal + max*sigVal);
        EXPECT_NEAR(integral->getVal(), target, target*prec)
          << "where mean=" << meanVal << ", sigma=" << sigVal;
      };

      //Test central quantiles
      runTest(-1., 1., oneSig);
      runTest(-2., 2., twoSig);
      runTest(-3., 3., thrSig);

      //Positive & negative, but one close to zero:
      runTest(0., 1., oneSig/2.);
      runTest(-0., 1., oneSig/2.);
      runTest(-2., 1.E-8, twoSig/2.);
      runTest(-1.E-9, 3., thrSig/2.);

      //Far from zero
      runTest(5., 11., 2.8665157E-7);
      runTest(-11., -5., 2.8665157E-7);
      runTest(20., 30., 2.7536241e-89);
      runTest(-30., -20., 2.7536241e-89);
    }
  }
}

