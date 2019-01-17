// Tests for the RooPoisson
// Authors: Stephan Hageboeck, CERN  01/2019

#include "RooRealVar.h"
#include "RooPoisson.h"
#include "TMath.h"
#include "Math/ProbFunc.h"

#include "gtest/gtest.h"


TEST(RooPoisson, AnalyticalIntegral)
{
  RooRealVar x("x", "x", 1, -1., 1.E20);
  RooRealVar lambda("lambda", "lambda", 1);
  RooPoisson pois("pois", "pois", x, lambda, true);
  std::unique_ptr<RooAbsReal> integral(pois.createIntegral(x,
      RooFit::NormSet(x), RooFit::Range("range")));



  //Test the bare CDF with some known values from publicly available tables:
  double precision = 5.E-5;
  auto checkVal = [=, &x, &lambda, &integral](double lambdaVal, double max, double target) {
    x.setRange("range", 0., max);
    lambda.setVal(lambdaVal);

    EXPECT_NEAR(integral->getVal(), target, precision)
      << "where test was CDF_Pois[" << 0. << "," << max << " | " << lambdaVal << "]";
  };

  checkVal(0.1,  0., 0.9048);
  checkVal(0.1,  1., 0.9953);
  checkVal(0.1,  3., 1.0000);
  checkVal(0.1, 10., 1.0000);
  checkVal(0.1, 14., 1.0000);

  checkVal(3.0,  0., 0.0498);
  checkVal(3.0,  1., 0.1991);
  checkVal(3.0,  3., 0.6472);
  checkVal(3.0, 10., 0.9997);
  checkVal(3.0, 14., 1.0000);

  checkVal( 5.,  0., 0.0067);
  checkVal( 5.,  1., 0.0404);
  checkVal( 5.,  3., 0.2650);
  checkVal( 5., 10., 0.9863);
  checkVal( 5., 14., 0.9998);

  checkVal(15.0,  0., 0.0000);
  checkVal(15.0,  5., 0.0028);
  checkVal(15.0, 10., 0.1185);
  checkVal(15.0, 15., 0.5681);
  checkVal(15.0, 30., 0.9998);


  //Now check various lambdas and ranges against ROOT's poisson_cdf:
  precision = 1.E-9;
  for (double lambdaVal : {0.1, 0.5, 1.0, 2., 3., 8., 10., 20., 0.}) {
    lambda.setVal(lambdaVal);

    auto runTest = [=, &x, &lambdaVal, &integral](double min, double max, double target) {
      x.setRange("range", min, max);
      EXPECT_NEAR(integral->getVal(), target, target*precision)
        << "where test was CDF_Pois[" << min << "," << max << " | " << lambdaVal << "]";
    };

    //Ask for the probability to obtain zero
    runTest(-1., 0., TMath::Poisson(0, lambdaVal));
    runTest(-1., 0.1, TMath::Poisson(0, lambdaVal));

    //And first two bins
    runTest(-1., 1., TMath::Poisson(0, lambdaVal) + TMath::Poisson(1, lambdaVal));
    
    //Not defined (should yield zero)
    runTest(-1., -0.5, 0.);
    
    //Some positive ranges
    runTest(0., 10., ROOT::Math::poisson_cdf(10, lambdaVal));
    runTest(0.4, 10.3, ROOT::Math::poisson_cdf(10, lambdaVal));

    //Not starting at zero
    runTest(1.3, 10., ROOT::Math::poisson_cdf(10, lambdaVal) - TMath::Poisson(0, lambdaVal));
    runTest(5.0, 10.3, ROOT::Math::poisson_cdf(10, lambdaVal) - ROOT::Math::poisson_cdf(4, lambdaVal));

    //Larger than max int:
    runTest(0., 1.E20, 1.);
  }



}
