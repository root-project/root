/*
 * RooGradMinimizer tests
 * Copyright (c) 2017, Patrick Bos, Netherlands eScience Center
 */

//#include <iostream>

#include <TRandom.h>
#include <RooWorkspace.h>
#include <RooDataSet.h>
#include <RooRealVar.h>
#include <RooAbsPdf.h>
#include <RooTimer.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>

#include <RooGradMinimizer.h>

#include "gtest/gtest.h"

#include "ULPdiff.h"


TEST(GradMinimizer, Gaussian1D) {
  for (int i = 0; i < 10; ++i) {
    // produce the same random stuff every time
    gRandom->SetSeed(1);

    RooWorkspace w = RooWorkspace();

    w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");

    auto x = w.var("x");
    RooAbsPdf *pdf = w.pdf("g");
    RooRealVar *mu = w.var("mu");

    RooDataSet *data = pdf->generate(RooArgSet(*x), 10000);
    mu->setVal(-2.9);
    // mu->setError(0.1);

    auto nll = pdf->createNLL(*data);

    // save initial values for the start of all minimizations
    RooArgSet values = RooArgSet(*mu, *pdf, *nll);

    RooArgSet *savedValues = dynamic_cast<RooArgSet *>(values.snapshot());
    if (savedValues == nullptr) {
      throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
    }

    // --------

    RooWallTimer wtimer;

    // --------

//  std::cout << "starting nominal calculation" << std::endl;

    RooMinimizer m0(*nll);
    m0.setMinimizerType("Minuit2");

    m0.setStrategy(0);
    m0.setPrintLevel(-1);

    wtimer.start();
    m0.migrad();
    wtimer.stop();

//  std::cout << "  -- nominal calculation wall clock time:        " << wtimer.timing_s() << "s" << std::endl;

    RooFitResult *m0result = m0.lastMinuitFit();
    double minNll0 = m0result->minNll();
    double edm0 = m0result->edm();
    double mu0 = mu->getVal();
    double muerr0 = mu->getError();

//  std::cout << " ======== resetting initial values ======== " << std::endl;
    values = *savedValues;


//  std::cout << "starting GradMinimizer" << std::endl;

    RooGradMinimizer m1(*nll);
    m1.setMinimizerType("Minuit2");

    m1.setStrategy(0);
    m1.setPrintLevel(-1);

    wtimer.start();
    m1.migrad();
    wtimer.stop();

//  std::cout << "  -- GradMinimizer calculation wall clock time:  " << wtimer.timing_s() << "s" << std::endl;

    RooFitResult *m1result = m1.lastMinuitFit();
    double minNll1 = m1result->minNll();
    double edm1 = m1result->edm();
    double mu1 = mu->getVal();
    double muerr1 = mu->getError();

    // for unknown reasons, the results are often not exactly equal
    // see discussion: https://github.com/roofit-dev/root/issues/11
    // we cast to float to do the comparison, because at double precision
    // the differences are often just slightly too large
    EXPECT_FLOAT_EQ(minNll0, minNll1);
    EXPECT_FLOAT_EQ(mu0, mu1);
    EXPECT_FLOAT_EQ(muerr0, muerr1);
    EXPECT_NEAR(edm0, edm1, 1e-4);
    // these ULP functions can be used for further analysis of the differences
//    std::cout << "ulp_diff muerr: " << ulp_diff(muerr0, muerr1) << std::endl;
//    std::cout << "ulp_diff muerr float-casted: " << ulp_diff((float) muerr0, (float) muerr1) << std::endl;
  }
}
