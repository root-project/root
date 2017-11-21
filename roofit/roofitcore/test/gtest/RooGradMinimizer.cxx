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


TEST(GradMinimizer, Gaussian1D) {
  // produce the same random stuff every time
  gRandom->SetSeed(1);

  RooWorkspace w = RooWorkspace();

  w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");

  auto x = w.var("x");
  RooAbsPdf * pdf = w.pdf("g");
  RooRealVar * mu = w.var("mu");

  RooDataSet * data = pdf->generate(RooArgSet(*x), 10000);
  mu->setVal(-2.9);
  // mu->setError(0.1);

  auto nll = pdf->createNLL(*data);

  // save initial values for the start of all minimizations
  RooArgSet values = RooArgSet(*mu, *pdf, *nll);

  RooArgSet* savedValues = dynamic_cast<RooArgSet*>(values.snapshot());
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

  RooFitResult * m0result = m0.lastMinuitFit();
  double minNll0 = m0result->minNll();
  double edm0 = m0result->edm();
  double mu0 = mu->getVal();

//  std::cout << " ======== resetting initial values ======== " << std::endl;
  values = *savedValues;


//  std::cout << "starting GradMinimizer" << std::endl;

  RooGradMinimizer m1(*nll);

  m1.setStrategy(0);
  m1.setPrintLevel(-1);

  wtimer.start();
  m1.migrad();
  wtimer.stop();

//  std::cout << "  -- GradMinimizer calculation wall clock time:  " << wtimer.timing_s() << "s" << std::endl;

  RooFitResult * m1result = m1.lastMinuitFit();
  double minNll1 = m1result->minNll();
  double edm1 = m1result->edm();
  double mu1 = mu->getVal();

  EXPECT_EQ(minNll0, minNll1);
  EXPECT_EQ(mu0, mu1);

  EXPECT_DOUBLE_EQ(edm0, edm1);
}
