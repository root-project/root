/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *   IP, Inti Pelupessy, Netherlands eScience Center, i.pelupessy@esciencecenter.nl
 *   VC, Vince Croft, DIANA / NYU, vincent.croft@cern.ch
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFit/TestStatistics/RooRealL.h>
#include <RooFit/TestStatistics/RooUnbinnedL.h>

#include <RooArgSet.h>
#include <RooRandom.h>
#include <RooWorkspace.h>
#include <RooAbsPdf.h>
#include <RooDataSet.h>
#include <RooHelpers.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include <RooProdPdf.h>
#include <RooAddition.h>
#include <RooConstraintSum.h>
#include <RooDataHist.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>

#include <algorithm> // count_if

#include "../gtest_wrapper.h"

class RooRealL : public ::testing::TestWithParam<std::tuple<std::size_t>> {};

TEST_P(RooRealL, getVal)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   // Real-life test: calculate a NLL using event-based parallelization. This
   // should replicate RooRealMPFE results.
   RooRandom::randomGenerator()->SetSeed(std::get<0>(GetParam()));
   RooWorkspace w;
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1,0.01,5.0])");
   auto x = w.var("x");
   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate(*x, 10000)};
   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data)};

   auto nominal_result = nll->getVal();

   RooFit::TestStatistics::RooRealL nll_new("nll_new", "new style NLL",
                                            std::make_unique<RooFit::TestStatistics::RooUnbinnedL>(pdf, data.get()));

   auto mp_result = nll_new.getVal();

   EXPECT_DOUBLE_EQ(nominal_result, mp_result);
}

INSTANTIATE_TEST_SUITE_P(NworkersModeSeed, RooRealL, ::testing::Values(2, 3)); // random seed

class RealLVsMPFE : public ::testing::TestWithParam<std::tuple<std::size_t>> {};

TEST_P(RealLVsMPFE, minimize)
{
   // do a minimization (e.g. like in GradMinimizer_Gaussian1D test)

   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   // parameters
   std::size_t seed = std::get<0>(GetParam());

   RooRandom::randomGenerator()->SetSeed(seed);

   RooWorkspace w = RooWorkspace();

   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1,0.01,5.0])");
   auto x = w.var("x");
   RooAbsPdf *pdf = w.pdf("g");
   RooRealVar *mu = w.var("mu");
   RooRealVar *sigma = w.var("sigma");

   std::unique_ptr<RooDataSet> data{pdf->generate(*x, 10000)};
   mu->setVal(-2.9);

   // If we don't set sigma constant, the fit is not stable as we start with mu
   // so close to the boundary
   sigma->setConstant(true);

   std::unique_ptr<RooAbsReal> nll_mpfe{pdf->createNLL(*data)};
   RooFit::TestStatistics::RooRealL nll_new("nll_new", "new style NLL",
                                            std::make_unique<RooFit::TestStatistics::RooUnbinnedL>(pdf, data.get()));

   // save initial values for the start of all minimizations
   RooArgSet values{*mu, *pdf};

   RooArgSet savedValues;
   values.snapshot(savedValues);

   // --------

   RooMinimizer m0(*nll_mpfe);
   m0.setMinimizerType("Minuit2");

   m0.setStrategy(0);
   m0.setPrintLevel(-1);

   m0.migrad();

   std::unique_ptr<RooFitResult> m0result{m0.lastMinuitFit()};
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();
   double mu0 = mu->getVal();
   double muerr0 = mu->getError();

   values.assign(savedValues);

   RooMinimizer m1(nll_new);
   m1.setMinimizerType("Minuit2");

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.migrad();

   std::unique_ptr<RooFitResult> m1result{m1.lastMinuitFit()};
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();
   double mu1 = mu->getVal();
   double muerr1 = mu->getError();

   EXPECT_EQ(minNll0, minNll1);
   EXPECT_EQ(mu0, mu1);
   EXPECT_EQ(muerr0, muerr1);
   EXPECT_EQ(edm0, edm1);
}

INSTANTIATE_TEST_SUITE_P(NworkersModeSeed, RealLVsMPFE, ::testing::Values(2, 3)); // random seed
