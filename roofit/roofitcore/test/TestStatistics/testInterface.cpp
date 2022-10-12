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
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include <RooProdPdf.h>
#include <RooAddition.h>
#include <RooConstraintSum.h>
#include <RooPolynomial.h>
#include <RooDataHist.h>
#include <RooRealSumPdf.h>
#include <RooNLLVar.h>
#include <RooRealVar.h>

#include <algorithm> // count_if

#include "gtest/gtest.h"

class Interface : public ::testing::TestWithParam<std::tuple<std::size_t>> {};

// Verifies that RooAbsPdf::createNLL() can create a valid RooAbsL wrapped in RooRealL
TEST(Interface, createNLLRooAbsL)
{
   using namespace RooFit;

   // Real-life test: calculate a NLL using event-based parallelization. This
   // should replicate RooRealMPFE results.
   RooRandom::randomGenerator()->SetSeed(42);
   RooWorkspace w;
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
   auto x = w.var("x");
   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate(RooArgSet(*x), 10000)};
   RooAbsReal *nll = pdf->createNLL(*data, RooFit::NewStyle(true));

   RooFit::TestStatistics::RooRealL *nll_real = dynamic_cast<RooFit::TestStatistics::RooRealL *>(nll);

   // Check if dynamic cast succesful
   EXPECT_TRUE(nll_real != nullptr);

   std::shared_ptr<RooFit::TestStatistics::RooAbsL> nll_absL = nll_real->getRooAbsL();
}

// Verifies that RooAbsPdf::fitTo() can create a valid RooAbsL wrapped in RooRealL and fit
TEST(Interface, fitTo)
{
   using namespace RooFit;
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");

   // Real-life test: calculate a NLL using event-based parallelization. This
   // should replicate RooRealMPFE results.
   RooRandom::randomGenerator()->SetSeed(42);
   RooWorkspace w;
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
   auto x = w.var("x");
   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate(RooArgSet(*x), 10000)};

   pdf->fitTo(*data, Parallelize(4, true, true));
}

// Verifies that RooAbsPdf::createNLL() can create a valid RooAbsL wrapped in RooRealL
TEST(Interface, RooMinimizer)
{
   using namespace RooFit;
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");

   // Real-life test: calculate a NLL using event-based parallelization. This
   // should replicate RooRealMPFE results.
   RooRandom::randomGenerator()->SetSeed(42);
   RooWorkspace w;
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
   auto x = w.var("x");
   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate(RooArgSet(*x), 10000)};

   // RooAbsReal* nll_1 = pdf->createNLL(*data, RooFit::NewStyle(true));
   // RooMinimizer m_1(*nll_1);
   // m_1.minimize("Minuit2");

   RooMinimizer::Config cfg;
   cfg.parallel_gradient = true;
   cfg.parallel_likelihood = true;
   cfg.nWorkers = 4;
   RooAbsReal *nll_2 = pdf->createNLL(*data, RooFit::NewStyle(true));
   RooMinimizer m_2(*nll_2, cfg);
   m_2.minimize("Minuit2");
}