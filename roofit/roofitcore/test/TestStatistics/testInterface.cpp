/*
 * Project: RooFit
 * Authors:
 *   ZW, Zef Wolffs, Nikhef, zefwolffs@gmail.com
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFit/TestStatistics/RooRealL.h>
#include <RooFit/TestStatistics/RooAbsL.h>

#include <RooRandom.h>
#include <RooWorkspace.h>
#include <RooAbsPdf.h>
#include <RooAbsReal.h>
#include <RooDataSet.h>
#include <RooRealVar.h>
#include <RooArgSet.h>
#include <RooGlobalFunc.h>
#include <RooFitResult.h>

#include "Math/Minimizer.h"

#include "gtest/gtest.h"

class Interface : public ::testing::Test {};

// Verifies that RooAbsPdf::createNLL() can create a valid RooAbsL wrapped in RooRealL
TEST(Interface, createNLLRooAbsL)
{
   using namespace RooFit;

   RooRandom::randomGenerator()->SetSeed(42);
   RooWorkspace w;
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
   auto x = w.var("x");
   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate(RooArgSet(*x), 10000)};
   RooAbsReal *nll = pdf->createNLL(*data, RooFit::NewStyle(true));

   RooFit::TestStatistics::RooRealL *nll_real = dynamic_cast<RooFit::TestStatistics::RooRealL *>(nll);

   EXPECT_TRUE(nll_real != nullptr);

   RooFit::TestStatistics::RooAbsL *nll_absL =
      dynamic_cast<RooFit::TestStatistics::RooAbsL *>(nll_real->getRooAbsL().get());

   EXPECT_TRUE(nll_absL != nullptr);
}

// Verifies that the fitTo parallelize interface creates a valid minimization
TEST(Interface, DISABLED_fitTo)
{
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");

   RooRandom::randomGenerator()->SetSeed(42);
   RooWorkspace w;
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
   auto x = w.var("x");
   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate(RooArgSet(*x), 10000)};

   RooArgSet *values = pdf->getParameters(data.get());

   values->add(*pdf);

   RooArgSet *savedValues = dynamic_cast<RooArgSet *>(values->snapshot());
   if (savedValues == nullptr) {
      throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
   }

   std::unique_ptr<RooFitResult> result1{pdf->fitTo(*data, RooFit::Save())};

   double minNll_nominal = result1->minNll();
   double edm_nominal = result1->edm();

   *values = *savedValues;

   std::unique_ptr<RooFitResult> result2{pdf->fitTo(*data, RooFit::Save(), RooFit::Parallelize(4, true, true))};

   double minNll_GradientJob = result2->minNll();
   double edm_GradientJob = result2->edm();

   EXPECT_NEAR(minNll_nominal, minNll_GradientJob, 1e-4);
   EXPECT_NEAR(edm_nominal, edm_GradientJob, 1e-4);
}
