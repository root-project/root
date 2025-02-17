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
#include <RooArgSet.h>
#include <RooDataSet.h>
#include <RooHelpers.h>
#include <RooRealVar.h>
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
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1, 0.01, 3])");
   RooRealVar *x = w.var("x");
   RooRealVar *sigma = w.var("sigma");
   sigma->setConstant(true);
   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate(*x, 10000)};
   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data, RooFit::ModularL(true))};

   auto *nll_real = dynamic_cast<RooFit::TestStatistics::RooRealL *>(&*nll);

   EXPECT_TRUE(nll_real != nullptr);

   auto *nll_absL = dynamic_cast<RooFit::TestStatistics::RooAbsL *>(nll_real->getRooAbsL().get());

   EXPECT_TRUE(nll_absL != nullptr);
}

// New Style likelihoods cannot be initialized with offsetting
TEST(Interface, createNLLModularLAndOffset)
{
   using namespace RooFit;

   RooRandom::randomGenerator()->SetSeed(42);
   RooWorkspace w;
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1, 0.01, 3])");
   RooRealVar *x = w.var("x");
   RooRealVar *sigma = w.var("sigma");
   sigma->setConstant(true);
   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate(*x, 10000)};

   RooHelpers::HijackMessageStream hijack(RooFit::ERROR, RooFit::InputArguments);

   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data, RooFit::Offset("initial"), RooFit::ModularL(true))};

   EXPECT_NE(hijack.str().find("ERROR"), std::string::npos) << "Stream contents: " << hijack.str();

   EXPECT_TRUE(nll == nullptr);
}

// Verifies that the fitTo parallelize interface creates a valid minimization
#ifdef ROOFIT_MULTIPROCESS
TEST(Interface, fitTo)
#else
TEST(Interface, DISABLED_fitTo)
#endif
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");

   RooRandom::randomGenerator()->SetSeed(42);
   RooWorkspace w;
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1, 0.01, 3])");
   auto x = w.var("x");
   RooRealVar *sigma = w.var("sigma");
   sigma->setConstant(true);
   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate({*x}, 10000)};

   RooArgSet values;
   pdf->getParameters(data->get(), values);

   values.add(*pdf);

   RooArgSet savedValues;
   values.snapshot(savedValues);

   std::unique_ptr<RooFitResult> result1{pdf->fitTo(*data, Save(), PrintLevel(-1))};

   double minNll_nominal = result1->minNll();
   double edm_nominal = result1->edm();

   values.assign(savedValues);

   std::unique_ptr<RooFitResult> result2{pdf->fitTo(*data, Save(), PrintLevel(-1), Parallelize(4),
                                                    Experimental::ParallelGradientOptions(true),
                                                    Experimental::ParallelDescentOptions(true))};

   double minNll_GradientJob = result2->minNll();
   double edm_GradientJob = result2->edm();

   EXPECT_NEAR(minNll_nominal, minNll_GradientJob, 1e-4);
   EXPECT_NEAR(edm_nominal, edm_GradientJob, 1e-4);
}
