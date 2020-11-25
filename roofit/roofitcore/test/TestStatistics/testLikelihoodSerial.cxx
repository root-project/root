/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2016-2020, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <stdexcept> // runtime_error

#include <RooRandom.h>
#include <RooWorkspace.h>
#include <RooTimer.h>

#include <RooMinimizer.h>
#include <RooGradMinimizerFcn.h>
#include <RooFitResult.h>

#include <TestStatistics/LikelihoodGradientJob.h>
#include <TestStatistics/LikelihoodSerial.h>
#include <TestStatistics/RooUnbinnedL.h>
#include <RooFit/MultiProcess/JobManager.h>
#include <RooFit/MultiProcess/ProcessManager.h> // need to complete type for debugging

#include "gtest/gtest.h"
#include "../test_lib.h" // generate_1D_gaussian_pdf_nll

TEST(LikelihoodSerial, UnbinnedGaussian1D)
{
   RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

   // parameters
   std::size_t seed = 23;

   RooRandom::randomGenerator()->SetSeed(seed);

   RooWorkspace w = RooWorkspace();

   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> values;
   RooAbsPdf *pdf;
   RooDataSet *data;
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 10000);
   // when c++17 support arrives, change to this:
   //  auto [nll, pdf, data, values] = generate_1D_gaussian_pdf_nll(w, 10000);

   // --------

   auto nll0 = nll->getVal();

   std::shared_ptr<RooFit::TestStatistics::RooAbsL> likelihood =
      std::make_shared<RooFit::TestStatistics::RooUnbinnedL>(pdf, data, false, 0, 0);
   auto clean_flags = std::make_shared<RooFit::TestStatistics::WrapperCalculationCleanFlags>();
   RooFit::TestStatistics::LikelihoodSerial nll_ts(likelihood, clean_flags, nullptr);

   nll_ts.evaluate();
   auto nll1 = nll_ts.return_result();

   EXPECT_EQ(nll0, nll1);
}

TEST(LikelihoodSerial, UnbinnedGaussianND)
{
   RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

   // parameters
   std::size_t seed = 23;

   RooRandom::randomGenerator()->SetSeed(seed);

   unsigned int N = 1;

   RooWorkspace w = RooWorkspace();

   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> values;
   RooAbsPdf *pdf;
   RooDataSet *data;
   std::tie(nll, pdf, data, values) = generate_ND_gaussian_pdf_nll(w, N, 1000);
   // when c++17 support arrives, change to this:
   //  auto [nll, all_values] = generate_ND_gaussian_pdf_nll(w, N, 1000);

   // --------

   auto nll0 = nll->getVal();

   std::shared_ptr<RooFit::TestStatistics::RooAbsL> likelihood =
      std::make_shared<RooFit::TestStatistics::RooUnbinnedL>(pdf, data, false, 0, 0);
   auto clean_flags = std::make_shared<RooFit::TestStatistics::WrapperCalculationCleanFlags>();
   RooFit::TestStatistics::LikelihoodSerial nll_ts(likelihood, clean_flags, nullptr);

   nll_ts.evaluate();
   auto nll1 = nll_ts.return_result();

   EXPECT_EQ(nll0, nll1);
}
