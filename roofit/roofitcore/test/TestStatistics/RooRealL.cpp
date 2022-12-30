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
#include <RooNLLVar.h>
#include <RooRealVar.h>

#include <algorithm> // count_if

#include <gtest/gtest.h>

// Backward compatibility for gtest version < 1.10.0
#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif

class RooRealL : public ::testing::TestWithParam<std::tuple<std::size_t>> {};

TEST_P(RooRealL, getVal)
{
   // Real-life test: calculate a NLL using event-based parallelization. This
   // should replicate RooRealMPFE results.
   RooRandom::randomGenerator()->SetSeed(std::get<0>(GetParam()));
   RooWorkspace w;
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1,0.01,5.0])");
   auto x = w.var("x");
   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate(RooArgSet(*x), 10000)};
   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data)};

   auto nominal_result = nll->getVal();

   RooFit::TestStatistics::RooRealL nll_new("nll_new", "new style NLL",
                                            std::make_shared<RooFit::TestStatistics::RooUnbinnedL>(pdf, data.get()));

   auto mp_result = nll_new.getVal();

   EXPECT_DOUBLE_EQ(nominal_result, mp_result);
}

void check_NLL_type(RooAbsReal *nll, bool verbose = false)
{
   if (dynamic_cast<RooAddition *>(nll) != nullptr) {
      if (verbose) {
         std::cout << "the NLL object is a RooAddition*..." << std::endl;
      }
      bool has_rooconstraintsum = false;
      for (const auto nll_component : static_cast<RooAddition *>(nll)->list()) {
         if (nll_component->IsA() == RooConstraintSum::Class()) {
            has_rooconstraintsum = true;
            if (verbose) {
               std::cout << "...containing a RooConstraintSum component: " << nll_component->GetName() << std::endl;
            }
            break;
         } else if (nll_component->IsA() != RooNLLVar::Class() && nll_component->IsA() != RooAddition::Class()) {
            std::cerr << "... containing an unexpected component class: " << nll_component->ClassName() << std::endl;
            throw std::runtime_error("RooAddition* type NLL object contains unexpected component class!");
         }
      }
      if (!has_rooconstraintsum) {
         if (verbose) {
            std::cout << "...containing only RooNLLVar components." << std::endl;
         }
      }
   } else if (dynamic_cast<RooNLLVar *>(nll) != nullptr) {
      if (verbose) {
         std::cout << "the NLL object is a RooNLLVar*" << std::endl;
      }
   }
}

void count_NLL_components(RooAbsReal *nll, bool verbose = false)
{
   if (dynamic_cast<RooAddition *>(nll) != nullptr) {
      if (verbose) {
         std::cout << "the NLL object is a RooAddition*..." << std::endl;
      }
      std::size_t nll_component_count = 0;
      for (const auto &component : *nll->getComponents()) {
         if (component->IsA() == RooNLLVar::Class()) {
            ++nll_component_count;
         }
      }
      if (verbose) {
         std::cout << "...containing " << nll_component_count << " RooNLLVar components." << std::endl;
      }
   } else if (dynamic_cast<RooNLLVar *>(nll) != nullptr) {
      if (verbose) {
         std::cout << "the NLL object is a RooNLLVar*" << std::endl;
      }
   }
}

TEST_P(RooRealL, getValRooAddition)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   RooRandom::randomGenerator()->SetSeed(std::get<0>(GetParam()));

   RooWorkspace w;
   w.factory("Gaussian::g(x[-10,10],mu[0,-3,3],sigma[1,0.01,5.0])");

   RooRealVar *x = w.var("x");
   x->setRange("x_range", -3, 0);
   x->setRange("another_range", 1, 7);

   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate(*x, 10000)};

   using namespace RooFit;
   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data, Range("x_range,another_range"))};

   check_NLL_type(nll.get());
   count_NLL_components(nll.get());
}

#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
TEST_P(RooRealL, getValRooConstraintSumAddition)
{
   // modified from
   // https://github.com/roofit-dev/rootbench/blob/43d12f33e8dac7af7d587b53a2804ddf6717e92f/root/roofit/roofit/RooFitASUM.cxx#L417

   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   RooWorkspace ws;
   ws.factory("Polynomial::p0(x[0, 10000])");
   ws.factory("Polynomial::p1(x, {a0[0], a1[1., 0., 2.], a2[0]}, 0)");

   RooRealVar &x = *ws.var("x");
   RooRealVar &a1 = *ws.var("a1");

   RooAbsPdf &p0 = *ws.pdf("p0");
   RooAbsPdf &p1 = *ws.pdf("p1");

   x.setBins(x.getMax());

   std::unique_ptr<RooDataHist> dh_bkg{p0.generateBinned(x, 1000000000)};
   std::unique_ptr<RooDataHist> dh_sig{p1.generateBinned(x, 100000000)};
   dh_bkg->SetName("dh_bkg");
   dh_sig->SetName("dh_sig");

   a1.setVal(2);
   std::unique_ptr<RooDataHist> dh_sig_up{p1.generateBinned(x, 1100000000)};
   dh_sig_up->SetName("dh_sig_up");
   a1.setVal(.5);
   std::unique_ptr<RooDataHist> dh_sig_down{p1.generateBinned(x, 900000000)};
   dh_sig_down->SetName("dh_sig_down");

   RooWorkspace w = RooWorkspace("w");
   w.import(x);
   w.import(*dh_sig);
   w.import(*dh_bkg);
   w.import(*dh_sig_up);
   w.import(*dh_sig_down);
   w.factory("HistFunc::hf_sig(x,dh_sig)");
   w.factory("HistFunc::hf_bkg(x,dh_bkg)");
   w.factory("HistFunc::hf_sig_up(x,dh_sig_up)");
   w.factory("HistFunc::hf_sig_down(x,dh_sig_down)");
   w.factory("PiecewiseInterpolation::pi_sig(hf_sig,hf_sig_down,hf_sig_up,alpha[-5,5])");

   w.factory("ASUM::model(mu[1,0,5]*pi_sig,nu[1]*hf_bkg)");
   w.factory("Gaussian::constraint(alpha,0,1)");
   w.factory("PROD::model2(model,constraint)");

   RooAbsPdf *pdf = w.pdf("model2");

   std::unique_ptr<RooDataHist> data{pdf->generateBinned(x, 1100000)};
   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data)};

   check_NLL_type(nll.get());
   count_NLL_components(nll.get());
}
#endif // !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)

TEST_P(RooRealL, setVal)
{
   // calculate the NLL twice with different parameters
   const bool verbose = false;

   RooRandom::randomGenerator()->SetSeed(std::get<0>(GetParam()));
   RooWorkspace w;
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1,0.01,5.0])");
   auto x = w.var("x");
   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate(RooArgSet(*x), 10000)};
   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data)};

   RooFit::TestStatistics::RooRealL nll_new("nll_new", "new style NLL",
                                            std::make_shared<RooFit::TestStatistics::RooUnbinnedL>(pdf, data.get()));

   // calculate first results
   auto nominal_result1 = nll->getVal();
   auto mp_result1 = nll_new.getVal();

   if (verbose) {
      std::cout << "nominal_result1 = " << nominal_result1 << ", mp_result1 = " << mp_result1 << std::endl;
   }

   EXPECT_EQ(nominal_result1, mp_result1);

   w.var("mu")->setVal(2);

   // calculate second results after parameter change
   auto nominal_result2 = nll->getVal();
   auto mp_result2 = nll_new.getVal();

   if (verbose) {
      std::cout << "nominal_result2 = " << nominal_result2 << ", mp_result2 = " << mp_result2 << std::endl;
   }

   EXPECT_EQ(nominal_result2, mp_result2);
   if (HasFailure()) {
      std::cout << "failed test had seed = " << std::get<0>(GetParam()) << std::endl;
   }
}

INSTANTIATE_TEST_SUITE_P(NworkersModeSeed, RooRealL, ::testing::Values(2, 3)); // random seed

class RealLVsMPFE : public ::testing::TestWithParam<std::tuple<std::size_t>> {};

TEST_P(RealLVsMPFE, getVal)
{
   // Compare our MP NLL to actual RooRealMPFE results using the same strategies.

   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   // parameters
   std::size_t seed = std::get<0>(GetParam());

   RooRandom::randomGenerator()->SetSeed(seed);

   RooWorkspace w;
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1,0.01,5.0])");
   auto x = w.var("x");
   RooAbsPdf *pdf = w.pdf("g");
   std::unique_ptr<RooDataSet> data{pdf->generate(RooArgSet(*x), 10000)};

   std::unique_ptr<RooAbsReal> nll_mpfe{pdf->createNLL(*data)};

   auto mpfe_result = nll_mpfe->getVal();

   RooFit::TestStatistics::RooRealL nll_new("nll_new", "new style NLL",
                                            std::make_shared<RooFit::TestStatistics::RooUnbinnedL>(pdf, data.get()));

   auto mp_result = nll_new.getVal();

   EXPECT_EQ(mpfe_result, mp_result);
   if (HasFailure()) {
      std::cout << "failed test had seed = " << std::get<0>(GetParam()) << std::endl;
   }
}

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

   std::unique_ptr<RooDataSet> data{pdf->generate(RooArgSet(*x), 10000)};
   mu->setVal(-2.9);

   // If we don't set sigma constant, the fit is not stable as we start with mu
   // so close to the boundary
   sigma->setConstant(true);

   std::unique_ptr<RooAbsReal> nll_mpfe{pdf->createNLL(*data)};
   RooFit::TestStatistics::RooRealL nll_new("nll_new", "new style NLL",
                                            std::make_shared<RooFit::TestStatistics::RooUnbinnedL>(pdf, data.get()));

   // save initial values for the start of all minimizations
   RooArgSet values = RooArgSet(*mu, *pdf);

   auto savedValues = dynamic_cast<RooArgSet *>(values.snapshot());
   if (savedValues == nullptr) {
      throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
   }

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

   values = *savedValues;

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

   delete savedValues;
}

INSTANTIATE_TEST_SUITE_P(NworkersModeSeed, RealLVsMPFE, ::testing::Values(2, 3)); // random seed
