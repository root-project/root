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

#include "RooFit/TestStatistics/LikelihoodGradientWrapper.h"

#include "RooRandom.h"
#include "RooWorkspace.h"
#include "RooDataHist.h" // complete type in Binned test
#include "RooCategory.h" // complete type in MultiBinnedConstraint test
#include "RooMinimizer.h"
#include "RooFitResult.h"
#include "RooFit/TestStatistics/LikelihoodWrapper.h"
#include "RooFit/TestStatistics/RooUnbinnedL.h"
#include "RooFit/TestStatistics/buildLikelihood.h"
#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/Config.h"
#include "RooStats/ModelConfig.h"

#include <TFile.h>

#include <stdexcept> // runtime_error

#include "gtest/gtest.h"
#include "../test_lib.h" // generate_1D_gaussian_pdf_nll

using RooFit::TestStatistics::LikelihoodWrapper;

class Environment : public testing::Environment {
public:
   void SetUp() override { RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR); }
};

// Previously, we just called AddGlobalTestEnvironment in global namespace, but this caused either a warning about an
// unused declared variable (the return value of the call) or a parsing problem that the compiler can't handle if you
// don't store the return value at all. The solution is to just define this manual main function. The default gtest
// main function does InitGoogleTest and RUN_ALL_TESTS, we add the environment call in the middle.
int main(int argc, char **argv)
{
   testing::InitGoogleTest(&argc, argv);
   testing::AddGlobalTestEnvironment(new Environment);
   return RUN_ALL_TESTS();
}

class LikelihoodGradientJob : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t>> {
};

TEST_P(LikelihoodGradientJob, Gaussian1D)
{
   // do a minimization, but now using GradMinimizer and its MP version

   // parameters
   std::size_t NWorkers = std::get<0>(GetParam());
   std::size_t seed = std::get<1>(GetParam());

   RooRandom::randomGenerator()->SetSeed(seed);

   RooWorkspace w = RooWorkspace();

   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> values;
   RooAbsPdf *pdf;
   RooDataSet *data;
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 10000);
   RooRealVar *mu = w.var("mu");

   RooArgSet *savedValues = dynamic_cast<RooArgSet *>(values->snapshot());
   if (savedValues == nullptr) {
      throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
   }

   // --------

   std::unique_ptr<RooMinimizer> m0 = std::make_unique<RooMinimizer>(*nll, RooMinimizer::FcnMode::gradient);
   m0->setMinimizerType("Minuit2");

   m0->setStrategy(0);
   m0->setPrintLevel(-1);

   m0->migrad();

   RooFitResult *m0result = m0->lastMinuitFit();
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();
   double mu0 = mu->getVal();
   double muerr0 = mu->getError();

   *values = *savedValues;

   RooFit::MultiProcess::Config::setDefaultNWorkers(NWorkers);
   auto likelihood = std::make_shared<RooFit::TestStatistics::RooUnbinnedL>(pdf, data);
   RooMinimizer m1(likelihood, RooFit::TestStatistics::LikelihoodMode::serial,
                   RooFit::TestStatistics::LikelihoodGradientMode::multiprocess);
   m1.setMinimizerType("Minuit2");

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.migrad();

   RooFitResult *m1result = m1.lastMinuitFit();
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();
   double mu1 = mu->getVal();
   double muerr1 = mu->getError();

   EXPECT_EQ(minNll0, minNll1);
   EXPECT_EQ(mu0, mu1);
   EXPECT_EQ(muerr0, muerr1);
   EXPECT_EQ(edm0, edm1);
}

TEST(LikelihoodGradientJob, RepeatMigrad)
{
   // do multiple minimizations using MP::GradMinimizer, testing breakdown and rebuild

   // parameters
   std::size_t NWorkers = 2;
   std::size_t seed = 5;

   RooRandom::randomGenerator()->SetSeed(seed);

   RooWorkspace w = RooWorkspace();

   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> values;
   RooAbsPdf *pdf;
   RooDataSet *data;
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 10000);

   RooArgSet *savedValues = dynamic_cast<RooArgSet *>(values->snapshot());
   if (savedValues == nullptr) {
      throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
   }

   // --------

   RooFit::MultiProcess::Config::setDefaultNWorkers(NWorkers);
   auto likelihood = std::make_shared<RooFit::TestStatistics::RooUnbinnedL>(pdf, data);
   RooMinimizer m1(likelihood, RooFit::TestStatistics::LikelihoodMode::serial,
                   RooFit::TestStatistics::LikelihoodGradientMode::multiprocess);

   m1.setMinimizerType("Minuit2");

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   std::cout << "... running migrad first time ..." << std::endl;
   m1.migrad();

   *values = *savedValues;

   std::cout << "... running migrad second time ..." << std::endl;
   m1.migrad();
}

TEST_P(LikelihoodGradientJob, GaussianND)
{
   // do a minimization, but now using GradMinimizer and its MP version

   // parameters
   std::size_t NWorkers = std::get<0>(GetParam());
   std::size_t seed = std::get<1>(GetParam());

   unsigned int N = 4;

   RooRandom::randomGenerator()->SetSeed(seed);

   RooWorkspace w = RooWorkspace();

   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> values;
   RooAbsPdf *pdf;
   RooDataSet *data;
   std::tie(nll, pdf, data, values) = generate_ND_gaussian_pdf_nll(w, N, 1000);

   RooArgSet *savedValues = dynamic_cast<RooArgSet *>(values->snapshot());
   if (savedValues == nullptr) {
      throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
   }

   // --------

   RooMinimizer m0(*nll, RooMinimizer::FcnMode::gradient);
   m0.setMinimizerType("Minuit2");

   m0.setStrategy(0);
   m0.setPrintLevel(-1);

   m0.migrad();

   RooFitResult *m0result = m0.lastMinuitFit();
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();
   double mean0[N];
   double std0[N];
   for (unsigned ix = 0; ix < N; ++ix) {
      {
         std::ostringstream os;
         os << "m" << ix;
         mean0[ix] = dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
      {
         std::ostringstream os;
         os << "s" << ix;
         std0[ix] = dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
   }

   // --------

   *values = *savedValues;

   // --------

   RooFit::MultiProcess::Config::setDefaultNWorkers(NWorkers);
   auto likelihood = std::make_shared<RooFit::TestStatistics::RooUnbinnedL>(pdf, data);
   RooMinimizer m1(likelihood, RooFit::TestStatistics::LikelihoodMode::serial,
                   RooFit::TestStatistics::LikelihoodGradientMode::multiprocess);
   m1.setMinimizerType("Minuit2");

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.migrad();

   RooFitResult *m1result = m1.lastMinuitFit();
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();
   double mean1[N];
   double std1[N];
   for (unsigned ix = 0; ix < N; ++ix) {
      {
         std::ostringstream os;
         os << "m" << ix;
         mean1[ix] = dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
      {
         std::ostringstream os;
         os << "s" << ix;
         std1[ix] = dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
   }

   EXPECT_EQ(minNll0, minNll1);
   EXPECT_EQ(edm0, edm1);

   for (unsigned ix = 0; ix < N; ++ix) {
      EXPECT_EQ(mean0[ix], mean1[ix]);
      EXPECT_EQ(std0[ix], std1[ix]);
   }
}

INSTANTIATE_TEST_SUITE_P(NworkersSeed, LikelihoodGradientJob,
                         ::testing::Combine(::testing::Values(1, 2, 3), // number of workers
                                            ::testing::Values(2, 3)));  // random seed

class BasicTest : public ::testing::Test {
protected:
   void SetUp() override
   {
      RooRandom::randomGenerator()->SetSeed(seed);
      clean_flags = std::make_shared<RooFit::TestStatistics::WrapperCalculationCleanFlags>();
   }

   std::size_t seed = 23;
   RooWorkspace w;
   std::unique_ptr<RooAbsReal> nll;
   RooAbsPdf *pdf;
   RooAbsData *data;
   std::shared_ptr<RooFit::TestStatistics::RooAbsL> likelihood;
   std::shared_ptr<RooFit::TestStatistics::WrapperCalculationCleanFlags> clean_flags;
};

class LikelihoodSimBinnedConstrainedTest : public BasicTest {
protected:
   void SetUp() override
   {
      BasicTest::SetUp();
      // Unbinned pdfs that define template histograms

      w.factory("Gaussian::gA(x[-10,10],-2,3)");
      w.factory("Gaussian::gB(x[-10,10],2,1)");
      w.factory("Uniform::u(x)");

      // Generate template histograms

      RooDataHist *h_sigA = w.pdf("gA")->generateBinned(*w.var("x"), 1000);
      RooDataHist *h_sigB = w.pdf("gB")->generateBinned(*w.var("x"), 1000);
      RooDataHist *h_bkg = w.pdf("u")->generateBinned(*w.var("x"), 1000);

      w.import(*h_sigA, RooFit::Rename("h_sigA"));
      w.import(*h_sigB, RooFit::Rename("h_sigB"));
      w.import(*h_bkg, RooFit::Rename("h_bkg"));

      // Construct binned pdf as sum of amplitudes
      w.factory("HistFunc::hf_sigA(x,h_sigA)");
      w.factory("HistFunc::hf_sigB(x,h_sigB)");
      w.factory("HistFunc::hf_bkg(x,h_bkg)");

      w.factory(
         "ASUM::model_phys_A(mu_sig[1,-1,10]*hf_sigA,expr::mu_bkg_A('1+0.02*alpha_bkg_A',alpha_bkg_A[-5,5])*hf_bkg)");
      w.factory("ASUM::model_phys_B(mu_sig*hf_sigB,expr::mu_bkg_B('1+0.05*alpha_bkg_B',alpha_bkg_B[-5,5])*hf_bkg)");

      // Construct L_subs: Gaussian subsidiary measurement that constrains alpha_bkg
      w.factory("Gaussian:model_subs_A(alpha_bkg_obs_A[0],alpha_bkg_A,1)");
      w.factory("Gaussian:model_subs_B(alpha_bkg_obs_B[0],alpha_bkg_B,1)");

      // Construct full pdfs for each component (A,B)
      w.factory("PROD::model_A(model_phys_A,model_subs_A)");
      w.factory("PROD::model_B(model_phys_B,model_subs_B)");

      // Construct simulatenous pdf
      w.factory("SIMUL::model(index[A,B],A=model_A,B=model_B)");

      pdf = w.pdf("model");
      // Construct dataset from physics pdf
      data = pdf->generate(RooArgSet(*w.var("x"), *w.cat("index")), RooFit::AllBinned());
   }
};

TEST_F(LikelihoodSimBinnedConstrainedTest, BasicParameters)
{
   // original test:
   nll.reset(pdf->createNLL(
      *data, RooFit::GlobalObservables(RooArgSet(*w.var("alpha_bkg_obs_A"), *w.var("alpha_bkg_obs_B")))));

   // --------

   auto nll0 = nll->getVal();

   likelihood = RooFit::TestStatistics::buildLikelihood(
      pdf, data, RooFit::TestStatistics::GlobalObservables({*w.var("alpha_bkg_obs_A"), *w.var("alpha_bkg_obs_B")}));
   auto nll_ts = LikelihoodWrapper::create(RooFit::TestStatistics::LikelihoodMode::serial, likelihood, clean_flags);

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_DOUBLE_EQ(nll0, nll1);
}

TEST_F(LikelihoodSimBinnedConstrainedTest, ConstrainedAndOffset)
{
   // do a minimization, but now using GradMinimizer and its MP version
   nll.reset(pdf->createNLL(*data, RooFit::Constrain(RooArgSet(*w.var("alpha_bkg_obs_A"))),
                            RooFit::GlobalObservables(RooArgSet(*w.var("alpha_bkg_obs_B"))), RooFit::Offset(true)));

   // parameters
   std::size_t NWorkers = 2;

   RooArgSet *values = pdf->getParameters(data);

   values->add(*pdf);
   values->add(*nll);

   RooArgSet *savedValues = dynamic_cast<RooArgSet *>(values->snapshot());
   if (savedValues == nullptr) {
      throw std::runtime_error("params->snapshot() cannot be casted to RooArgSet!");
   }

   // --------

   RooMinimizer m0(*nll, RooMinimizer::FcnMode::gradient);

   m0.setMinimizerType("Minuit2");
   m0.setStrategy(0);
   m0.setPrintLevel(1);

   m0.migrad();

   RooFitResult *m0result = m0.lastMinuitFit();
   double minNll_nominal = m0result->minNll();
   double edm_nominal = m0result->edm();
   double alpha_bkg_A_nominal = w.var("alpha_bkg_A")->getVal();
   double alpha_bkg_A_error_nominal = w.var("alpha_bkg_A")->getError();
   double alpha_bkg_B_nominal = w.var("alpha_bkg_B")->getVal();
   double alpha_bkg_B_error_nominal = w.var("alpha_bkg_B")->getError();
   double mu_sig_nominal = w.var("mu_sig")->getVal();
   double mu_sig_error_nominal = w.var("mu_sig")->getError();

   *values = *savedValues;

   RooFit::MultiProcess::Config::setDefaultNWorkers(NWorkers);

   likelihood = RooFit::TestStatistics::buildLikelihood(
      pdf, data, RooFit::TestStatistics::ConstrainedParameters(RooArgSet(*w.var("alpha_bkg_obs_A"))),
      RooFit::TestStatistics::GlobalObservables(RooArgSet(*w.var("alpha_bkg_obs_B"))));

   RooMinimizer m1(likelihood, RooFit::TestStatistics::LikelihoodMode::serial,
                   RooFit::TestStatistics::LikelihoodGradientMode::multiprocess);
   m1.setOffsetting(true);

   m1.setMinimizerType("Minuit2");
   m1.setStrategy(0);
   m1.setPrintLevel(1);
   m1.optimizeConst(2);

   m1.migrad();

   RooFitResult *m1result = m1.lastMinuitFit();
   double minNll_GradientJob = m1result->minNll();
   double edm_GradientJob = m1result->edm();
   double alpha_bkg_A_GradientJob = w.var("alpha_bkg_A")->getVal();
   double alpha_bkg_A_error_GradientJob = w.var("alpha_bkg_A")->getError();
   double alpha_bkg_B_GradientJob = w.var("alpha_bkg_B")->getVal();
   double alpha_bkg_B_error_GradientJob = w.var("alpha_bkg_B")->getError();
   double mu_sig_GradientJob = w.var("mu_sig")->getVal();
   double mu_sig_error_GradientJob = w.var("mu_sig")->getError();

   // Because offsetting is handled differently in the TestStatistics classes
   // compared to the way it was done in the object returned from
   // RooAbsPdf::createNLL (a RooAddition of an offset RooNLLVar and a
   // non-offset RooConstraintSum, whereas RooSumL applies the offset to the
   // total sum of its binned, unbinned and constraint components),
   // we cannot always expect exactly equal results for fits with likelihood
   // offsetting enabled. See also the LikelihoodSerialSimBinnedConstrainedTest.
   // ConstrainedAndOffset test case in testLikelihoodSerial.
   EXPECT_FLOAT_EQ(minNll_nominal, minNll_GradientJob);
   EXPECT_NEAR(edm_nominal, edm_GradientJob, 1e-5);
   EXPECT_FLOAT_EQ(alpha_bkg_A_nominal, alpha_bkg_A_GradientJob);
   EXPECT_FLOAT_EQ(alpha_bkg_A_error_nominal, alpha_bkg_A_error_GradientJob);
   EXPECT_FLOAT_EQ(alpha_bkg_B_nominal, alpha_bkg_B_GradientJob);
   EXPECT_FLOAT_EQ(alpha_bkg_B_error_nominal, alpha_bkg_B_error_GradientJob);
   EXPECT_FLOAT_EQ(mu_sig_nominal, mu_sig_GradientJob);
   EXPECT_FLOAT_EQ(mu_sig_error_nominal, mu_sig_error_GradientJob);
}
