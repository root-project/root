/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooFit/TestStatistics/LikelihoodWrapper.h"
#include "RooFit/TestStatistics/SharedOffset.h"

#include <RooRandom.h>
#include <RooWorkspace.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include <RooDataHist.h> // complete type in Binned test
#include <RooCategory.h> // complete type in MultiBinnedConstraint test
#include <RooFit/TestStatistics/RooUnbinnedL.h>
#include <RooFit/TestStatistics/RooBinnedL.h>
#include <RooFit/TestStatistics/buildLikelihood.h>
#include <RooFit/TestStatistics/RooRealL.h>
#include <RooFit/MultiProcess/Config.h>
#include <RooHelpers.h>

#include "Math/Util.h" // KahanSum

#include <stdexcept> // runtime_error

#include "../gtest_wrapper.h"

#include "../test_lib.h" // generate_1D_gaussian_pdf_nll

namespace RFMP = RooFit::MultiProcess;
namespace RFTS = RooFit::TestStatistics;

class Environment : public testing::Environment {
public:
   void SetUp() override
   {
      _changeMsgLvl = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::ERROR);
      RFMP::Config::setDefaultNWorkers(2);
   }
   void TearDown() override { _changeMsgLvl.reset(); }

private:
   std::unique_ptr<RooHelpers::LocalChangeMsgLevel> _changeMsgLvl;
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

class LikelihoodJobTest : public ::testing::Test {
protected:
   void SetUp() override
   {
      RooRandom::randomGenerator()->SetSeed(seed);
      clean_flags = std::make_unique<RFTS::WrapperCalculationCleanFlags>();
   }

   std::size_t seed = 23;
   RooWorkspace w;
   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> values;
   RooAbsPdf *pdf;
   std::unique_ptr<RooAbsData> data;
   std::shared_ptr<RFTS::RooAbsL> likelihood;
   std::shared_ptr<RFTS::WrapperCalculationCleanFlags> clean_flags;
};

class LikelihoodJobBinnedDatasetTest : public LikelihoodJobTest {
protected:
   void SetUp() override
   {
      LikelihoodJobTest::SetUp();

      // Unbinned pdfs that define template histograms
      w.factory("Gaussian::g(x[-10,10],0,2)");
      w.factory("Uniform::u(x)");

      // Generate template histograms
      std::unique_ptr<RooDataHist> h_sig{w.pdf("g")->generateBinned(*w.var("x"), 1000)};
      std::unique_ptr<RooDataHist> h_bkg{w.pdf("u")->generateBinned(*w.var("x"), 1000)};

      w.import(*h_sig, RooFit::Rename("h_sig"));
      w.import(*h_bkg, RooFit::Rename("h_bkg"));

      // Construct binned pdf as sum of amplitudes
      w.factory("HistFunc::hf_sig(x,h_sig)");
      w.factory("HistFunc::hf_bkg(x,h_bkg)");
      w.factory("ASUM::model(mu_sig[1,-1,10]*hf_sig,mu_bkg[1,-1,10]*hf_bkg)");

      pdf = w.pdf("model");
   }
};

TEST_F(LikelihoodJobTest, UnbinnedGaussian1D)
{
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 10000);
   likelihood = RFTS::buildLikelihood(pdf, data.get());
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);

   auto nll0 = nll->getVal();

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_EQ(nll0, nll1.Sum());
}

TEST_F(LikelihoodJobTest, UnbinnedGaussian1DSelectedParameterValues)
{
   // The parameter mu values in this test were selected because they were shown to generate deviant values
   // for the LikelihoodJob in other tests.
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 9860);
   // Bisecting the number of events to find the number at which the deviation starts occurring in the
   // likelihood result showed that event 9860 was the main culprit (which has index 9859)

   // We also need to split over events precisely with 2 workers to trigger the deviation:
   RFMP::Config::LikelihoodJob::defaultNEventTasks = RFMP::Config::getDefaultNWorkers();
   RFMP::Config::LikelihoodJob::defaultNComponentTasks = 1;

   RooRealVar *mu = w.var("mu");
   mu->setVal(-2.8991551193432676392);

   likelihood = RFTS::buildLikelihood(pdf, data.get());
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);

   auto nll0 = nll->getVal();

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_NE(nll0, nll1.Sum());
   // they differ only a bit:
   EXPECT_DOUBLE_EQ(nll0, nll1.Sum());

   // reset static variables to automatic
   RFMP::Config::LikelihoodJob::defaultNEventTasks = RFMP::Config::LikelihoodJob::automaticNEventTasks;
   RFMP::Config::LikelihoodJob::defaultNComponentTasks = RFMP::Config::LikelihoodJob::automaticNComponentTasks;
}

// This test was disabled because it was occasionally timing out on the CI.
// Evaluating the same likelihood twice should not be a problem anymore, and if
// it would be it would also manifest in other tests.
TEST_F(LikelihoodJobTest, DISABLED_UnbinnedGaussian1DTwice)
{
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 10000);
   likelihood = RFTS::buildLikelihood(pdf, data.get());
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);

   auto nll0 = nll->getVal();

   // do it twice; this failed due to a bug
   nll_ts->evaluate();
   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_EQ(nll0, nll1.Sum());
}

#ifdef ROOFIT_LEGACY_EVAL_BACKEND
TEST_F(LikelihoodJobTest, UnbinnedGaussianND)
{
   using namespace RooFit;
   unsigned int N = 4;

   std::tie(nll, pdf, data, values) = generate_ND_gaussian_pdf_nll(w, N, 1000, EvalBackend::Legacy());
   likelihood = TestStatistics::buildLikelihood(pdf, data.get());
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);

   auto nll0 = nll->getVal();

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_EQ(nll0, nll1.Sum());
}
#endif // ROOFIT_LEGACY_EVAL_BACKEND

TEST_F(LikelihoodJobBinnedDatasetTest, UnbinnedPdf)
{
   data = std::unique_ptr<RooDataHist>{pdf->generateBinned(*w.var("x"))};

   nll = std::unique_ptr<RooAbsReal>{pdf->createNLL(*data)};

   likelihood = RFTS::buildLikelihood(pdf, data.get());
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);

   auto nll0 = nll->getVal();

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_DOUBLE_EQ(nll0, nll1.Sum());
}

TEST_F(LikelihoodJobBinnedDatasetTest, BinnedNLL)
{
   pdf->setAttribute("BinnedLikelihood");
   data = std::unique_ptr<RooDataHist>{pdf->generateBinned(*w.var("x"))};

   nll = std::unique_ptr<RooAbsReal>{pdf->createNLL(*data)};

   likelihood = RFTS::buildLikelihood(pdf, data.get());
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);

   auto nll0 = nll->getVal();

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_EQ(nll0, nll1.Sum());
}

TEST_F(LikelihoodJobTest, SimBinned)
{
   // Unbinned pdfs that define template histograms
   w.factory("Gaussian::gA(x[-10,10],-2,3)");
   w.factory("Gaussian::gB(x[-10,10],2,1)");
   w.factory("Uniform::u(x)");

   // Generate template histograms
   std::unique_ptr<RooDataHist> h_sigA{w.pdf("gA")->generateBinned(*w.var("x"), 1000)};
   std::unique_ptr<RooDataHist> h_sigB{w.pdf("gB")->generateBinned(*w.var("x"), 1000)};
   std::unique_ptr<RooDataHist> h_bkg{w.pdf("u")->generateBinned(*w.var("x"), 1000)};

   w.import(*h_sigA, RooFit::Rename("h_sigA"));
   w.import(*h_sigB, RooFit::Rename("h_sigB"));
   w.import(*h_bkg, RooFit::Rename("h_bkg"));

   // Construct L_phys: binned pdf as sum of amplitudes
   w.factory("HistFunc::hf_sigA(x,h_sigA)");
   w.factory("HistFunc::hf_sigB(x,h_sigB)");
   w.factory("HistFunc::hf_bkg(x,h_bkg)");

   w.factory("ASUM::model_A(mu_sig[1,-1,10]*hf_sigA,mu_bkg_A[1,-1,10]*hf_bkg)");
   w.factory("ASUM::model_B(mu_sig*hf_sigB,mu_bkg_B[1,-1,10]*hf_bkg)");

   w.pdf("model_A")->setAttribute("BinnedLikelihood");
   w.pdf("model_B")->setAttribute("BinnedLikelihood");

   // Construct simultaneous pdf
   w.factory("SIMUL::model(index[A,B],A=model_A,B=model_B)");

   // Construct dataset
   pdf = w.pdf("model");
   data = std::unique_ptr<RooDataSet>{pdf->generate({*w.var("x"), *w.cat("index")}, RooFit::AllBinned())};

   nll = std::unique_ptr<RooAbsReal>{pdf->createNLL(*data)};

   likelihood = RFTS::buildLikelihood(pdf, data.get());
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);

   auto nll0 = nll->getVal();

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_DOUBLE_EQ(nll0, nll1.Sum());
}

TEST_F(LikelihoodJobTest, BinnedConstrained)
{
   // Unbinned pdfs that define template histograms

   w.factory("Gaussian::g(x[-10,10],0,2)");
   w.factory("Uniform::u(x)");

   // Generate template histograms

   std::unique_ptr<RooDataHist> h_sig{w.pdf("g")->generateBinned(*w.var("x"), 1000)};
   std::unique_ptr<RooDataHist> h_bkg{w.pdf("u")->generateBinned(*w.var("x"), 1000)};

   w.import(*h_sig, RooFit::Rename("h_sig"));
   w.import(*h_bkg, RooFit::Rename("h_bkg"));

   // Construct binned pdf as sum of amplitudes
   w.factory("HistFunc::hf_sig(x,h_sig)");
   w.factory("HistFunc::hf_bkg(x,h_bkg)");
   w.factory("ASUM::model_phys(mu_sig[1,-1,10]*hf_sig,expr::mu_bkg('1+0.02*alpha_bkg',alpha_bkg[-5,5])*hf_bkg)");

   // Construct L_subs: Gaussian subsidiary measurement that constrains alpha_bkg
   w.factory("Gaussian:model_subs(alpha_bkg_obs[0],alpha_bkg,1)");

   // Construct full pdf
   w.factory("PROD::model(model_phys,model_subs)");

   pdf = w.pdf("model");
   // Construct dataset from physics pdf
   data = std::unique_ptr<RooDataHist>{w.pdf("model_phys")->generateBinned(*w.var("x"))};

   nll = std::unique_ptr<RooAbsReal>{pdf->createNLL(*data, RooFit::GlobalObservables(*w.var("alpha_bkg_obs")))};

   // --------

   auto nll0 = nll->getVal();

   likelihood = RFTS::NLLFactory{*pdf, *data}.GlobalObservables(*w.var("alpha_bkg_obs")).build();
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_EQ(nll0, nll1.Sum());
}

TEST_F(LikelihoodJobTest, SimUnbinned)
{
   // SIMULTANEOUS FIT OF 2 UNBINNED DATASETS
   // This is a simultaneous fit, so its likelihood has multiple components. In that case, splitting over
   // components is always preferable, since it is more precise, due to component offsets matching
   // the (-log) function values better.
   RFMP::Config::LikelihoodJob::defaultNEventTasks = 1;
   RFMP::Config::LikelihoodJob::defaultNComponentTasks = 99999; // just a high number, so every component is a task

   w.factory("ExtendPdf::egA(Gaussian::gA(x[-10,10],mA[2,-10,10],s[3,0.1,10]),nA[1000])");
   w.factory("ExtendPdf::egB(Gaussian::gB(x,mB[-2,-10,10],s),nB[100])");
   w.factory("SIMUL::model(index[A,B],A=egA,B=egB)");

   pdf = w.pdf("model");
   // Construct dataset from physics pdf
   data = std::unique_ptr<RooDataSet>{pdf->generate({*w.var("x"), *w.cat("index")})};

   nll = std::unique_ptr<RooAbsReal>{pdf->createNLL(*data)};

   // --------

   auto nll0 = nll->getVal();

   likelihood = RFTS::buildLikelihood(pdf, data.get());
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_EQ(nll0, nll1.Sum());

   // reset static variables to automatic
   RFMP::Config::LikelihoodJob::defaultNEventTasks = RFMP::Config::LikelihoodJob::automaticNEventTasks;
   RFMP::Config::LikelihoodJob::defaultNComponentTasks = RFMP::Config::LikelihoodJob::automaticNComponentTasks;
}

TEST_F(LikelihoodJobTest, SimUnbinnedNonExtended)
{
   // SIMULTANEOUS FIT OF 2 UNBINNED DATASETS

   w.factory("ExtendPdf::egA(Gaussian::gA(x[-10,10],mA[2,-10,10],s[3,0.1,10]),nA[1000])");
   w.factory("ExtendPdf::egB(Gaussian::gB(x,mB[-2,-10,10],s),nB[100])");
   w.factory("SIMUL::model(index[A,B],A=gA,B=gB)");

   std::unique_ptr<RooDataSet> dA{w.pdf("gA")->generate(*w.var("x"), 1)};
   std::unique_ptr<RooDataSet> dB{w.pdf("gB")->generate(*w.var("x"), 1)};
   w.cat("index")->setLabel("A");
   dA->addColumn(*w.cat("index"));
   w.cat("index")->setLabel("B");
   dB->addColumn(*w.cat("index"));

   data = std::unique_ptr<RooDataSet>{static_cast<RooDataSet *>(dA->Clone())};
   static_cast<RooDataSet &>(*data).append(*dB);

   pdf = w.pdf("model");

   nll = std::unique_ptr<RooAbsReal>{pdf->createNLL(*data)};

   likelihood = RFTS::buildLikelihood(pdf, data.get());
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);

   auto nll0 = nll->getVal();

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_EQ(nll0, nll1.Sum());
}

class LikelihoodJobSimBinnedConstrainedTest : public LikelihoodJobTest {
protected:
   void SetUp() override
   {
      LikelihoodJobTest::SetUp();
      // Unbinned pdfs that define template histograms

      w.factory("Gaussian::gA(x[-10,10],-2,3)");
      w.factory("Gaussian::gB(x[-10,10],2,1)");
      w.factory("Uniform::u(x)");

      // Generate template histograms

      std::unique_ptr<RooDataHist> h_sigA{w.pdf("gA")->generateBinned(*w.var("x"), 1000)};
      std::unique_ptr<RooDataHist> h_sigB{w.pdf("gB")->generateBinned(*w.var("x"), 1000)};
      std::unique_ptr<RooDataHist> h_bkg{w.pdf("u")->generateBinned(*w.var("x"), 1000)};

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

      // Construct simultaneous pdf
      w.factory("SIMUL::model(index[A,B],A=model_A,B=model_B)");

      pdf = w.pdf("model");
      // Construct dataset from physics pdf
      data = std::unique_ptr<RooDataSet>{pdf->generate({*w.var("x"), *w.cat("index")}, RooFit::AllBinned())};
   }
};

TEST_F(LikelihoodJobSimBinnedConstrainedTest, BasicParameters)
{
   // original test:
   nll = std::unique_ptr<RooAbsReal>{
      pdf->createNLL(*data, RooFit::GlobalObservables(*w.var("alpha_bkg_obs_A"), *w.var("alpha_bkg_obs_B")))};

   // --------

   auto nll0 = nll->getVal();

   likelihood =
      RFTS::NLLFactory{*pdf, *data}.GlobalObservables({*w.var("alpha_bkg_obs_A"), *w.var("alpha_bkg_obs_B")}).build();
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_DOUBLE_EQ(nll0, nll1.Sum());
}

#ifdef ROOFIT_LEGACY_EVAL_BACKEND
TEST_F(LikelihoodJobSimBinnedConstrainedTest, ConstrainedAndOffset)
{
   // A variation to test some additional parameters (ConstrainedParameters and offsetting)

   // The reference likelihood is using the legacy evaluation backend, because
   // the multiprocess test statistics classes were designed to give values
   // that are bit-by-bit identical with the old test statistics based on
   // RooAbsTestStatistic.
   nll = std::unique_ptr<RooAbsReal>{pdf->createNLL(*data, RooFit::Constrain(*w.var("alpha_bkg_A")),
                                                    RooFit::GlobalObservables(*w.var("alpha_bkg_obs_B")),
                                                    RooFit::Offset(true), RooFit::EvalBackend::Legacy())};

   // --------

   auto nll0 = nll->getVal();

   likelihood = RFTS::NLLFactory{*pdf, *data}
                   .ConstrainedParameters(*w.var("alpha_bkg_A"))
                   .GlobalObservables(*w.var("alpha_bkg_obs_B"))
                   .build();
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);
   nll_ts->enableOffsetting(true);

   nll_ts->evaluate();
   // The RFTS classes used for minimization (RooAbsL and Wrapper derivatives) will return offset
   // values, whereas RooNLLVar::getVal will always return the non-offset value, since that is the "actual" likelihood
   // value. RooRealL will also give the non-offset value, so that can be directly compared to the RooNLLVar::getVal
   // result (the nll0 vs nll2 comparison below). To compare to the raw RooAbsL/Wrapper value nll1, however, we need to
   // manually add the offset.
   ROOT::Math::KahanSum<double> nll1 = nll_ts->getResult();
   ROOT::Math::KahanSum<double> nll_ts_offset;
   for (auto &offset_comp : offset.offsets()) {
      nll1 += offset_comp;
      nll_ts_offset += offset_comp;
   }

   EXPECT_EQ(nll0, nll1.Sum());
   EXPECT_FALSE(nll_ts_offset.Sum() == 0);

   // also check against RooRealL value
   RFTS::RooRealL nll_real("real_nll", "RooRealL version", likelihood);

   auto nll2 = nll_real.getVal();

   EXPECT_EQ(nll0, nll2);
   EXPECT_EQ(nll1.Sum(), nll2);
}
#endif // ROOFIT_LEGACY_EVAL_BACKEND

TEST_F(LikelihoodJobTest, BatchedUnbinnedGaussianND)
{
   unsigned int N = 4;

   auto backend = RooFit::EvalBackend::Cpu();

   std::tie(nll, pdf, data, values) = generate_ND_gaussian_pdf_nll(w, N, 1000, backend);
   auto nll0 = nll->getVal();

   likelihood = RFTS::NLLFactory{*pdf, *data}.EvalBackend(backend).build();
   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_DOUBLE_EQ(nll0, nll1.Sum());
}

class LikelihoodJobSplitStrategies : public LikelihoodJobSimBinnedConstrainedTest,
                                     public testing::WithParamInterface<std::tuple<std::size_t, std::size_t>> {};

#ifdef ROOFIT_LEGACY_EVAL_BACKEND
TEST_P(LikelihoodJobSplitStrategies, SimBinnedConstrainedAndOffset)
#else
TEST_P(LikelihoodJobSplitStrategies, DISABLED_SimBinnedConstrainedAndOffset)
#endif
{
   // Based on ConstrainedAndOffset, this test tests different parallelization strategies

   // The reference likelihood is using the legacy evaluation backend, because
   // the multiprocess test statistics classes were designed to give values
   // that are bit-by-bit identical with the old test statistics based on
   // RooAbsTestStatistic.
   nll = std::unique_ptr<RooAbsReal>{pdf->createNLL(*data, RooFit::Constrain(*w.var("alpha_bkg_A")),
                                                    RooFit::GlobalObservables(*w.var("alpha_bkg_obs_B")),
                                                    RooFit::Offset(true), RooFit::EvalBackend::Legacy())};

   // --------

   auto nll0 = nll->getVal();

   likelihood = RFTS::NLLFactory{*pdf, *data}
                   .ConstrainedParameters(*w.var("alpha_bkg_A"))
                   .GlobalObservables(*w.var("alpha_bkg_obs_B"))
                   .build();

   RFMP::Config::LikelihoodJob::defaultNEventTasks = std::get<0>(GetParam());
   RFMP::Config::LikelihoodJob::defaultNComponentTasks = std::get<1>(GetParam());

   // dummy offsets (normally they are shared with other objects):
   SharedOffset offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(RFTS::LikelihoodMode::multiprocess, likelihood, clean_flags, offset);
   nll_ts->enableOffsetting(true);

   nll_ts->evaluate();
   // The RFTS classes used for minimization (RooAbsL and Wrapper derivatives) will return offset
   // values, whereas RooNLLVar::getVal will always return the non-offset value, since that is the "actual" likelihood
   // value. RooRealL will also give the non-offset value, so that can be directly compared to the RooNLLVar::getVal
   // result (the nll0 vs nll2 comparison below). To compare to the raw RooAbsL/Wrapper value nll1, however, we need to
   // manually add the offset.
   ROOT::Math::KahanSum<double> nll1 = nll_ts->getResult();
   ROOT::Math::KahanSum<double> nll_ts_offset;
   for (auto &offset_comp : offset.offsets()) {
      nll1 += offset_comp;
      nll_ts_offset += offset_comp;
   }

   // We can expect minor bit-wise differences here in some cases when splitting over events.
   // Note that many of these cases are, in fact, exactly bit-wise equal.
   EXPECT_DOUBLE_EQ(nll0, nll1.Sum());
   EXPECT_FALSE(nll_ts_offset.Sum() == 0);

   // also check against RooRealL value
   RFTS::RooRealL nll_real("real_nll", "RooRealL version", likelihood);

   auto nll2 = nll_real.getVal();

   EXPECT_EQ(nll0, nll2);
   EXPECT_DOUBLE_EQ(nll1.Sum(), nll2);

   // reset static variables to automatic
   RFMP::Config::LikelihoodJob::defaultNEventTasks = RFMP::Config::LikelihoodJob::automaticNEventTasks;
   RFMP::Config::LikelihoodJob::defaultNComponentTasks = RFMP::Config::LikelihoodJob::automaticNComponentTasks;
}

INSTANTIATE_TEST_SUITE_P(SplitStrategies, LikelihoodJobSplitStrategies,
                         testing::Combine(
                            // number of event tasks:
                            testing::Values(0, 1, 2, 50, 100, 101, 102,
                                            100000), // the last value is larger than number of events to test that
                            // number of component tasks:
                            testing::Values(0, 1, 2, 4) // the last value is larger than number of components
                            ));
