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
#include "RooHelpers.h"
#include "RooMinimizer.h"
#include "RooFitResult.h"
#include "RooFit/TestStatistics/LikelihoodWrapper.h"
#include "RooFit/TestStatistics/RooUnbinnedL.h"
#include "RooFit/TestStatistics/RooRealL.h"
#include "RooFit/TestStatistics/buildLikelihood.h"
#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/Config.h"

#include "Math/Minimizer.h"

#include <stdexcept> // runtime_error

#include "../gtest_wrapper.h"

#include "../test_lib.h" // generate_1D_gaussian_pdf_nll

namespace {

struct ValAndError {
   ValAndError(double inVal, double inError) : val{inVal}, error{inError} {}
   const double val;
   const double error;
};

ValAndError getValAndError(RooArgSet const &parsFinal, const char *name)
{
   auto &var = static_cast<RooRealVar const &>(parsFinal[name]);
   return {var.getVal(), var.getError()};
};

} // namespace

using RooFit::TestStatistics::LikelihoodWrapper;

class Environment : public testing::Environment {
public:
   void SetUp() override
   {
      _changeMsgLvl = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::ERROR);
      ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");
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

class LikelihoodGradientJobTest : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t>> {};

TEST_P(LikelihoodGradientJobTest, Gaussian1D)
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
   std::unique_ptr<RooDataSet> data;
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 10000);
   RooRealVar *mu = w.var("mu");

   RooArgSet savedValues;
   values->snapshot(savedValues);

   // --------

   RooMinimizer m0{*nll};

   m0.setStrategy(0);
   m0.setPrintLevel(-1);

   m0.minimize("Minuit2", "migrad");

   std::unique_ptr<RooFitResult> m0result{m0.save()};
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();
   double mu0 = mu->getVal();
   double muerr0 = mu->getError();

   values->assign(savedValues);

   RooFit::MultiProcess::Config::setDefaultNWorkers(NWorkers);
   RooFit::TestStatistics::RooRealL likelihood("likelihood", "likelihood",
                                               std::make_unique<RooFit::TestStatistics::RooUnbinnedL>(pdf, data.get()));

   // Convert to RooRealL to enter into minimizer
   RooMinimizer::Config cfg1;
   cfg1.parallelize = -1;
   RooMinimizer m1(likelihood, cfg1);

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.minimize("Minuit2", "migrad");

   std::unique_ptr<RooFitResult> m1result{m1.save()};
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
   constexpr bool verbose = false;

   RooRandom::randomGenerator()->SetSeed(seed);

   RooWorkspace w = RooWorkspace();

   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> values;
   RooAbsPdf *pdf;
   std::unique_ptr<RooDataSet> data;
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 10000);

   RooArgSet savedValues;
   values->snapshot(savedValues);

   // --------

   RooFit::MultiProcess::Config::setDefaultNWorkers(NWorkers);
   RooFit::TestStatistics::RooRealL likelihood("likelihood", "likelihood",
                                               std::make_unique<RooFit::TestStatistics::RooUnbinnedL>(pdf, data.get()));
   RooMinimizer::Config cfg;
   cfg.parallelize = -1;
   RooMinimizer m1(likelihood, cfg);

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   if (verbose)
      std::cout << "... running migrad first time ..." << std::endl;
   m1.minimize("Minuit2", "migrad");

   values->assign(savedValues);

   if (verbose)
      std::cout << "... running migrad second time ..." << std::endl;
   m1.minimize("Minuit2", "migrad");
}

TEST_P(LikelihoodGradientJobTest, GaussianND)
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
   std::unique_ptr<RooDataSet> data;
   std::tie(nll, pdf, data, values) = generate_ND_gaussian_pdf_nll(w, N, 1000);

   RooArgSet savedValues;
   values->snapshot(savedValues);

   // --------

   RooMinimizer m0(*nll);

   m0.setStrategy(0);
   m0.setPrintLevel(-1);

   m0.minimize("Minuit2", "migrad");

   std::unique_ptr<RooFitResult> m0result{m0.save()};
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

   values->assign(savedValues);

   // --------

   RooFit::MultiProcess::Config::setDefaultNWorkers(NWorkers);
   RooFit::TestStatistics::RooRealL likelihood("likelihood", "likelihood",
                                               std::make_unique<RooFit::TestStatistics::RooUnbinnedL>(pdf, data.get()));
   RooMinimizer::Config cfg1;
   cfg1.parallelize = -1;
   RooMinimizer m1(likelihood, cfg1);

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.minimize("Minuit2", "migrad");

   std::unique_ptr<RooFitResult> m1result{m1.save()};
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();
   double mean1[N];
   double std1[N];
   for (unsigned ix = 0; ix < N; ++ix) {
      {
         std::ostringstream os;
         os << "m" << ix;
         mean1[ix] = static_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
      {
         std::ostringstream os;
         os << "s" << ix;
         std1[ix] = static_cast<RooRealVar *>(w.arg(os.str().c_str()))->getVal();
      }
   }

   EXPECT_EQ(minNll0, minNll1);
   EXPECT_EQ(edm0, edm1);

   for (unsigned ix = 0; ix < N; ++ix) {
      EXPECT_EQ(mean0[ix], mean1[ix]);
      EXPECT_EQ(std0[ix], std1[ix]);
   }
}

INSTANTIATE_TEST_SUITE_P(NworkersSeed, LikelihoodGradientJobTest,
                         ::testing::Combine(::testing::Values(1, 2, 3), // number of workers
                                            ::testing::Values(2, 3)));  // random seed

std::unique_ptr<RooWorkspace> makeSimBinnedConstrainedWorkspace()
{
   std::size_t seed = 23;
   RooRandom::randomGenerator()->SetSeed(seed);

   auto wPtr = std::make_unique<RooWorkspace>();
   auto &w = *wPtr;

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

   // Construct simulatenous pdf
   w.factory("SIMUL::model(index[A,B],A=model_A,B=model_B)");

   // Construct dataset from physics pdf
   std::unique_ptr<RooAbsData> data{w.pdf("model")->generate({*w.var("x"), *w.cat("index")}, RooFit::AllBinned())};

   w.import(*data, RooFit::Rename("data"));

   return wPtr;
}

TEST(SimBinnedConstrainedTestBasic, BasicParameters)
{
   std::unique_ptr<RooWorkspace> wPtr = makeSimBinnedConstrainedWorkspace();
   auto &w = *wPtr;

   RooAbsPdf *pdf = w.pdf("model");
   RooAbsData *data = w.data("data");

   // original test:
   std::unique_ptr<RooAbsReal> nll{
      pdf->createNLL(*data, RooFit::GlobalObservables(*w.var("alpha_bkg_obs_A"), *w.var("alpha_bkg_obs_B")))};

   // --------

   const double nll0 = nll->getVal();

   auto nll_ts = LikelihoodWrapper::create(RooFit::TestStatistics::LikelihoodMode::serial,
                                           RooFit::TestStatistics::NLLFactory{*pdf, *data}
                                              .GlobalObservables({*w.var("alpha_bkg_obs_A"), *w.var("alpha_bkg_obs_B")})
                                              .build(),
                                           std::make_unique<RooFit::TestStatistics::WrapperCalculationCleanFlags>());

   nll_ts->evaluate();
   auto nll1 = nll_ts->getResult();

   EXPECT_DOUBLE_EQ(nll0, nll1.Sum());
}

class SimBinnedConstrainedTest : public ::testing::TestWithParam<std::tuple<bool>> {};

TEST_P(SimBinnedConstrainedTest, ConstrainedAndOffset)
{
   using namespace RooFit;

   // parameters
   const bool parallelLikelihood = std::get<0>(GetParam());

   std::unique_ptr<RooWorkspace> wPtr = makeSimBinnedConstrainedWorkspace();
   auto &w = *wPtr;

   RooAbsPdf *pdf = w.pdf("model");
   RooAbsData *data = w.data("data");

   // do a minimization, but now using GradMinimizer and its MP version
   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data, Constrain(*w.var("alpha_bkg_obs_A")),
                                                  GlobalObservables(*w.var("alpha_bkg_obs_B")), Offset(true))};

   // parameters
   std::size_t NWorkers = 2;

   std::unique_ptr<RooArgSet> values(pdf->getParameters(data));

   RooArgSet savedValues;
   values->snapshot(savedValues);

   // --------

   RooMinimizer m0(*nll);

   m0.setStrategy(0);
   m0.setPrintLevel(-1);

   m0.minimize("Minuit2", "migrad");

   std::unique_ptr<RooFitResult> m0result{m0.save()};
   RooArgSet parsFinal_nominal{m0result->floatParsFinal()};

   const double minNll_nominal = m0result->minNll();
   ValAndError alpha_bkg_A_nominal = getValAndError(parsFinal_nominal, "alpha_bkg_A");
   ValAndError alpha_bkg_B_nominal = getValAndError(parsFinal_nominal, "alpha_bkg_B");
   ValAndError mu_sig_nominal = getValAndError(parsFinal_nominal, "mu_sig");

   values->assign(savedValues);

   RooFit::MultiProcess::Config::setDefaultNWorkers(NWorkers);

   std::unique_ptr<RooAbsReal> likelihoodAbsReal{pdf->createNLL(
      *data, Constrain(*w.var("alpha_bkg_obs_A")), GlobalObservables(*w.var("alpha_bkg_obs_B")), ModularL(true))};

   RooMinimizer::Config cfg1;
   cfg1.parallelize = -1;
   cfg1.enableParallelDescent = parallelLikelihood;
   RooMinimizer m1(*likelihoodAbsReal, cfg1);

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.setOffsetting(true);
   m1.optimizeConst(2);

   m1.minimize("Minuit2", "migrad");

   std::unique_ptr<RooFitResult> m1result{m1.save()};
   RooArgSet parsFinal_GradientJob{m1result->floatParsFinal()};

   const double minNll_GradientJob = m1result->minNll();
   ValAndError alpha_bkg_A_GradientJob = getValAndError(parsFinal_GradientJob, "alpha_bkg_A");
   ValAndError alpha_bkg_B_GradientJob = getValAndError(parsFinal_GradientJob, "alpha_bkg_B");
   ValAndError mu_sig_GradientJob = getValAndError(parsFinal_GradientJob, "mu_sig");

   // Because offsetting is handled differently in the TestStatistics classes
   // compared to the way it was done in the object returned from
   // RooAbsPdf::createNLL (a RooAddition of an offset RooNLLVar and a
   // non-offset RooConstraintSum, whereas RooSumL applies the offset to the
   // total sum of its binned, unbinned and constraint components),
   // we cannot always expect exactly equal results for fits with likelihood
   // offsetting enabled. See also the LikelihoodSerialSimBinnedConstrainedTest.
   // ConstrainedAndOffset test case in testLikelihoodSerial and other tests
   // below with offsetting. We do test whether they are near, because any
   // changes that cause further divergence should be carefully considered.
   // Note that we also omit the edm checks for these test cases. They will
   // always (by default) be below 1e-3 for successful fits (by definition),
   // so it tests nothing specific about the likelihood classes, it's just a
   // postcondition of the Minuit fit.
#define EXPECT_NEAR_REL(a, b, c) EXPECT_NEAR(a, b, std::abs(a *c))
#define EXPECT_NEAR_REL_INCL_ERROR(a, b, c)       \
   EXPECT_NEAR(a.val, b.val, std::abs(a.val *c)); \
   EXPECT_NEAR(a.error, b.error, std::abs(a.error *c))
   EXPECT_NEAR_REL(minNll_nominal, minNll_GradientJob, 1e-6);
   EXPECT_NEAR_REL_INCL_ERROR(alpha_bkg_A_nominal, alpha_bkg_A_GradientJob, 1e-6);
   EXPECT_NEAR_REL_INCL_ERROR(alpha_bkg_B_nominal, alpha_bkg_B_GradientJob, 1e-6);
   EXPECT_NEAR_REL_INCL_ERROR(mu_sig_nominal, mu_sig_GradientJob, 1e-6);
#undef EXPECT_NEAR_REL
#undef EXPECT_NEAR_REL_INCL_ERROR
}

INSTANTIATE_TEST_SUITE_P(LikelihoodGradientJob, SimBinnedConstrainedTest, testing::Values(false, true),
                         [](testing::TestParamInfo<SimBinnedConstrainedTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << (std::get<0>(paramInfo.param) ? "AlsoWithLikelihoodJob" : "NoLikelihoodJob");
                            return ss.str();
                         });

TEST_P(LikelihoodGradientJobTest, Gaussian1DAlsoWithLikelihoodJob)
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
   std::unique_ptr<RooDataSet> data;
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 10000);
   RooRealVar *mu = w.var("mu");

   RooArgSet savedValues;
   values->snapshot(savedValues);

   // --------

   RooMinimizer m0{*nll};

   m0.setStrategy(0);
   m0.setPrintLevel(-1);
   m0.setVerbose(false);

   m0.minimize("Minuit2", "migrad");

   std::unique_ptr<RooFitResult> m0result{m0.save()};
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();
   double mu0 = mu->getVal();
   double muerr0 = mu->getError();

   values->assign(savedValues);

   RooFit::MultiProcess::Config::setDefaultNWorkers(NWorkers);
   RooFit::TestStatistics::RooRealL likelihood("likelihood", "likelihood",
                                               std::make_unique<RooFit::TestStatistics::RooUnbinnedL>(pdf, data.get()));
   RooMinimizer::Config cfg;
   cfg.parallelize = -1;
   cfg.enableParallelDescent = true;
   RooMinimizer m1(likelihood, cfg);
   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.setVerbose(false);

   m1.minimize("Minuit2", "migrad");

   std::unique_ptr<RooFitResult> m1result{m1.save()};
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();
   double mu1 = mu->getVal();
   double muerr1 = mu->getError();

   EXPECT_FLOAT_EQ(minNll0, minNll1);
   EXPECT_FLOAT_EQ(mu0, mu1);
   EXPECT_FLOAT_EQ(muerr0, muerr1);
   EXPECT_NEAR(edm0, edm1, 1e-5);
}
