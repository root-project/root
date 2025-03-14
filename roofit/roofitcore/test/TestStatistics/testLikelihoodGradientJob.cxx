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
#include "RooFit/TestStatistics/SharedOffset.h"

#include "RooCategory.h" // complete type in MultiBinnedConstraint test
#include "RooDataHist.h" // complete type in Binned test
#include "RooFitResult.h"
#include "RooGenericPdf.h"
#include "RooHelpers.h"
#include "RooMinimizer.h"
#include "RooRandom.h"
#include "RooWorkspace.h"
#include "RooFit/TestStatistics/LikelihoodWrapper.h"
#include "RooFit/TestStatistics/RooUnbinnedL.h"
#include "RooFit/TestStatistics/RooRealL.h"
#include "RooFit/TestStatistics/buildLikelihood.h"
#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/Config.h"

// for MinuitFcnGrad test:
#include "RooFit/ModelConfig.h"
#include "TFile.h"
#include "RooRealSumPdf.h"
#include "../../src/RooMinimizerFcn.h"
#include "../../src/TestStatistics/MinuitFcnGrad.h"
// end for MinuitFcnGrad test

#include <ROOT/TestSupport.hxx>

#include "TH1D.h"
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

std::vector<double> getParamVals(RooAbsMinimizerFcn &fcn)
{
   std::vector<double> values(fcn.getNDim());

   for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = fcn.floatableParam(i).getVal();
   }

   return values;
}

std::unique_ptr<RooFitResult> runMinimizer(RooAbsReal &nll, bool offsetting)
{
   RooMinimizer m0{nll};

   m0.setVerbose(false);
   m0.setStrategy(0);
   m0.setPrintLevel(-1);
   m0.setOffsetting(offsetting);

   m0.minimize("Minuit2", "migrad");

   return std::unique_ptr<RooFitResult>{m0.save()};
}

} // namespace

namespace RFMP = RooFit::MultiProcess;
namespace RFTS = RooFit::TestStatistics;

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

class LikelihoodGradientJobTest : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, bool>> {
   void SetUp() override
   {
      NWorkers = std::get<0>(GetParam());
      seed = std::get<1>(GetParam());
      offsetting = std::get<2>(GetParam());

      RooRandom::randomGenerator()->SetSeed(seed);
   }

protected:
   std::size_t NWorkers = 0;
   std::size_t seed = 0;
   bool offsetting = false;
};

TEST_P(LikelihoodGradientJobTest, Gaussian1D)
{
   // do a minimization, but now using GradMinimizer and its MP version

   // in the 1D Gaussian tests, we suppress the positive definiteness
   // warnings coming from Minuit2 with offsetting; they are errors both
   // in serial RooFit and in the MultiProcess-enabled back-end
   ROOT::TestSupport::CheckDiagsRAII checkDiag;
   if (offsetting) {
      checkDiag.requiredDiag(kError, "Minuit2", "VariableMetricBuilder Initial matrix not pos.def.");
   }

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

   std::unique_ptr<RooFitResult> m0result{runMinimizer(*nll, offsetting)};
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();
   double mu0 = mu->getVal();
   double muerr0 = mu->getError();

   values->assign(savedValues);

   RFTS::RooRealL likelihood("likelihood", "likelihood", std::make_unique<RFTS::RooUnbinnedL>(pdf, data.get()));

   // Convert to RooRealL to enter into minimizer
   RooMinimizer::Config cfg1;
   cfg1.parallelize = NWorkers;
   RooMinimizer m1(likelihood, cfg1);

   m1.setStrategy(0);
   m1.setPrintLevel(-1);
   m1.setOffsetting(offsetting);

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

   RFTS::RooRealL likelihood("likelihood", "likelihood", std::make_unique<RFTS::RooUnbinnedL>(pdf, data.get()));
   RooMinimizer::Config cfg;
   cfg.parallelize = NWorkers;
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

   unsigned int N = 4;

   RooWorkspace w = RooWorkspace();

   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> values;
   RooAbsPdf *pdf;
   std::unique_ptr<RooDataSet> data;
   std::tie(nll, pdf, data, values) = generate_ND_gaussian_pdf_nll(w, N, 1000, RooFit::EvalBackend::Legacy());

   RooArgSet savedValues;
   values->snapshot(savedValues);

   // --------

   std::unique_ptr<RooFitResult> m0result{runMinimizer(*nll, offsetting)};
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();
   std::vector<double> mean0(N);
   std::vector<double> std0(N);
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

   RFTS::RooRealL likelihood("likelihood", "likelihood", std::make_unique<RFTS::RooUnbinnedL>(pdf, data.get()));
   RooMinimizer::Config cfg1;
   cfg1.parallelize = NWorkers;
   RooMinimizer m1(likelihood, cfg1);

   m1.setStrategy(0);
   m1.setPrintLevel(-1);
   m1.setOffsetting(offsetting);

   m1.minimize("Minuit2", "migrad");

   std::unique_ptr<RooFitResult> m1result{m1.save()};
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();
   std::vector<double> mean1(N);
   std::vector<double> std1(N);
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
                         ::testing::Combine(::testing::Values(1, 2, 3),      // number of workers
                                            ::testing::Values(2, 3),         // random seed
                                            ::testing::Values(false, true)), //
                         [](testing::TestParamInfo<LikelihoodGradientJobTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << std::get<0>(paramInfo.param) << "workers_seed" << std::get<1>(paramInfo.param)
                               << (std::get<2>(paramInfo.param) ? "_WithOffsetting" : "_WithoutOffsetting");
                            return ss.str();
                         });

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

   // Construct simultaneous pdf
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

   // dummy offsets (normally they are shared with other objects):
   SharedOffset shared_offset;
   auto nll_ts = RFTS::LikelihoodWrapper::create(
      RFTS::LikelihoodMode::serial,
      RFTS::NLLFactory{*pdf, *data}.GlobalObservables({*w.var("alpha_bkg_obs_A"), *w.var("alpha_bkg_obs_B")}).build(),
      std::make_unique<RFTS::WrapperCalculationCleanFlags>(), shared_offset);

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
   // This is a simultaneous fit, so its likelihood has multiple components. In that case, splitting over
   // components is always preferable, since it is more precise, due to component offsets matching
   // the (-log) function values better.
   RFMP::Config::LikelihoodJob::defaultNEventTasks = 1;
   RFMP::Config::LikelihoodJob::defaultNComponentTasks = 99999; // just a high number, so every component is a task

   std::unique_ptr<RooWorkspace> wPtr = makeSimBinnedConstrainedWorkspace();
   auto &w = *wPtr;

   RooAbsPdf *pdf = w.pdf("model");
   RooAbsData *data = w.data("data");

   // do a minimization, but now using GradMinimizer and its MP version
   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data, Constrain(*w.var("alpha_bkg_A")),
                                                  GlobalObservables(*w.var("alpha_bkg_obs_B")), Offset("initial"))};

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

   std::unique_ptr<RooAbsReal> likelihoodAbsReal{pdf->createNLL(
      *data, Constrain(*w.var("alpha_bkg_A")), GlobalObservables(*w.var("alpha_bkg_obs_B")), ModularL(true))};

   RooMinimizer::Config cfg1;
   cfg1.parallelize = NWorkers;
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

   EXPECT_FLOAT_EQ(minNll_nominal, minNll_GradientJob);
   EXPECT_FLOAT_EQ(alpha_bkg_A_nominal.val, alpha_bkg_A_GradientJob.val);
   EXPECT_FLOAT_EQ(alpha_bkg_A_nominal.error, alpha_bkg_A_GradientJob.error);
   EXPECT_FLOAT_EQ(alpha_bkg_B_nominal.val, alpha_bkg_B_GradientJob.val);
   EXPECT_FLOAT_EQ(alpha_bkg_B_nominal.error, alpha_bkg_B_GradientJob.error);
   EXPECT_FLOAT_EQ(mu_sig_nominal.val, mu_sig_GradientJob.val);
   EXPECT_FLOAT_EQ(mu_sig_nominal.error, mu_sig_GradientJob.error);

   // reset static variables to automatic
   RFMP::Config::LikelihoodJob::defaultNEventTasks = RFMP::Config::LikelihoodJob::automaticNEventTasks;
   RFMP::Config::LikelihoodJob::defaultNComponentTasks = RFMP::Config::LikelihoodJob::automaticNComponentTasks;
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

   // in the 1D Gaussian tests, we suppress the positive definiteness
   // warnings coming from Minuit2 with offsetting; they are errors both
   // in serial RooFit and in the MultiProcess-enabled back-end
   ROOT::TestSupport::CheckDiagsRAII checkDiag;
   if (offsetting) {
      checkDiag.requiredDiag(kError, "Minuit2", "VariableMetricBuilder Initial matrix not pos.def.");
   }

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

   std::unique_ptr<RooFitResult> m0result{runMinimizer(*nll, offsetting)};
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();
   double mu0 = mu->getVal();
   double muerr0 = mu->getError();

   values->assign(savedValues);

   RFTS::RooRealL likelihood("likelihood", "likelihood", std::make_unique<RFTS::RooUnbinnedL>(pdf, data.get()));
   RooMinimizer::Config cfg;
   cfg.parallelize = NWorkers;
   cfg.enableParallelDescent = true;
   RooMinimizer m1(likelihood, cfg);
   m1.setStrategy(0);
   m1.setPrintLevel(-1);
   m1.setOffsetting(offsetting);

   m1.setVerbose(false);

   m1.minimize("Minuit2", "migrad");

   std::unique_ptr<RooFitResult> m1result{m1.save()};
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();
   double mu1 = mu->getVal();
   double muerr1 = mu->getError();

   if (NWorkers > 1 && offsetting) {
      // Note: we cannot expect exact equal results here in most cases when using
      // event-based splitting (which is currently the default; THIS MAY CHANGE!).
      // See LikelihoodJobTest, UnbinnedGaussian1DSelectedParameterValues for an
      // example of where slight bit-wise differences can pop up in fits like this
      // due to minor bit-wise errors in Kahan summation due to different split
      // ups over the event range. We do expect pretty close results, though,
      // because this fit only has 4 iterations and bit-wise differences should
      // not add up too much.
#define EXPECT_NEAR_REL(a, b, c) EXPECT_NEAR(a, b, std::abs(a *c))
      EXPECT_NEAR_REL(minNll0, minNll1, 1e-7);
      EXPECT_NEAR_REL(mu0, mu1, 1e-7);
      EXPECT_NEAR_REL(muerr0, muerr1, 1e-7);
      EXPECT_NEAR(edm0, edm1, 1e-3);
   } else {
      EXPECT_NEAR_REL(minNll0, minNll1, 1e-10);
      EXPECT_NEAR_REL(mu0, mu1, 1e-10);
      EXPECT_NEAR_REL(muerr0, muerr1, 1e-10);
      EXPECT_NEAR(edm0, edm1, 1e-5);
   }
}
#undef EXPECT_NEAR_REL

class LikelihoodGradientJobErrorTest
   : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, bool, bool>> {
   void SetUp() override
   {
      NWorkers = std::get<0>(GetParam());
      seed = std::get<1>(GetParam());
      parallelLikelihood = std::get<2>(GetParam());
      binned = std::get<3>(GetParam());

      RooRandom::randomGenerator()->SetSeed(seed);

      // we want to split only over components so we can test component-offsets precisely
      // (event-offsets give more variation)
      RFMP::Config::LikelihoodJob::defaultNEventTasks = 1; // just one events task (i.e. don't split over events)
      RFMP::Config::LikelihoodJob::defaultNComponentTasks =
         1000000; // assuming components < 1000000: each component = 1 separate task
   }

   void TearDown() override
   {
      // reset static variables to automatic
      RFMP::Config::LikelihoodJob::defaultNEventTasks = RFMP::Config::LikelihoodJob::automaticNEventTasks;
      RFMP::Config::LikelihoodJob::defaultNComponentTasks = RFMP::Config::LikelihoodJob::automaticNComponentTasks;
   }

protected:
   std::size_t NWorkers = 0;
   std::size_t seed = 0;
   bool parallelLikelihood = false;
   bool binned = false;
};

TEST_P(LikelihoodGradientJobErrorTest, ErrorHandling)
{
   // In this test, we setup a model that we know will give evaluation errors, because Minuit will try parameters
   // outside of the physical range during line search. Using the error handling mechanism in RooMinimizerFcn and
   // MinuitFcnGrad, Minuit should get sent out of this area again.
   // Specifically, this test triggers the classic error handling mechanism (logEvalError).

   RooWorkspace w("w");
   w.factory("ArgusBG::model(m[5.2,5.3],m0[5.28,5.2,5.3],c[-2,-10,0])");

   RooAbsPdf *pdf = w.pdf("model");
   std::unique_ptr<RooAbsData> data;
   if (binned) {
      data = std::unique_ptr<RooDataHist>{pdf->generateBinned(*w.var("m"), 10000)};
   } else {
      data = std::unique_ptr<RooDataSet>{pdf->generate(*w.var("m"), 10000)};
   }
   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data, RooFit::EvalBackend::Legacy())};

   // if m0 were constant (i.e. setConstant(true)), the fit would converge without errors, because m0 outside of the
   // physical area of the Argus distribution is what causes the errors in the line search phase of the fit
   w.var("m0")->setConstant(false);

   RooArgSet values{*w.var("m"), *w.var("m0"), *w.var("c"), "values"};
   RooArgSet savedValues;
   values.snapshot(savedValues);

   std::unique_ptr<RooFitResult> m0result{runMinimizer(*nll, false)};
   double minNll0 = m0result->minNll();
   double edm0 = m0result->edm();
   double m_0 = w.var("m")->getVal();
   double m0_0 = w.var("m0")->getVal();
   double c_0 = w.var("c")->getVal();

   values.assign(savedValues);

   std::unique_ptr<RooAbsReal> likelihoodAbsReal{pdf->createNLL(*data, RooFit::ModularL(true))};

   RooMinimizer::Config cfg;
   cfg.parallelize = NWorkers;
   cfg.enableParallelDescent = parallelLikelihood;
   //   cfg.printEvalErrors = 200;
   RooMinimizer m1(*likelihoodAbsReal, cfg);
   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.setVerbose(false);

   m1.minimize("Minuit2", "migrad");

   std::unique_ptr<RooFitResult> m1result{m1.save()};
   double minNll1 = m1result->minNll();
   double edm1 = m1result->edm();
   double m_1 = w.var("m")->getVal();
   double m0_1 = w.var("m0")->getVal();
   double c_1 = w.var("c")->getVal();

   EXPECT_EQ(minNll0, minNll1);
   EXPECT_EQ(edm0, edm1);
   EXPECT_EQ(m_0, m_1);
   EXPECT_EQ(m0_0, m0_1);
   EXPECT_EQ(c_0, c_1);
}

// TODO: https://github.com/root-project/root/pull/12328 meenemen!

/// Fit a simple linear function, that starts in the negative. Triggers RooNaNPacker error handling.
TEST_P(LikelihoodGradientJobErrorTest, FitSimpleLinear)
{
   RooRealVar x("x", "x", -10, 10);
   RooRealVar a1("a1", "a1", 12., -5., 15.);
   RooGenericPdf pdf("pdf", "a1 + x", RooArgSet(x, a1));
   std::unique_ptr<RooAbsData> data;
   if (binned) {
      data = std::unique_ptr<RooDataHist>{pdf.generateBinned(x, 1000)};
   } else {
      data = std::unique_ptr<RooDataSet>{pdf.generate(x, 1000)};
   }
   std::unique_ptr<RooAbsReal> nll(pdf.createNLL(*data, RooFit::EvalBackend::Legacy()));

   RooArgSet normSet{x};
   ASSERT_FALSE(std::isnan(pdf.getVal(normSet)));
   a1.setVal(-5.);
   ASSERT_TRUE(std::isnan(pdf.getVal(normSet)));

   RooMinimizer minim(*nll);
   minim.setPrintLevel(-1);
   minim.setVerbose(false);
   //   minim.setPrintEvalErrors(200);
   minim.migrad();
   minim.hesse();
   std::unique_ptr<RooFitResult> fitResult{minim.save()};
   auto a1Result = a1.getVal();

   // now with multiprocess
   std::unique_ptr<RooAbsReal> nll_mp(pdf.createNLL(*data, RooFit::ModularL(true)));

   a1.setVal(-5.);
   a1.removeError();
   ASSERT_TRUE(std::isnan(pdf.getVal(normSet)));

   RooMinimizer::Config cfg;
   cfg.parallelize = NWorkers;
   cfg.enableParallelDescent = parallelLikelihood;
   //   cfg.printEvalErrors = 200;

   RooMinimizer minim_mp(*nll_mp, cfg);
   minim_mp.setPrintLevel(-1);
   minim_mp.setStrategy(0);
   minim_mp.setVerbose(false);
   minim_mp.migrad();
   minim_mp.hesse();
   std::unique_ptr<RooFitResult> fitResult_mp{minim_mp.save()};
   auto a1Result_mp = a1.getVal();

   EXPECT_EQ(fitResult_mp->status(), 0);
   EXPECT_EQ(a1Result, a1Result_mp);
   EXPECT_EQ(a1Result - a1Result_mp, 0);
}

// TODO: add error handling tests that trigger the RooNaNPacker error handling paths (see testNaNPacker for example
//       setups). In particular a fit of a simultaneous or constrained likelihood to trigger the RooSumL path which has
//       additional handling of the packed NaNs that isn't tested now.

INSTANTIATE_TEST_SUITE_P(LikelihoodGradientJob, LikelihoodGradientJobErrorTest,
                         ::testing::Combine(::testing::Values(1, 2, 3),      // number of workers
                                            ::testing::Values(2, 3),         // random seed
                                            ::testing::Values(false, true),  // with or without LikelihoodJob
                                            ::testing::Values(false, true)), // binned or not
                         [](testing::TestParamInfo<LikelihoodGradientJobErrorTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << std::get<0>(paramInfo.param) << "workers_seed" << std::get<1>(paramInfo.param)
                               << (std::get<2>(paramInfo.param) ? "AlsoWithLikelihoodJob" : "NoLikelihoodJob")
                               << (std::get<3>(paramInfo.param) ? "_Binned" : "_Unbinned");
                            return ss.str();
                         });

class LikelihoodGradientJobBinnedErrorTest : public ::testing::TestWithParam<std::tuple<bool, std::size_t, bool>> {
   void SetUp() override
   {
      do_error = std::get<0>(GetParam());
      NWorkers = std::get<1>(GetParam());
      parallelLikelihood = std::get<2>(GetParam());

      RooRandom::randomGenerator()->SetSeed(20);

      // we want to split only over components so we can test component-offsets precisely
      // (event-offsets give more variation)
      RFMP::Config::LikelihoodJob::defaultNEventTasks = 1; // just one events task (i.e. don't split over events)
      RFMP::Config::LikelihoodJob::defaultNComponentTasks =
         1000000; // assuming components < 1000000: each component = 1 separate task
   }

   void TearDown() override
   {
      // reset static variables to automatic
      RFMP::Config::LikelihoodJob::defaultNEventTasks = RFMP::Config::LikelihoodJob::automaticNEventTasks;
      RFMP::Config::LikelihoodJob::defaultNComponentTasks = RFMP::Config::LikelihoodJob::automaticNComponentTasks;
   }

protected:
   bool do_error = false;
   std::size_t NWorkers = 0;
   bool parallelLikelihood = false;
};

TEST_P(LikelihoodGradientJobBinnedErrorTest, TriggerMuLEZero)
{
   auto th_data = std::make_unique<TH1D>("h_data", "data", 10, 0, 10);
   auto th_sig = std::make_unique<TH1D>("h_sig", "signal", 10, 0, 10);
   auto th_bkg = std::make_unique<TH1D>("h_bkg", "background", 10, 0, 10);

   for (int i = 0; i < 10; i++) {
      th_data->SetBinContent(i + 1, i + 1);
      th_sig->SetBinContent(i + 1, i);
      th_bkg->SetBinContent(i + 1, 1);
   }

   if (do_error) {
      // Trigger error condition by setting both sig and bkg
      // to zero in bin zero, thus triggering a likelihood
      // error since Poisson(N|0) is undefined
      th_sig->SetBinContent(1, 0);
      th_bkg->SetBinContent(1, 0);
   }

   RooWorkspace w("w");
   auto x = w.factory("x[0,10]");
   w.factory("index[A,B]");

   dynamic_cast<RooRealVar *>(x)->setBins(10);

   // we have to build a simultaneous binned likelihood to trigger the "binnedL" evaluation path

   RooDataHist h_sigA("h_sigA", "h_sigA", *w.var("x"), th_sig.get());
   RooDataHist h_sigB("h_sigB", "h_sigB", *w.var("x"), th_sig.get());
   RooDataHist h_bkg("h_bkg", "h_bkg", *w.var("x"), th_bkg.get());

   w.import(h_sigA);
   w.import(h_sigB);
   w.import(h_bkg);
   w.factory("HistPdf::sigA(x,h_sigA)");
   w.factory("HistPdf::sigB(x,h_sigB)");
   w.factory("HistPdf::bkg(x,h_bkg)");

   w.factory("ASUM::model_A(mu_sig[1,-1,10]*sigA,mu_bkg_A[1,-1,10]*bkg)");
   w.factory("ASUM::model_B(mu_sig*sigB,mu_bkg_B[1,-1,10]*bkg)");

   w.pdf("model_A")->setAttribute("BinnedLikelihood");
   w.pdf("model_B")->setAttribute("BinnedLikelihood");

   // Construct simultaneous pdf
   w.factory("SIMUL::model(index[A,B],A=model_A,B=model_B)");

   // Construct dataset
   std::map<std::string, TH1 *> th_data_2D;
   th_data_2D["A"] = th_data.get();
   th_data_2D["B"] = th_data.get();
   RooDataHist h_data("h_data", "h_data", *w.var("x"), *w.cat("index"), th_data_2D);

   // store initial parameters for reuse in second fit
   std::unique_ptr<RooArgSet> values(w.pdf("model")->getParameters(h_data));
   RooArgSet savedValues;
   values->snapshot(savedValues);

   // legacy RooFit fit
   std::unique_ptr<RooAbsReal> nll(w.pdf("model")->createNLL(h_data, RooFit::EvalBackend::Legacy()));

   double nll0BeforeFit = nll->getVal();

   std::unique_ptr<RooFitResult> m0result{runMinimizer(*nll, false)};
   double minNll0 = m0result->minNll();
   double mu_sig0 = w.var("mu_sig")->getVal();
   double mu_bkg_A0 = w.var("mu_bkg_A")->getVal();
   double mu_bkg_B0 = w.var("mu_bkg_B")->getVal();

   values->assign(savedValues);

   std::unique_ptr<RooAbsReal> likelihoodAbsReal{w.pdf("model")->createNLL(h_data, RooFit::ModularL(true))};

   RooMinimizer::Config cfg;
   cfg.parallelize = NWorkers;
   cfg.enableParallelDescent = parallelLikelihood;
   //   cfg.printEvalErrors = 200;
   RooMinimizer m1(*likelihoodAbsReal, cfg);

   double nll1BeforeFit = likelihoodAbsReal->getVal();

   m1.setStrategy(0);
   m1.setPrintLevel(-1);

   m1.setVerbose(false);

   m1.minimize("Minuit2", "migrad");

   std::unique_ptr<RooFitResult> m1result{m1.save()};
   double minNll1 = m1result->minNll();
   double mu_sig1 = w.var("mu_sig")->getVal();
   double mu_bkg_A1 = w.var("mu_bkg_A")->getVal();
   double mu_bkg_B1 = w.var("mu_bkg_B")->getVal();

   EXPECT_EQ(minNll0, minNll1);
   if (do_error) {
      EXPECT_NE(nll0BeforeFit, minNll0);
   } else {
      // These really should be equal, but on most platforms/builds, for some reason it
      // isn't exactly. The exceptions are Apple ARM builds and some builds on x64 when
      // using -march=native.
      EXPECT_DOUBLE_EQ(nll0BeforeFit, minNll0);
   }
   EXPECT_EQ(nll0BeforeFit, nll1BeforeFit);
   EXPECT_EQ(mu_sig0, mu_sig1);
   EXPECT_EQ(mu_bkg_A0, mu_bkg_A1);
   EXPECT_EQ(mu_bkg_B0, mu_bkg_B1);
}

INSTANTIATE_TEST_SUITE_P(LikelihoodGradientJob, LikelihoodGradientJobBinnedErrorTest,
                         ::testing::Combine(::testing::Values(false, true), // trigger error or don't
                                            ::testing::Values(1, 2, 3),     // number of workers
                                            ::testing::Values(false, true)  // with or without LikelihoodJob
                                            ),
                         [](testing::TestParamInfo<LikelihoodGradientJobBinnedErrorTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << std::get<1>(paramInfo.param) << "workers"
                               << (std::get<2>(paramInfo.param) ? "AlsoWithLikelihoodJob" : "NoLikelihoodJob")
                               << (std::get<0>(paramInfo.param) ? "ErrorTriggered" : "");
                            return ss.str();
                         });

TEST(MinuitFcnGrad, DISABLED_CompareToRooMinimizerFcn)
{
   const char *fname = "/Users/pbos/projects/roofit-ssi/benchmark_roofit/data/workspaces/HZy_split.root";
   const char *dataset_name = "combData";

   TFile *f = TFile::Open(fname);

   RooWorkspace *w = (RooWorkspace *)f->Get("combWS");

   // Fixes for known features, binned likelihood optimization
   for (RooAbsArg *arg : w->components()) {
      if (arg->IsA() == RooRealSumPdf::Class()) {
         arg->setAttribute("BinnedLikelihood");
         std::cout << "Activating binned likelihood attribute for " << arg->GetName() << std::endl;
      }
   }

   RooAbsData *data = w->data(dataset_name);
   auto mc = dynamic_cast<RooFit::ModelConfig *>(w->genobj("ModelConfig"));
   auto global_observables = mc->GetGlobalObservables();
   auto nuisance_parameters = mc->GetNuisanceParameters();
   auto pdf = w->pdf(mc->GetPdf()->GetName());

   std::unique_ptr<RooAbsReal> nll_modularL{pdf->createNLL(*data, RooFit::Constrain(*nuisance_parameters),
                                                           RooFit::GlobalObservables(*global_observables),
                                                           RooFit::ModularL(true))};

   std::unique_ptr<RooAbsReal> nll_vanilla{pdf->createNLL(*data, RooFit::Constrain(*nuisance_parameters),
                                                          RooFit::GlobalObservables(*global_observables),
                                                          RooFit::EvalBackend::Legacy()
                                                          /*, RooFit::Offset(true)*/)};

   double vanilla_val = nll_vanilla->getVal();
   double modular_val = nll_modularL->getVal();

   // sanity check
   EXPECT_EQ(modular_val, vanilla_val);

   // set up minimizers
   RooMinimizer m_vanilla(*nll_vanilla);
   // we want to split only over components so we can test component-offsets
   RFMP::Config::LikelihoodJob::defaultNEventTasks = 1; // just one events task (i.e. don't split over events)
   RFMP::Config::LikelihoodJob::defaultNComponentTasks =
      1000000; // assuming components < 1000000: each component = 1 separate task
   RooMinimizer::Config cfg;
   cfg.parallelize = 1;
   cfg.enableParallelDescent = false;
   cfg.enableParallelGradient = true;
   RooMinimizer m_modularL(*nll_modularL, cfg);

   // now use these minimizers to build the corresponding external RooAbsMinimizerFcns
   auto nll_real = dynamic_cast<RFTS::RooRealL *>(nll_modularL.get());
   RFTS::MinuitFcnGrad modularL_fcn(nll_real->getRooAbsL(), &m_modularL, m_modularL.fitter()->Config().ParamsSettings(),
                                    cfg.enableParallelDescent ? RFTS::LikelihoodMode::multiprocess
                                                              : RFTS::LikelihoodMode::serial,
                                    RFTS::LikelihoodGradientMode::multiprocess);
   RooMinimizerFcn vanilla_fcn(nll_vanilla.get(), &m_vanilla);

   EXPECT_EQ(vanilla_fcn(getParamVals(vanilla_fcn).data()), modularL_fcn(getParamVals(modularL_fcn).data()));
   // let's also check with absolutely certain same parameter values, both of them
   EXPECT_EQ(vanilla_fcn(getParamVals(vanilla_fcn).data()), modularL_fcn(getParamVals(vanilla_fcn).data()));
   EXPECT_EQ(vanilla_fcn(getParamVals(modularL_fcn).data()), modularL_fcn(getParamVals(modularL_fcn).data()));

   // reset static variables to automatic
   RFMP::Config::LikelihoodJob::defaultNEventTasks = RFMP::Config::LikelihoodJob::automaticNEventTasks;
   RFMP::Config::LikelihoodJob::defaultNComponentTasks = RFMP::Config::LikelihoodJob::automaticNComponentTasks;
}
