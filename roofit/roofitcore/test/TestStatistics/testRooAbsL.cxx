/*
 * Project: RooFit
 * Authors:
 *   ZW, Zef Wolffs, Nikhef, z.wolffs@nikhef.nl
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFit/TestStatistics/LikelihoodWrapper.h>

#include <RooRandom.h>
#include <RooWorkspace.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include "RooDataHist.h" // complete type in Binned test
#include "RooCategory.h" // complete type in MultiBinnedConstraint test
#include <RooFit/TestStatistics/RooUnbinnedL.h>
#include <RooFit/TestStatistics/RooBinnedL.h>
#include <RooFit/TestStatistics/RooSumL.h>
#include <RooFit/TestStatistics/buildLikelihood.h>
#include <RooFit/TestStatistics/RooRealL.h>
#include <RooHelpers.h>

#include "Math/Util.h" // KahanSum

#include <stdexcept> // runtime_error

#include "gtest/gtest.h"
#include "../test_lib.h" // generate_1D_gaussian_pdf_nll

using RooFit::TestStatistics::LikelihoodWrapper;

class Environment : public testing::Environment {
public:
   void SetUp() override { _changeMsgLvl = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::ERROR); }
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

class RooAbsLTest : public ::testing::Test {
protected:
   void SetUp() override
   {
      RooRandom::randomGenerator()->SetSeed(seed);
      clean_flags = std::make_unique<RooFit::TestStatistics::WrapperCalculationCleanFlags>();
   }

   std::size_t seed = 23;
   RooWorkspace w;
   std::unique_ptr<RooAbsReal> nll;
   std::unique_ptr<RooArgSet> values;
   RooAbsPdf *pdf;
   std::unique_ptr<RooAbsData> data;
   std::shared_ptr<RooFit::TestStatistics::RooAbsL> likelihood;
   std::shared_ptr<RooFit::TestStatistics::WrapperCalculationCleanFlags> clean_flags;
};

class BinnedDatasetTest : public RooAbsLTest {
protected:
   void SetUp() override
   {
      RooAbsLTest::SetUp();

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

class SimBinnedConstrainedTest : public RooAbsLTest {
protected:
   void SetUp() override
   {
      RooAbsLTest::SetUp();
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

// Note: these tests could be simpler; the models were just copied
// from testLikelihoodSerial for convenience.

// Introspection tests
TEST_F(RooAbsLTest, UnbinnedLikelihoodIntrospection)
{
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 10000);
   likelihood = RooFit::TestStatistics::buildLikelihood(pdf, data.get());

   EXPECT_STREQ("RooUnbinnedL", (likelihood->GetClassName()).c_str());
   EXPECT_STREQ("RooUnbinnedL::g", (likelihood->GetInfo()).c_str());
}

TEST_F(BinnedDatasetTest, BinnedLikelihoodIntrospection)
{
   pdf->setAttribute("BinnedLikelihood");
   data = std::unique_ptr<RooDataHist>{pdf->generateBinned(*w.var("x"))};
   likelihood = RooFit::TestStatistics::buildLikelihood(pdf, data.get());

   EXPECT_STREQ("RooBinnedL", (likelihood->GetClassName()).c_str());
   EXPECT_STREQ("RooBinnedL::model", (likelihood->GetInfo()).c_str());
}

TEST_F(RooAbsLTest, SumLikelihoodIntrospection)
{
   w.factory("ExtendPdf::egA(Gaussian::gA(x[-10,10],mA[2,-10,10],s[3,0.1,10]),nA[1000])");
   w.factory("ExtendPdf::egB(Gaussian::gB(x,mB[-2,-10,10],s),nB[100])");
   w.factory("SIMUL::model(index[A,B],A=egA,B=egB)");

   pdf = w.pdf("model");
   // Construct dataset from physics pdf
   data = std::unique_ptr<RooDataSet>{pdf->generate({*w.var("x"), *w.cat("index")})};

   likelihood = RooFit::TestStatistics::buildLikelihood(pdf, data.get());

   EXPECT_STREQ("RooSumL", (likelihood->GetClassName()).c_str());
   EXPECT_STREQ("RooSumL::model", (likelihood->GetInfo()).c_str());
}

TEST_F(SimBinnedConstrainedTest, SumSubsidiaryLikelihoodIntrospection)
{

   likelihood = RooFit::TestStatistics::NLLFactory{*pdf, *data}
                   .GlobalObservables({*w.var("alpha_bkg_obs_A"), *w.var("alpha_bkg_obs_B")})
                   .build();

   EXPECT_STREQ("RooSumL", (likelihood->GetClassName()).c_str());
   EXPECT_STREQ("RooSumL::model", (likelihood->GetInfo()).c_str());

   // Is RooSumL so we can cast to this type to use its further functionality
   auto sum_likelihood = static_cast<RooFit::TestStatistics::RooSumL *>(likelihood.get());

   EXPECT_STREQ("RooUnbinnedL", (sum_likelihood->GetComponents()[0]->GetClassName()).c_str());
   EXPECT_STREQ("RooUnbinnedL::model_A", (sum_likelihood->GetComponents()[0]->GetInfo()).c_str());
   EXPECT_STREQ("RooUnbinnedL", (sum_likelihood->GetComponents()[1]->GetClassName()).c_str());
   EXPECT_STREQ("RooUnbinnedL::model_B", (sum_likelihood->GetComponents()[1]->GetInfo()).c_str());
   EXPECT_STREQ("RooSubsidiaryL", (sum_likelihood->GetComponents()[2]->GetClassName()).c_str());
   EXPECT_STREQ("RooSubsidiaryL::likelihood for pdf model", (sum_likelihood->GetComponents()[2]->GetInfo()).c_str());
}

TEST_F(BinnedDatasetTest, EventSections)
{
   // Test whether the summed total of multiple sections gives the same result
   // as an evaluation with a single section over the whole event range.
   pdf->setAttribute("BinnedLikelihood");
   data = std::unique_ptr<RooDataHist>{pdf->generateBinned(*w.var("x"))};
   likelihood = RooFit::TestStatistics::buildLikelihood(pdf, data.get());

   auto whole = likelihood->evaluatePartition({0, 1}, 0, 0);
   auto part1 = likelihood->evaluatePartition({0, 0.5}, 0, 0);
   auto part2 = likelihood->evaluatePartition({0.5, 1}, 0, 0);

   // We cannot EXPECT_EQ in this test, because we compare actually different
   // calculations. The multiple additions and FMA operations involved in the
   // calculation of the multiple parts introduces different rounding errors
   // on the CPU level than the single calculation over all events at once.
   EXPECT_DOUBLE_EQ(whole.Sum(), (part1 + part2).Sum());
}

TEST_F(SimBinnedConstrainedTest, EventSections)
{
   // Test whether the summed total of multiple sections gives the same result
   // as an evaluation with a single section over the whole event range.
   likelihood = RooFit::TestStatistics::NLLFactory{*pdf, *data}
                   .GlobalObservables({*w.var("alpha_bkg_obs_A"), *w.var("alpha_bkg_obs_B")})
                   .build();

   auto whole = likelihood->evaluatePartition({0, 1}, 0, likelihood->getNComponents());
   auto part1 = likelihood->evaluatePartition({0, 0.5}, 0, likelihood->getNComponents());
   auto part2 = likelihood->evaluatePartition({0.5, 1}, 0, likelihood->getNComponents());

   // See comment in first EventSections test for explanation on why no EXPECT_EQ.
   EXPECT_DOUBLE_EQ(whole.Sum(), (part1 + part2).Sum());

   auto part1of4 = likelihood->evaluatePartition({0, 0.25}, 0, likelihood->getNComponents());
   auto part2of4 = likelihood->evaluatePartition({0.25, 0.5}, 0, likelihood->getNComponents());
   auto part3of4 = likelihood->evaluatePartition({0.5, 0.75}, 0, likelihood->getNComponents());
   auto part4of4 = likelihood->evaluatePartition({0.75, 1}, 0, likelihood->getNComponents());

   // See comment in first EventSections test for explanation on why no EXPECT_EQ.
   EXPECT_DOUBLE_EQ(whole.Sum(), (part1of4 + part2of4 + part3of4 + part4of4).Sum());
}

TEST_F(RooAbsLTest, SubEventSections)
{
   // Here we test behavior when sections are smaller than a single event or
   // generally speaking when they do not align perfectly with integer event
   // boundaries. For instance, when we have 10 events, but 9 sections (one
   // must contain two events, but which section?), or 11 (one must be empty,
   // but which one?).
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 10);
   likelihood = RooFit::TestStatistics::buildLikelihood(pdf, data.get());

   auto whole = likelihood->evaluatePartition({0, 1}, 0, likelihood->getNComponents());
   ROOT::Math::KahanSum<double> nine_parts;
   ROOT::Math::KahanSum<double> eleven_parts;
   ROOT::Math::KahanSum<double> twenty_parts;

   for (std::size_t ix = 0; ix < 9; ++ix) {
      nine_parts += likelihood->evaluatePartition({static_cast<double>(ix) / 9, static_cast<double>(ix + 1) / 9}, 0,
                                                  likelihood->getNComponents());
   }
   // See comment in first EventSections test for explanation on why no EXPECT_EQ.
   EXPECT_DOUBLE_EQ(whole.Sum(), nine_parts.Sum());

   for (std::size_t ix = 0; ix < 11; ++ix) {
      eleven_parts += likelihood->evaluatePartition({static_cast<double>(ix) / 11, static_cast<double>(ix + 1) / 11}, 0,
                                                    likelihood->getNComponents());
   }
   // See comment in first EventSections test for explanation on why no EXPECT_EQ.
   EXPECT_DOUBLE_EQ(whole.Sum(), eleven_parts.Sum());

   for (std::size_t ix = 0; ix < 20; ++ix) {
      twenty_parts += likelihood->evaluatePartition({static_cast<double>(ix) / 20, static_cast<double>(ix + 1) / 20}, 0,
                                                    likelihood->getNComponents());
   }
   // See comment in first EventSections test for explanation on why no EXPECT_EQ.
   EXPECT_DOUBLE_EQ(whole.Sum(), twenty_parts.Sum());
}

TEST_F(SimBinnedConstrainedTest, SubEventSections)
{
   // Like RooAbsLTest.SubEventSections, but here we can also check behavior
   // for when component likelihoods of a top-level RooSumL have a different
   // number (less) of events than the RooSumL itself. Moreover, this more
   // complex likelihood has an extended term and a subsidiary component
   // which also depend on section so will also be checked here.
   likelihood = RooFit::TestStatistics::NLLFactory{*pdf, *data}
                   .GlobalObservables({*w.var("alpha_bkg_obs_A"), *w.var("alpha_bkg_obs_B")})
                   .build();

   auto whole = likelihood->evaluatePartition({0, 1}, 0, likelihood->getNComponents());

   ROOT::Math::KahanSum<double> N_events_total_parts;
   auto N_events_total = likelihood->getNEvents();

   for (std::size_t ix = 0; ix < N_events_total; ++ix) {
      N_events_total_parts += likelihood->evaluatePartition(
         {static_cast<double>(ix) / N_events_total, static_cast<double>(ix + 1) / N_events_total}, 0,
         likelihood->getNComponents());
   }
   // See comment in first EventSections test for explanation on why no EXPECT_EQ.
   EXPECT_DOUBLE_EQ(whole.Sum(), N_events_total_parts.Sum());

   // now let's do it again over a number of sections 3 times the number of events
   ROOT::Math::KahanSum<double> thrice_N_events_total_parts;

   for (std::size_t ix = 0; ix < 3 * N_events_total; ++ix) {
      thrice_N_events_total_parts += likelihood->evaluatePartition(
         {static_cast<double>(ix) / (3 * N_events_total), static_cast<double>(ix + 1) / (3 * N_events_total)}, 0,
         likelihood->getNComponents());
   }
   // See comment in BinnedDatasetTest.EventSections for explanation on why no EXPECT_EQ.
   EXPECT_DOUBLE_EQ(whole.Sum(), thrice_N_events_total_parts.Sum());
}

TEST_F(RooAbsLTest, VSRooNLLVar)
{
   // compare the value of the likelihood to that generated by a similarly configured RooNLLVar
   std::tie(nll, pdf, data, values) = generate_1D_gaussian_pdf_nll(w, 10);
   likelihood = RooFit::TestStatistics::buildLikelihood(pdf, data.get());

   auto AbsL_value = likelihood->evaluatePartition({0, 1}, 0, likelihood->getNComponents());
   auto RooNLL_value = nll->getVal();
   EXPECT_EQ(AbsL_value.Sum(), RooNLL_value);
}

TEST_F(BinnedDatasetTest, VSRooNLLVar)
{
   // compare the value of the likelihood to that generated by a similarly configured RooNLLVar
   pdf->setAttribute("BinnedLikelihood");
   data = std::unique_ptr<RooAbsData>{pdf->generateBinned(*w.var("x"))};
   likelihood = RooFit::TestStatistics::buildLikelihood(pdf, data.get());
   nll = std::unique_ptr<RooAbsReal>{pdf->createNLL(*data)};

   auto AbsL_value = likelihood->evaluatePartition({0, 1}, 0, likelihood->getNComponents());
   auto RooNLL_value = nll->getVal();
   EXPECT_EQ(AbsL_value.Sum(), RooNLL_value);
}

TEST_F(SimBinnedConstrainedTest, VSRooNLLVar)
{
   RooArgSet globalObservables{*w.var("alpha_bkg_obs_A"), *w.var("alpha_bkg_obs_B")};

   // compare the value of the likelihood to that generated by a similarly configured RooNLLVar
   likelihood = RooFit::TestStatistics::NLLFactory(*pdf, *data).GlobalObservables(globalObservables).build();
   nll = std::unique_ptr<RooAbsReal>{pdf->createNLL(*data, RooFit::GlobalObservables(globalObservables))};

   auto AbsL_value = likelihood->evaluatePartition({0, 1}, 0, likelihood->getNComponents());
   auto RooNLL_value = nll->getVal();
   EXPECT_EQ(AbsL_value.Sum(), RooNLL_value);
}

// TODO: add tests covering all constOptimizeTestStatistic opcode cases.
