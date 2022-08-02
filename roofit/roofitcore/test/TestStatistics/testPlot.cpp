/*
 * Project: RooFit
 * Authors:
 *   ZW, Zef Wolffs, NIKHEF, zefwolffs@gmail.com
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooWorkspace.h>
#include <RooPlot.h>
#include <RooDataSet.h>
#include <RooFit/TestStatistics/buildLikelihood.h>
#include <RooAbsPdf.h>
#include <RooRealVar.h>
#include <RooFit/TestStatistics/RooRealL.h>
#include <RooFit/MultiProcess/Config.h>
#include <RooMinimizer.h>
#include <RooUnitTest.h>

#include <TFile.h>

#include <gtest/gtest.h>

#include <memory>

using namespace RooFit;

class Environment : public testing::Environment {
public:
   void SetUp() override {
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
      ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");
   }
};

int main(int argc, char **argv)
{
   testing::InitGoogleTest(&argc, argv);
   testing::AddGlobalTestEnvironment(new Environment);
   return RUN_ALL_TESTS();
}

class TestRooRealLPlot : public RooUnitTest {
public:
   TestRooRealLPlot(TFile &refFile, bool writeRef, int verbose, std::string const &batchMode)
      : RooUnitTest("Plotting and minimization with RooFit::TestStatistics", &refFile, writeRef, verbose, batchMode){};
   bool testCode() override
   {

      // C r e a t e   m o d e l  a n d  d a t a
      // ---------------------------------------
      // Constructing a workspace with pdf and dataset
      RooWorkspace w("w");
      w.factory("expr::Nexp('mu*S+B',mu[1,-1,10],S[10],B[20])");
      w.factory("Poisson::model(Nobs[0,100],Nexp)");
      w.var("Nobs")->setBins(4);
      RooDataSet d("d", "d", *w.var("Nobs"));
      w.var("Nobs")->setVal(25);
      d.add(*w.var("Nobs"));

      // P e r f o r m   a  p a r a l l e l  l i k e l i h o o d  m i n i m i z a t i o n
      // --------------------------------------------------------------------------------

      // Creating a RooAbsL likelihood
      std::shared_ptr<RooFit::TestStatistics::RooAbsL> likelihood =
         RooFit::TestStatistics::buildLikelihood(w.pdf("model"), &d);

      // Creating a minimizer and explicitly setting type of parallelization
      std::size_t nWorkers = 1;
      RooFit::MultiProcess::Config::setDefaultNWorkers(nWorkers);
      RooMinimizer m(likelihood, RooFit::TestStatistics::LikelihoodMode::serial,
                     RooFit::TestStatistics::LikelihoodGradientMode::multiprocess);

      // Minimize
      m.migrad();

      // C o n v e r t  t o  R o o R e a l L  a n d  p l o t
      // ---------------------------------------------------

      // Create a RooRealL which has plotting functionality
      std::shared_ptr<RooFit::TestStatistics::RooRealL> likelihood_real(
         new RooFit::TestStatistics::RooRealL("likelihood", "", likelihood));
      RooPlot *xframe = w.var("mu")->frame(-1, 10);
      likelihood_real->plotOn(xframe, RooFit::Precision(1));

      // Clean up the minimizer
      m.cleanup();

      // --- Post processing for RooUnitTest ---
      regPlot(xframe, "TestRooRealLPlot_plot");

      return true;
   }
};

TEST(TestStatisticsPlot, RooRealL)
{
   // Run the RooUnitTest and assert that it succeeds with gtest

   RooUnitTest::setMemDir(gDirectory);

   TFile fref("TestStatistics_ref.root");

   TestRooRealLPlot plotTest{fref, false, 0, "off"};
   bool result = plotTest.runTest();
   ASSERT_TRUE(result);
}
