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

#include <RooAbsPdf.h>
#include <RooCurve.h>
#include <RooDataSet.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooMinimizer.h>
#include <RooPlot.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

using namespace RooFit;

/// Plotting and minimization with RooFit::TestStatistics: minimize a
/// RooFit::TestStatistics likelihood with the parallel gradient and plot it as
/// a function of the parameter. The results are validated against analytic
/// expectations and against direct evaluations of the likelihood, instead of
/// the RooUnitTest reference file that was used before.
TEST(TestStatisticsPlot, RooRealL)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl{RooFit::WARNING};

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

   RooRealVar &mu = *w.var("mu");

   // P e r f o r m   a  p a r a l l e l  l i k e l i h o o d  m i n i m i z a t i o n
   // --------------------------------------------------------------------------------

   // Creating a RooAbsL likelihood
   std::unique_ptr<RooAbsReal> likelihood{w.pdf("model")->createNLL(d, ModularL(true))};

   // Creating a minimizer and explicitly setting type of parallelization
   std::size_t nWorkers = 1;
   RooMinimizer::Config cfg;
   cfg.parallelize = nWorkers;
   cfg.enableParallelDescent = false;
   cfg.enableParallelGradient = true;
   RooMinimizer m(*likelihood, cfg);

   // Minimize
   m.setPrintLevel(-1);
   m.migrad();

   // The analytic maximum likelihood estimate is at Nexp = Nobs, i.e.
   // mu = (Nobs - B) / S. The tolerance is at the scale of the Minuit
   // convergence criterion, given that sigma(mu) = sqrt(Nobs) / S = 0.5.
   EXPECT_NEAR(mu.getVal(), 0.5, 0.05 * 0.5);

   // C o n v e r t  t o  R o o R e a l L  a n d  p l o t
   // ---------------------------------------------------
   std::unique_ptr<RooPlot> xframe{mu.frame(-1, 10)};
   likelihood->plotOn(xframe.get(), RooFit::Precision(1));
   RooCurve *curve = xframe->getCurve();
   ASSERT_NE(curve, nullptr);
   ASSERT_GT(curve->GetN(), 1);

   // Every point of the plotted curve must match a direct evaluation of the
   // likelihood at that parameter value
   for (int i = 0; i < curve->GetN(); ++i) {
      const double muVal = curve->GetPointX(i);
      if (muVal < mu.getMin() || muVal > mu.getMax())
         continue;
      mu.setVal(muVal);
      const double directVal = likelihood->getVal();
      EXPECT_NEAR(curve->GetPointY(i), directVal, 1e-6 * std::max(1.0, std::abs(directVal)))
         << "curve point " << i << " at mu = " << muVal;
   }

   // The likelihood must also match the analytically known Poisson -log L,
   // which is defined up to a mu-independent constant. The comparison is
   // restricted to moderate Nexp values, where the truncation of the Poisson
   // normalization to the observable range is negligible.
   auto analyticNll = [](double muVal) {
      const double nexp = 10 * muVal + 20;
      return nexp - 25 * std::log(nexp);
   };
   mu.setVal(0.5);
   const double nllOffset = likelihood->getVal() - analyticNll(0.5);
   for (double muVal : {0.0, 1.0, 2.0, 3.0}) {
      mu.setVal(muVal);
      EXPECT_NEAR(likelihood->getVal(), analyticNll(muVal) + nllOffset, 1e-4) << "at mu = " << muVal;
   }
}
