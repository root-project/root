/*
 * Project: RooFit
 * Authors:
 *   Garima Singh, CERN 2023
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooAbsPdf.h>
#include <RooAddition.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooFitResult.h>
#include <RooExponential.h>
#include <RooFuncWrapper.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooMinimizer.h>
#include <RooProduct.h>
#include <RooPolyVar.h>
#include <RooRealVar.h>

#include <TROOT.h>
#include <TSystem.h>
#include <TMath.h>
#include <Math/Factory.h>
#include <Math/Minimizer.h>

#include "gtest/gtest.h"

namespace {

// Function to get the derivative of pdf wrt var.
double getNumDerivative(const RooAbsReal &pdf, RooRealVar &var, const RooArgSet &normSet, double eps = 1e-8)
{
   double orig = var.getVal();
   if (!var.inRange(orig + eps, nullptr)) {
      throw std::runtime_error("getNumDerivative(): positive variation outside of range!");
   }
   if (!var.inRange(orig - eps, nullptr)) {
      throw std::runtime_error("getNumDerivative(): negative variation outside of range!");
   }
   var.setVal(orig + eps);
   double plus = pdf.getVal(normSet);
   var.setVal(orig - eps);
   double minus = pdf.getVal(normSet);
   var.setVal(orig);

   return (plus - minus) / (2 * eps);
}

} // namespace

TEST(RooFuncWrapper, GaussianNormalizedHardcoded)
{
   using namespace RooFit;

   auto inf = std::numeric_limits<double>::infinity();
   RooRealVar x("x", "x", 0, -inf, inf);
   RooRealVar mu("mu", "mu", 0, -10, 10);
   RooRealVar sigma("sigma", "sigma", 2.0, 0.01, 10);
   RooGaussian gauss{"gauss", "gauss", x, mu, sigma};

   RooArgSet normSet{x};
   RooArgSet paramsGauss;
   RooArgSet paramsMyGauss;

   std::string func = "const double arg = params[0] - params[1];"
                      "const double sig = params[2];"
                      "double out = std::exp(-0.5 * arg * arg / (sig * sig));"
                      "return 1. / (std::sqrt(TMath::TwoPi()) * sig) * out;";
   RooFuncWrapper gaussFunc("myGauss1", "myGauss1", func, {x, mu, sigma}, {});

   // Check if functions results are the same even after changing parameters.
   EXPECT_NEAR(gauss.getVal(normSet), gaussFunc.getVal(), 1e-8);

   mu.setVal(1);
   EXPECT_NEAR(gauss.getVal(normSet), gaussFunc.getVal(), 1e-8);

   // Check if the parameter layout and size is the same.
   gauss.getParameters(&normSet, paramsGauss);
   gaussFunc.getParameters(&normSet, paramsMyGauss);

   EXPECT_TRUE(paramsMyGauss.hasSameLayout(paramsGauss));
   EXPECT_EQ(paramsMyGauss.size(), paramsGauss.size());

   // Get AD based derivative
   // Get number of actual parameters directly from the wrapper as not always will they be the same as paramsMyGauss.
   std::vector<double> dMyGauss(gaussFunc.getNumParams(), 0);
   gaussFunc.getGradient(dMyGauss.data());

   // Check if derivatives are equal
   EXPECT_NEAR(getNumDerivative(gauss, x, normSet), dMyGauss[0], 1e-8);
   EXPECT_NEAR(getNumDerivative(gauss, mu, normSet), dMyGauss[1], 1e-8);
   EXPECT_NEAR(getNumDerivative(gauss, sigma, normSet), dMyGauss[2], 1e-8);
}

TEST(RooFuncWrapper, GaussianNormalized)
{
   RooRealVar x("x", "x", 0, -10, std::numeric_limits<double>::infinity());

   RooRealVar mu("mu", "mu", 0, -10, 10);
   RooRealVar shift("shift", "shift", 1.0, -10, 10);
   RooAddition muShifted("mu_shifted", "mu_shifted", {mu, shift});

   RooRealVar sigma("sigma", "sigma", 2.0, 0.01, 10);
   RooConstVar scale("scale", "scale", 1.5);
   RooProduct sigmaScaled("sigma_scaled", "sigma_scaled", sigma, scale);

   RooGaussian gauss{"gauss", "gauss", x, muShifted, sigmaScaled};

   RooArgSet normSet{x};

   RooFuncWrapper gaussFunc("myGauss3", "myGauss3", gauss, normSet);

   RooArgSet paramsGauss;
   gauss.getParameters(nullptr, paramsGauss);

   // Check if functions results are the same even after changing parameters.
   EXPECT_NEAR(gauss.getVal(normSet), gaussFunc.getVal(), 1e-8);

   mu.setVal(1);
   EXPECT_NEAR(gauss.getVal(normSet), gaussFunc.getVal(), 1e-8);

   // Get AD based derivative
   std::vector<double> dMyGauss(gaussFunc.getNumParams(), 0);
   gaussFunc.getGradient(dMyGauss.data());

   // Check if derivatives are equal
   for (std::size_t i = 0; i < paramsGauss.size(); ++i) {
      EXPECT_NEAR(getNumDerivative(gauss, static_cast<RooRealVar &>(*paramsGauss[i]), normSet), dMyGauss[i], 1e-8);
   }
}

TEST(RooFuncWrapper, Exponential)
{
   RooRealVar x("x", "x", 1.0, 0, 10);
   RooRealVar c("c", "c", 0.1, 0, 10);

   RooExponential expo("expo", "expo", x, c);

   RooArgSet normSet{x};

   RooFuncWrapper expoFunc("expo", "expo", expo, normSet);

   RooArgSet params;
   expo.getParameters(nullptr, params);

   EXPECT_NEAR(expo.getVal(normSet), expoFunc.getVal(), 1e-8);

   // Get AD based derivative
   std::vector<double> dExpo(expoFunc.getNumParams(), 0);
   expoFunc.getGradient(dExpo.data());

   // Check if derivatives are equal
   for (std::size_t i = 0; i < params.size(); ++i) {
      EXPECT_NEAR(getNumDerivative(expo, static_cast<RooRealVar &>(*params[i]), normSet), dExpo[i], 1e-8)
         << params[i]->GetName();
   }
}

TEST(RooFuncWrapper, Nll)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRealVar x("x", "x", 0, -10, 10);

   RooRealVar mu("mu", "mu", 0, -10, 10);
   RooRealVar shift("shift", "shift", 1.0, -10, 10);
   RooAddition muShifted("mu_shifted", "mu_shifted", {mu, shift});

   RooRealVar sigma("sigma", "sigma", 3.0, 0.01, 10);
   RooConstVar scale("scale", "scale", 1.5);
   RooProduct sigmaScaled("sigma_scaled", "sigma_scaled", sigma, scale);

   RooGaussian gauss{"gauss", "gauss", x, muShifted, sigmaScaled};

   RooArgSet normSet{x};

   std::size_t nEvents = 10;
   std::unique_ptr<RooDataSet> data0{gauss.generate(x, nEvents)};
   std::unique_ptr<RooAbsData> data{data0->binnedClone()};
   std::unique_ptr<RooAbsReal> nllRef{gauss.createNLL(*data, RooFit::BatchMode("cpu"))};
   auto nllRefResolved = static_cast<RooAbsReal *>(nllRef->servers()[0]);

   RooFuncWrapper nllFunc("myNll", "myNll", *nllRefResolved, normSet, data.get());

   // Check if functions results are the same even after changing parameters.
   EXPECT_NEAR(nllRef->getVal(normSet), nllFunc.getVal(), 1e-8);

   mu.setVal(1);
   EXPECT_NEAR(nllRef->getVal(normSet), nllFunc.getVal(), 1e-8);

   // Check if the parameter layout and size is the same.
   RooArgSet paramsRefNll;
   nllRef->getParameters(nullptr, paramsRefNll);
   RooArgSet paramsMyNLL;
   nllFunc.getParameters(&normSet, paramsMyNLL);

   EXPECT_TRUE(paramsMyNLL.hasSameLayout(paramsRefNll));
   EXPECT_EQ(paramsMyNLL.size(), paramsRefNll.size());

   // Get AD based derivative
   std::vector<double> dMyNLL(nllFunc.getNumParams(), 0);
   nllFunc.getGradient(dMyNLL.data());

   // Check if derivatives are equal
   for (std::size_t i = 0; i < paramsMyNLL.size(); ++i) {
      EXPECT_NEAR(getNumDerivative(*nllRef, static_cast<RooRealVar &>(*paramsMyNLL[i]), normSet), dMyNLL[i], 1e-4);
   }

   // Remember parameter state before minimization
   RooArgSet parametersOrig;
   paramsRefNll.snapshot(parametersOrig);

   auto runMinimizer = [&](RooAbsReal &absReal, RooMinimizer::Config cfg = {}) -> std::unique_ptr<RooFitResult> {
      RooMinimizer m{absReal, cfg};
      m.setPrintLevel(-1);
      m.setStrategy(0);
      m.minimize("Minuit2");
      auto result = std::unique_ptr<RooFitResult>{m.save()};
      // reset parameters
      paramsRefNll.assign(parametersOrig);
      return result;
   };

   // Minimize the RooFuncWrapper Implementation
   auto result = runMinimizer(nllFunc);

   // Minimize the RooFuncWrapper Implementation with AD
   RooMinimizer::Config minimizerCfgAd;
   std::size_t nGradientCalls = 0;
   minimizerCfgAd.gradFunc = [&](double *out) {
      nllFunc.getGradient(out);
      ++nGradientCalls;
   };
   auto resultAd = runMinimizer(nllFunc, minimizerCfgAd);
   EXPECT_GE(nGradientCalls, 1); // make sure the gradient function was actually called

   // Minimize the reference NLL
   auto resultRef = runMinimizer(*nllRef);

   // Compare minimization results
   EXPECT_TRUE(result->isIdentical(*resultRef, 1e-4));
   EXPECT_TRUE(resultAd->isIdentical(*resultRef, 1e-4));
}

TEST(RooFuncWrapper, NllPolyVar)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRealVar x("x", "x", -5, 5);
   RooRealVar y("y", "y", -5, 5);

   // Create function f(y) = a0 + a1*y
   RooRealVar a0("a0", "a0", -0.5, -5, 5);
   RooRealVar a1("a1", "a1", -0.5, -1, 1);
   RooPolyVar fy("fy", "fy", y, RooArgSet(a0, a1, y));

   // Create gauss(x,f(y),s)
   RooRealVar sigma("sigma", "width of gaussian", 0.5, 0.01, 10);
   RooGaussian gauss("gauss", "Gaussian with shifting mean", x, fy, sigma);

   RooArgSet normSet{x};

   std::size_t nEvents = 10;
   std::unique_ptr<RooDataSet> data0{gauss.generate({x, y}, nEvents)};
   std::unique_ptr<RooAbsData> data{data0->binnedClone()};
   std::unique_ptr<RooAbsReal> nllRef{
      gauss.createNLL(*data, RooFit::ConditionalObservables(y), RooFit::BatchMode("cpu"))};
   auto nllRefResolved = static_cast<RooAbsReal *>(nllRef->servers()[0]);

   RooFuncWrapper nllFunc("myNllPolyVar", "myNllPolyVar", *nllRefResolved, normSet, data.get());

   // Check if functions results are the same even after changing parameters.
   EXPECT_NEAR(nllRef->getVal(normSet), nllFunc.getVal(), 1e-8);

   y.setVal(1);
   EXPECT_NEAR(nllRef->getVal(normSet), nllFunc.getVal(), 1e-8);

   // Check if the parameter layout and size is the same.
   RooArgSet paramsRefNll;
   nllRef->getParameters(nullptr, paramsRefNll);
   RooArgSet paramsMyNLL;
   nllFunc.getParameters(&normSet, paramsMyNLL);

   EXPECT_TRUE(paramsMyNLL.hasSameLayout(paramsRefNll));
   EXPECT_EQ(paramsMyNLL.size(), paramsRefNll.size());

   // Get AD based derivative
   std::vector<double> dMyNLL(nllFunc.getNumParams(), 0);
   nllFunc.getGradient(dMyNLL.data());

   // Check if derivatives are equal
   for (std::size_t i = 0; i < paramsMyNLL.size(); ++i) {
      EXPECT_NEAR(getNumDerivative(*nllRef, static_cast<RooRealVar &>(*paramsMyNLL[i]), normSet), dMyNLL[i], 1e-4);
   }

   // Remember parameter state before minimization
   RooArgSet parametersOrig;
   paramsRefNll.snapshot(parametersOrig);

   auto runMinimizer = [&](RooAbsReal &absReal, RooMinimizer::Config cfg = {}) -> std::unique_ptr<RooFitResult> {
      RooMinimizer m{absReal, cfg};
      m.setPrintLevel(-1);
      m.setStrategy(0);
      m.minimize("Minuit2");
      auto result = std::unique_ptr<RooFitResult>{m.save()};
      // reset parameters
      paramsRefNll.assign(parametersOrig);
      return result;
   };

   // Minimize the RooFuncWrapper Implementation
   auto result = runMinimizer(nllFunc);

   // Minimize the RooFuncWrapper Implementation with AD
   RooMinimizer::Config minimizerCfgAd;
   std::size_t nGradientCalls = 0;
   minimizerCfgAd.gradFunc = [&](double *out) {
      nllFunc.getGradient(out);
      ++nGradientCalls;
   };
   auto resultAd = runMinimizer(nllFunc, minimizerCfgAd);
   EXPECT_GE(nGradientCalls, 1); // make sure the gradient function was actually called

   // Minimize the reference NLL
   auto resultRef = runMinimizer(*nllRef);

   // Compare minimization results
   EXPECT_TRUE(result->isIdentical(*resultRef, 1e-4));
   EXPECT_TRUE(resultAd->isIdentical(*resultRef, 1e-4));
}
