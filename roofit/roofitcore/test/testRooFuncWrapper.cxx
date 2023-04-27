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
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooFitResult.h>
#include <RooFuncWrapper.h>
#include <RooHelpers.h>
#include <RooMinimizer.h>
#include <RooRealVar.h>
#include <RooRandom.h>
#include <RooWorkspace.h>

#include <TROOT.h>
#include <TSystem.h>
#include <TMath.h>
#include <Math/Factory.h>
#include <Math/Minimizer.h>

#include <gtest/gtest.h>

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

void randomizeParameters(const RooArgSet &parameters, ULong_t seed = 0)
{
   auto random = RooRandom::randomGenerator();
   if (seed != 0)
      random->SetSeed(seed);

   for (auto param : parameters) {
      auto par = static_cast<RooAbsRealLValue *>(param);
      const double uni = random->Uniform();
      const double min = par->getMin();
      const double max = par->getMax();
      par->setVal(min + uni * (max - min));
   }
}

} // namespace

TEST(RooFuncWrapper, GaussianNormalizedHardcoded)
{
   using namespace RooFit;
   auto inf = std::numeric_limits<double>::infinity();

   RooWorkspace ws;
   ws.import(RooRealVar{"x", "x", 0, -inf, inf});
   ws.factory("Gaussian::gauss(x, mu[0, -10, 10], sigma[2.0, 0.01, 10])");

   RooAbsPdf &gauss = *ws.pdf("gauss");
   RooRealVar &x = *ws.var("x");
   RooRealVar &mu = *ws.var("mu");
   RooRealVar &sigma = *ws.var("sigma");

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
   RooWorkspace ws;
   ws.import(RooRealVar{"x", "x", 0, -10, std::numeric_limits<double>::infinity()});
   ws.factory("sum::mu_shifted(mu[0, -10, 10], shift[1.0, -10, 10])");
   ws.factory("prod::sigma_scaled(sigma[2.0, 0.01, 10], 1.5)");
   ws.factory("Gaussian::gauss(x, mu_shifted, sigma_scaled)");

   RooAbsPdf &gauss = *ws.pdf("gauss");
   RooRealVar &x = *ws.var("x");
   RooRealVar &mu = *ws.var("mu");

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
   RooWorkspace ws;
   ws.factory("Exponential::expo(x[1.0, 0, 10], c[0.1, 0, 10])");

   RooAbsPdf &expo = *ws.pdf("expo");
   RooRealVar &x = *ws.var("x");

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

/// Initial test for NLL minimization that was not based on any other tutorial/test.
TEST(RooFuncWrapper, Nll)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;
   ws.factory("sum::mu_shifted(mu[0, -10, 10], shift[1.0, -10, 10])");
   ws.factory("prod::sigma_scaled(sigma[3.0, 0.01, 10], 1.5)");
   ws.factory("Gaussian::gauss(x[0, -10, 10], mu_shifted, sigma_scaled)");

   RooAbsPdf &pdf = *ws.pdf("gauss");
   RooRealVar &x = *ws.var("x");
   RooRealVar &mu = *ws.var("mu");

   RooArgSet normSet{x};

   std::size_t nEvents = 10;
   std::unique_ptr<RooDataSet> data0{pdf.generate(x, nEvents)};
   std::unique_ptr<RooAbsData> data{data0->binnedClone()};
   std::unique_ptr<RooAbsReal> nllRef{pdf.createNLL(*data, RooFit::BatchMode("cpu"))};
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

/// Test based on the rf301 tutorial.
TEST(RooFuncWrapper, NllPolyVar)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;
   // Create function f(y) = a0 + a1*y + y*y*y
   ws.factory("PolyVar::fy(y[-5, 5], {a0[-0.5, -5, 5], a1[-0.5, -1, 1], y})");
   // Create gauss(x,f(y),s)
   ws.factory("Gaussian::gauss(x[-5, 5], fy, sigma[0.5, 0.01, 10])");

   RooRealVar &x = *ws.var("x");
   RooRealVar &y = *ws.var("y");
   RooAbsPdf &pdf = *ws.pdf("gauss");

   RooArgSet normSet{x};

   std::size_t nEvents = 10;
   std::unique_ptr<RooDataSet> data0{pdf.generate({x, y}, nEvents)};
   std::unique_ptr<RooAbsData> data{data0->binnedClone()};
   std::unique_ptr<RooAbsReal> nllRef{pdf.createNLL(*data, ConditionalObservables(y), BatchMode("cpu"))};
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

/// Test based on the rf201 tutorial.
TEST(RooFuncWrapper, NllAddPdf)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;
   ws.factory("Gaussian::sig1(x[0, 10], mean[5, -10, 10], sigma1[0.50, .01, 10])");
   ws.factory("Gaussian::sig2(x, mean, sigma2[5, .01, 10])");
   ws.factory("Chebychev::bkg(x, {a0[0.3, 0., 0.5], a1[0.2, 0., 0.5]})");
   ws.factory("SUM::sig(sig1frac[0.8, 0.0, 1.0] * sig1, sig2)");
   ws.factory("SUM::model(bkgfrac[0.5, 0.0, 1.0] * bkg, sig)");

   RooRealVar &x = *ws.var("x");
   RooAbsPdf &model = *ws.pdf("model");

   RooArgSet normSet{x};

   std::size_t nEvents = 10;
   std::unique_ptr<RooDataSet> data0{model.generate({x}, nEvents)};
   std::unique_ptr<RooAbsData> data{data0->binnedClone()};
   std::unique_ptr<RooAbsReal> nllRef{model.createNLL(*data, RooFit::BatchMode("cpu"))};
   auto nllRefResolved = static_cast<RooAbsReal *>(nllRef->servers()[0]);

   RooFuncWrapper nllFunc("myNllAddPdf", "myNllAddPdf", *nllRefResolved, normSet, data.get());

   // Check if functions results are the same even after changing parameters.
   EXPECT_NEAR(nllRef->getVal(normSet), nllFunc.getVal(), 1e-8);

   // Check if the parameter layout and size is the same.
   RooArgSet paramsRefNll;
   nllRef->getParameters(nullptr, paramsRefNll);
   RooArgSet paramsMyNLL;
   nllFunc.getParameters(&normSet, paramsMyNLL);

   randomizeParameters(paramsMyNLL, 1337);

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
   EXPECT_TRUE(result->isIdentical(*resultRef, 1e-3));
   EXPECT_TRUE(resultAd->isIdentical(*resultRef, 1e-3));
}

/// Test based on the rf604 tutorial.
TEST(RooFuncWrapper, NllConstraintSum)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;
   ws.factory("RealSumFunc::mu_func({mu[-1, -10, 10], 4.0}, {1.1, 0.3})");
   ws.factory("Gaussian::gauss(x[-10, 10], mu_func, sigma[2, 0.1, 10])");
   ws.factory("Polynomial::poly(x)");
   ws.factory("SUM::model(f[0.5, 0.0, 1.0] * gauss, poly)");
   ws.factory("Gaussian::fconstext(f, 0.2, 0.1)");

   RooRealVar &x = *ws.var("x");
   RooAbsPdf &model = *ws.pdf("model");

   RooArgSet normSet{x};

   std::size_t nEvents = 10;
   std::unique_ptr<RooDataSet> data0{model.generate({x}, nEvents)};
   std::unique_ptr<RooAbsData> data{data0->binnedClone()};
   std::unique_ptr<RooAbsReal> nllRef{
      model.createNLL(*data, RooFit::ExternalConstraints(*ws.pdf("fconstext")), RooFit::BatchMode("cpu"))};
   auto nllRefResolved = static_cast<RooAbsReal *>(nllRef->servers()[0]);

   RooFuncWrapper nllFunc("myNllConstraintSum", "myNllConstraintSum", *nllRefResolved, normSet, data.get());

   // Check if functions results are the same even after changing parameters.
   EXPECT_NEAR(nllRef->getVal(normSet), nllFunc.getVal(), 1e-8);

   // Check if the parameter layout and size is the same.
   RooArgSet paramsRefNll;
   nllRef->getParameters(nullptr, paramsRefNll);
   RooArgSet paramsMyNLL;
   nllFunc.getParameters(&normSet, paramsMyNLL);

   randomizeParameters(paramsMyNLL, 1337);

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
