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
#include <RooFitResult.h>
#include <RooFuncWrapper.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooMinimizer.h>
#include <RooProduct.h>
#include <RooRealIntegral.h>
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
   double dMyGauss[3] = {};
   gaussFunc.getGradient(dMyGauss);

   // Check if derivatives are equal
   EXPECT_NEAR(getNumDerivative(gauss, x, normSet), dMyGauss[0], 1e-8);
   EXPECT_NEAR(getNumDerivative(gauss, mu, normSet), dMyGauss[1], 1e-8);
   EXPECT_NEAR(getNumDerivative(gauss, sigma, normSet), dMyGauss[2], 1e-8);
}

TEST(RooFuncWrapper, NllWithObservables)
{
   using namespace RooFit;

   auto inf = std::numeric_limits<double>::infinity();
   RooRealVar x("x", "x", 0, -inf, inf);
   RooRealVar mu("mu", "mu", 0, -10, 10);
   RooRealVar sigma("sigma", "sigma", 2.0, 0.01, 10);
   RooGaussian gauss{"gauss", "gauss", x, mu, sigma};

   mu.setError(2);
   sigma.setError(1);

   RooArgSet normSet{x};

   std::size_t nEvents = 10;
   std::unique_ptr<RooDataSet> data{gauss.generate(x, nEvents)};
   std::unique_ptr<RooAbsReal> nllRef{gauss.createNLL(*data)};

   RooArgSet parameters;
   gauss.getParameters(data->get(), parameters);

   RooArgSet observables;
   gauss.getObservables(data->get(), observables);

   // clang-format off
   std::stringstream func;
   func <<  "double nllSum = 0;"
            "const double sig = params[1];"
            "for (int i = 0; i <" << data->numEntries() << "; i++) {"
               "const double arg = obs[i] - params[0];"
               "double out = std::exp(-0.5 * arg * arg / (sig * sig));"
               "out = 1. / (std::sqrt(TMath::TwoPi()) * sig) * out;"
               "nllSum -= std::log(out);"
            "}"
            "return nllSum;";
   // clang-format on
   RooFuncWrapper nllFunc("myNLL", "myNLL", func.str(), parameters, observables, data.get());

   // Check if functions results are the same even after changing parameters.
   EXPECT_NEAR(nllRef->getVal(normSet), nllFunc.getVal(), 1e-8);

   mu.setVal(1);
   EXPECT_NEAR(nllRef->getVal(normSet), nllFunc.getVal(), 1e-8);

   // Check if the parameter layout and size is the same.
   RooArgSet paramsMyNLL;
   nllFunc.getParameters(&normSet, paramsMyNLL);

   EXPECT_TRUE(paramsMyNLL.hasSameLayout(parameters));
   EXPECT_EQ(paramsMyNLL.size(), parameters.size());

   // Get AD based derivative
   double dMyNLL[2] = {};
   nllFunc.getGradient(dMyNLL);

   // Check if derivatives are equal
   EXPECT_NEAR(getNumDerivative(*nllRef, mu, normSet), dMyNLL[0], 1e-6);
   EXPECT_NEAR(getNumDerivative(*nllRef, sigma, normSet), dMyNLL[1], 1e-6);

   // Remember parameter state before minimization
   RooArgSet parametersOrig;
   parameters.snapshot(parametersOrig);

   auto runMinimizer = [&](RooAbsReal &absReal, RooMinimizer::Config cfg = {}) -> std::unique_ptr<RooFitResult> {
      RooMinimizer m{absReal, cfg};
      m.setPrintLevel(-1);
      m.setStrategy(0);
      m.minimize("Minuit2");
      auto result = std::unique_ptr<RooFitResult>{m.save()};
      // reset parameters
      parameters.assign(parametersOrig);
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
   // TODO: the (global) correlation coefficients are still wrong. This needs
   // to be understood next, and then we can also use the regular
   // isIdentical().
   EXPECT_TRUE(result->isIdenticalNoCov(*resultRef, 1e-5));
   EXPECT_TRUE(resultAd->isIdenticalNoCov(*resultRef, 1e-5));
}

TEST(RooFuncWrapper, GaussianNormalized)
{
   using namespace RooFit;

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
   double dMyGauss[3] = {};
   gaussFunc.getGradient(dMyGauss);

   // Check if derivatives are equal
   for (std::size_t i = 0; i < paramsGauss.size(); ++i) {
      EXPECT_NEAR(getNumDerivative(gauss, static_cast<RooRealVar &>(*paramsGauss[i]), normSet), dMyGauss[i], 1e-8);
   }
}
