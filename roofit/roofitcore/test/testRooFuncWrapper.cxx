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
#include <RooFuncWrapper.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooMinimizer.h>
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

std::unique_ptr<ROOT::Math::Minimizer>
doMinimization(RooFuncWrapper const &inFunc, RooArgSet const &parameters, bool useAnalyticGradient)
{
   std::unique_ptr<ROOT::Math::Minimizer> myMinimizer{ROOT::Math::Factory::CreateMinimizer("Minuit2")};

   myMinimizer->SetPrintLevel(-1);
   myMinimizer->SetErrorDef(0.5);
   myMinimizer->SetTolerance(1);
   myMinimizer->SetMaxFunctionCalls(1000);
   if (useAnalyticGradient)
      myMinimizer->SetFunction(inFunc.getGradFunctor());
   else
      myMinimizer->SetFunction(inFunc.getFunctor());

   int cnt = 0;
   for (auto *param : parameters) {
      auto realParam = static_cast<RooRealVar *>(param);
      myMinimizer->SetLimitedVariable(cnt++, realParam->GetName(), realParam->getVal(), realParam->getError(),
                                      realParam->getMin(), realParam->getMax());
   }

   myMinimizer->Minimize();

   return myMinimizer;
}

void checkMinimizationResults(ROOT::Math::Minimizer const &minimizer, RooArgSet const &parameters, double eps = 1e-8)
{
   int cnt = 0;
   for (auto *param : parameters) {
      auto realParam = static_cast<RooRealVar *>(param);
      EXPECT_NEAR(minimizer.X()[cnt], realParam->getVal(), eps);
      EXPECT_NEAR(minimizer.Errors()[cnt], realParam->getError(), eps);
      cnt++;
   }
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

   // Minimize the RooFuncWrapper Implementation
   // Compare reference with AD gradient based minimization and numerical gradient based minimization.
   auto myMinimizer = doMinimization(nllFunc, paramsMyNLL, true);
   auto myMinimizerNumDiff = doMinimization(nllFunc, paramsMyNLL, false);

   // Minimize the reference NLL
   RooMinimizer refMinimizer{*nllRef};
   refMinimizer.setPrintLevel(-1);
   refMinimizer.minimize("Minuit2");

   // Compare minimization results
   checkMinimizationResults(*myMinimizer, parameters, 1e-6);
   checkMinimizationResults(*myMinimizerNumDiff, parameters);
}

namespace {

std::string valName(RooAbsArg const &arg)
{
   std::string name = arg.GetName();
   std::replace(name.begin(), name.end(), '[', '_');
   std::replace(name.begin(), name.end(), ']', '_');
   return name + "Val";
}

std::string valToString(double val)
{
   // std::numeric_limits<double>::infinity() doesn't seem work with clad!
   if (val == std::numeric_limits<double>::infinity())
      return "1e30";
   if (val == -std::numeric_limits<double>::infinity())
      return "-1e30";
   return std::to_string(val);
}

std::string integralCode(RooRealIntegral const &integral)
{
   std::stringstream ss;
   // TODO: assert also that numIntCatVars() and numIntRealVars() are empty

   RooArgSet anaIntVars{integral.anaIntVars()};
   RooArgSet empty;
   auto range = integral.intRange();
   const int code = integral.integrand().getAnalyticalIntegralWN(anaIntVars, empty, nullptr, range);
   if (code == 1) {
      auto gauss = dynamic_cast<RooGaussian const *>(&integral.integrand());
      RooRealVar const &x = static_cast<RooRealVar const &>(gauss->getX());
      ss << "GaussianEvalIntegralOverX(" << valToString(x.getMin(range)) << "," << valToString(x.getMax()) << ", "
         << valName(gauss->getMean()) << ", " << valName(gauss->getSigma()) << ")";
   }
   return ss.str();
}

std::string gaussianCode(RooGaussian const &gauss)
{
   std::stringstream ss;

   ss << "GaussianEval(" << valName(gauss.getX()) << ", " << valName(gauss.getMean()) << ", "
      << valName(gauss.getSigma()) << ")";

   return ss.str();
}

std::string normalizedPdfCode(RooAbsArg const &pdf)
{
   std::stringstream ss;

   ss << valName(*pdf.servers()[0]) << " / " << valName(*pdf.servers()[1]);

   return ss.str();
}

std::string generateCode(RooAbsReal const &func, RooArgSet const &variables)
{

   RooArgSet nodes;
   RooHelpers::getSortedComputationGraph(func, nodes);

   std::stringstream ss;

   for (RooAbsArg *node : nodes) {
      auto var = dynamic_cast<RooRealVar *>(node);
      int idx = variables.index(node);
      if (var && idx >= 0) {
         ss << "const double " << valName(*var) << " = params[" << idx << "];\n";
      } else if (var) {
         ss << "const double " << valName(*var) << " = " << var->getVal() << ";\n";
      } else if (auto gauss = dynamic_cast<RooGaussian *>(node)) {
         ss << "const double " << valName(*gauss) << " = " << gaussianCode(*gauss) << ";\n";
      } else if (auto integral = dynamic_cast<RooRealIntegral *>(node)) {
         ss << "const double " << valName(*integral) << " = " << integralCode(*integral) << ";\n";
      } else if (node == &func) {
         ss << "const double " << valName(*node) << " = " << normalizedPdfCode(*node) << ";\n";
      }
   }
   ss << "return " << valName(func) << ";\n";

   std::string out = ss.str();

   return out;
}

} // namespace

TEST(RooFuncWrapper, GaussianNormalized)
{
   using namespace RooFit;

   // clang-format off
   gInterpreter->Declare(
   "double GaussianEval(double x, double mean, double sigma)"
   "{"
   "   const double arg = x - mean;"
   "   return std::exp(-0.5 * arg * arg / (sigma * sigma));"
   "}"
   ""
   "double GaussianEvalIntegralOverX(double /*xMin*/, double /*xMax*/, double mean, double sigma)"
   "{"
       // With clad, we can't use std::erfc yet, so we hardcode integration over infinity"
   "   return std::sqrt(TMath::TwoPi()) * sigma;"
   "}"
   );
   // clang-format on

   auto inf = std::numeric_limits<double>::infinity();
   RooRealVar x("x", "x", 0, -inf, inf);
   RooRealVar mu("mu", "mu", 0, -10, 10);
   RooRealVar sigma("sigma", "sigma", 2.0, 0.01, 10);
   RooGaussian gauss{"gauss", "gauss", x, mu, sigma};

   RooArgSet normSet{x};

   // Compile the computation graph for the norm set, such that we also get the
   // integrals explicitly in the graph
   std::unique_ptr<RooAbsReal> pdf{RooFit::Detail::compileForNormSet(gauss, normSet)};

   RooArgSet paramsGauss;
   pdf->getParameters(nullptr, paramsGauss);

   // The code generation in this test is a rough prototype for how the code
   // generation might work like in the end
   std::string func = generateCode(*pdf, paramsGauss);

   RooFuncWrapper gaussFunc("myGauss2", "myGauss2", func, paramsGauss, {});

   // Check if functions results are the same even after changing parameters.
   EXPECT_NEAR(pdf->getVal(normSet), gaussFunc.getVal(), 1e-8);

   mu.setVal(1);
   EXPECT_NEAR(pdf->getVal(normSet), gaussFunc.getVal(), 1e-8);

   // Get AD based derivative
   double dMyGauss[3] = {};
   gaussFunc.getGradient(dMyGauss);

   // Check if derivatives are equal
   for (std::size_t i = 0; i < paramsGauss.size(); ++i) {
      EXPECT_NEAR(getNumDerivative(*pdf, static_cast<RooRealVar &>(*paramsGauss[i]), normSet), dMyGauss[i], 1e-8);
   }
}
