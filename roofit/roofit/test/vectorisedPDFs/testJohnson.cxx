// Author: Stephan Hageboeck, CERN  26 Apr 2019

/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2019, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#include "VectorisedPDFTests.h"

#include "RooJohnson.h"
#include "RooAddition.h"
#include "RooFormulaVar.h"

class TestJohnson : public PDFTest {
protected:
   TestJohnson() : PDFTest("Johnson")
   {
      auto mass = std::make_unique<RooRealVar>("mass", "mass", 0., 0., 500.);
      auto mu = std::make_unique<RooRealVar>("mu", "Location parameter of normal distribution", 300., 0., 500.);
      auto lambda = std::make_unique<RooRealVar>("lambda", "Two sigma of normal distribution", 100., 10, 200.);
      auto gamma = std::make_unique<RooRealVar>("gamma", "gamma", 0.5, -10., 10.);
      auto delta = std::make_unique<RooRealVar>("delta", "delta", 2., 1., 10.);
      // delta is highly correlated with lambda. This troubles the fit.
      delta->setConstant();

      _pdf = std::make_unique<RooJohnson>("johnson", "johnson", *mass, *mu, *lambda, *gamma, *delta, -1.E300);

      _variables.addOwned(std::move(mass));

      //      _variablesToPlot.add(*mass);

      _parameters.addOwned(std::move(mu));
      _parameters.addOwned(std::move(lambda));
      _parameters.addOwned(std::move(gamma));
      _parameters.addOwned(std::move(delta));

      _toleranceCompareBatches = 6.E-12;
      _toleranceCompareLogs = 6.E-12;

      // For i686
      _toleranceCorrelation = 5.E-4;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestJohnson, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestJohnson, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestJohnson, CompareFixedNormLog)

FIT_TEST_SCALAR(TestJohnson, FitScalar)
FIT_TEST_BATCH(TestJohnson, FitBatch)
FIT_TEST_BATCH_VS_SCALAR(TestJohnson, FitBatchVsScalar)

class TestJohnsonInMassAndMu : public PDFTest {
protected:
   TestJohnsonInMassAndMu() : PDFTest("Johnson in mass and mu")
   {
      auto mass = std::make_unique<RooRealVar>("mass", "mass", 0., -100., 500.);
      auto mu = std::make_unique<RooRealVar>("mu", "Location parameter of normal distribution", 100., 90., 110.);
      auto lambda = std::make_unique<RooRealVar>("lambda", "Two sigma of normal distribution", 20., 10., 30.);
      auto gamma = std::make_unique<RooRealVar>("gamma", "gamma", -0.7, -2., 2.);
      auto delta = std::make_unique<RooRealVar>("delta", "delta", 1.337, 0.9, 2.);

      _pdf = std::make_unique<RooJohnson>("johnson", "johnson", *mass, *mu, *lambda, *gamma, *delta);

      _variables.addOwned(std::move(mass));
      _variables.addOwned(std::move(mu));

      //      _variablesToPlot.add(*mass);

      _parameters.addOwned(std::move(lambda));
      _parameters.addOwned(std::move(gamma));
      _parameters.addOwned(std::move(delta));

      _toleranceCompareBatches = 6.E-12;
      _toleranceCompareLogs = 6.E-12;
      //      _printLevel = 2;
      //      _toleranceParameter = 5.E-5;
      _toleranceCorrelation = 1.5E-3;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestJohnsonInMassAndMu, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestJohnsonInMassAndMu, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestJohnsonInMassAndMu, CompareFixedNormLog)

// Is it clear that the fits can infer the value of lambda when generating in mu?
FIT_TEST_SCALAR(TestJohnsonInMassAndMu, DISABLED_FitScalar)
FIT_TEST_BATCH(TestJohnsonInMassAndMu, DISABLED_FitBatch)
FIT_TEST_BATCH_VS_SCALAR(TestJohnsonInMassAndMu, CompareBatchScalar)

class TestJohnsonWithFormulaParameters : public PDFTest {
protected:
   TestJohnsonWithFormulaParameters() : PDFTest("Johnson with formula")
   {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto mass = std::make_unique<RooRealVar>("mass", "mass", 0., -500., 500.);
      auto mu = std::make_unique<RooRealVar>("mu", "Location parameter of normal distribution", -50, -150., 200.);
      auto lambda = std::make_unique<RooRealVar>("lambda", "Two sigma of normal distribution", 120., 10, 180.);
      auto gamma = std::make_unique<RooRealVar>("gamma", "gamma", -1.8, -10., 10.);
      auto delta = std::make_unique<RooFormulaVar>("delta", "delta", "1.337 + 0.1*gamma", RooArgList(*gamma));

      _pdf = std::make_unique<RooJohnson>("johnson", "johnson", *mass, *mu, *lambda, *gamma, *delta, -1.E300);

      //      _variablesToPlot.add(*mass);

      _variables.addOwned(std::move(mass));

      _parameters.addOwned(std::move(mu));
      _parameters.addOwned(std::move(lambda));
      _parameters.addOwned(std::move(gamma));

      _otherObjects.addOwned(std::move(delta));

      _toleranceCompareBatches = 6.E-12;
      _toleranceCompareLogs = 6.E-12;
      _toleranceParameter = 3.E-5;
      // For i686:
      _toleranceCorrelation = 1.5E-3;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestJohnsonWithFormulaParameters, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestJohnsonWithFormulaParameters, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestJohnsonWithFormulaParameters, CompareFixedNormLog)

FIT_TEST_SCALAR(TestJohnsonWithFormulaParameters, RunScalar)
FIT_TEST_BATCH(TestJohnsonWithFormulaParameters, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestJohnsonWithFormulaParameters, CompareBatchScalar)
