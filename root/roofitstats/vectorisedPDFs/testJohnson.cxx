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

class TestJohnson : public PDFTest
{
  protected:
    TestJohnson() :
      PDFTest("Johnson", 200000)
  {
      auto mass = new RooRealVar("mass", "mass", 0., 0., 500.);
      auto mu = new RooRealVar("mu", "Location parameter of normal distribution", 300., 0., 500.);
      auto lambda = new RooRealVar ("lambda", "Two sigma of normal distribution", 100., 10, 200.);
      auto gamma = new RooRealVar ("gamma", "gamma", 0.5, -10., 10.);
      auto delta = new RooRealVar ("delta", "delta", 2., 1., 10.);
      // delta is highly correlated with lambda. This troubles the fit.
      delta->setConstant();

      _pdf = std::make_unique<RooJohnson>("johnson", "johnson", *mass, *mu, *lambda, *gamma, *delta, -1.E300);


      _variables.addOwned(*mass);

//      _variablesToPlot.add(*mass);

      for (auto par : {mu, lambda, gamma, delta}) {
        _parameters.addOwned(*par);
      }

      _toleranceCompareBatches = 5.E-13;
      _toleranceCompareLogs = 6.E-13;
//      _toleranceParameter = 1.E-5;
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestJohnson, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestJohnson, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestJohnson, CompareFixedNormLog)

FIT_TEST_SCALAR(TestJohnson, FitScalar)
FIT_TEST_BATCH(TestJohnson, FitBatch)
FIT_TEST_BATCH_VS_SCALAR(TestJohnson, FitBatchVsScalar)



class TestJohnsonInMassAndMu : public PDFTest
{
  protected:
    TestJohnsonInMassAndMu() :
      PDFTest("Johnson in mass and mu", 200000)
  {
      auto mass = new RooRealVar("mass", "mass", 0., -100., 500.);
      auto mu = new RooRealVar("mu", "Location parameter of normal distribution", 100., 90., 110.);
      auto lambda = new RooRealVar ("lambda", "Two sigma of normal distribution", 20., 10., 30.);
      auto gamma = new RooRealVar ("gamma", "gamma", -0.7, -2., 2.);
      auto delta = new RooRealVar ("delta", "delta", 1.337, 0.9, 2.);

      _pdf = std::make_unique<RooJohnson>("johnson", "johnson", *mass, *mu, *lambda, *gamma, *delta);


      _variables.addOwned(*mass);
      _variables.addOwned(*mu);

//      _variablesToPlot.add(*mass);

      for (auto par : {gamma, lambda, delta}) {
        _parameters.addOwned(*par);
      }

      _toleranceCompareBatches = 7.E-13;
      _toleranceCompareLogs = 1.E-13;
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



class TestJohnsonWithFormulaParameters : public PDFTest
{
  protected:
    TestJohnsonWithFormulaParameters() :
      PDFTest("Johnson with formula", 100000)
  {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto mass = new RooRealVar("mass", "mass", 0., -500., 500.);
      auto mu = new RooRealVar("mu", "Location parameter of normal distribution", -50, -150., 200.);
      auto lambda = new RooRealVar ("lambda", "Two sigma of normal distribution", 120., 10, 180.);
      auto gamma = new RooRealVar ("gamma", "gamma", -1.8, -10., 10.);
      auto delta = new RooFormulaVar("delta", "delta", "1.337 + 0.1*gamma", RooArgList(*gamma));

      _pdf = std::make_unique<RooJohnson>("johnson", "johnson", *mass, *mu, *lambda, *gamma, *delta, -1.E300);

//      _variablesToPlot.add(*mass);

      for (auto var : {mass}) {
        _variables.addOwned(*var);
      }

      for (auto par : {mu, lambda, gamma}) {
        _parameters.addOwned(*par);
      }

      _otherObjects.addOwned(*delta);

      _toleranceCompareBatches = 2.E-12;
      _toleranceCompareLogs = 5.E-14;
      _toleranceParameter = 3.E-5;
      _toleranceCorrelation = 1.E-3;
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestJohnsonWithFormulaParameters, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestJohnsonWithFormulaParameters, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestJohnsonWithFormulaParameters, CompareFixedNormLog)

FIT_TEST_SCALAR(TestJohnsonWithFormulaParameters, RunScalar)
FIT_TEST_BATCH(TestJohnsonWithFormulaParameters, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestJohnsonWithFormulaParameters, CompareBatchScalar)
