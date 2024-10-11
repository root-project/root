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

#include "RooChebychev.h"

class TestChebychev2 : public PDFTest {
protected:
   TestChebychev2() : PDFTest("Chebychev2")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", -10, 10);
      auto a1 = std::make_unique<RooRealVar>("a1", "a1", 0.3, -0.5, 0.5);
      auto a2 = std::make_unique<RooRealVar>("a2", "a2", -0.2, -0.5, 0.5);

      _pdf = std::make_unique<RooChebychev>("chebychev2", "chebychev PDF 2 coefficients", *x, RooArgSet(*a1, *a2));

      //_variablesToPlot.add(*x);

      _variables.addOwned(std::move(x));

      _parameters.addOwned(std::move(a1));
      _parameters.addOwned(std::move(a2));
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestChebychev2, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestChebychev2, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestChebychev2, CompareFixedNormLog)
FIT_TEST_SCALAR(TestChebychev2, DISABLED_RunScalar)
FIT_TEST_BATCH(TestChebychev2, DISABLED_RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestChebychev2, CompareBatchScalar)

class TestChebychev5 : public PDFTest {
protected:
   TestChebychev5() : PDFTest("Chebychev5")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", -10, 10);
      auto a1 = std::make_unique<RooRealVar>("a1", "a1", 0.15, -0.3, 0.3);
      auto a2 = std::make_unique<RooRealVar>("a2", "a2", -0.15, -0.3, 0.3);
      auto a3 = new RooRealVar("a3", "a3", 0.20, 0.10, 0.30);
      auto a4 = std::make_unique<RooRealVar>("a4", "a4", 0.35, 0.3, 0.5);
      auto a5 = std::make_unique<RooRealVar>("a5", "a5", -0.07, -0.2, 0.2);
      a2->setConstant(true);
      a3->setConstant(true);

      _pdf = std::make_unique<RooChebychev>("chebychev5", "chebychev PDF 5 coefficients", *x,
                                            RooArgSet{*a1, *a2, *a3, *a4, *a5});

      //_variablesToPlot.add(*x);

      _variables.addOwned(std::move(x));

      _parameters.addOwned(std::move(a1));
      _parameters.addOwned(std::move(a2));
      _parameters.addOwned(std::move(a4));
      _parameters.addOwned(std::move(a5));

      _toleranceParameter = 2e-6;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestChebychev5, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestChebychev5, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestChebychev5, CompareFixedNormLog)
FIT_TEST_SCALAR(TestChebychev5, RunScalar)
FIT_TEST_BATCH(TestChebychev5, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestChebychev5, CompareBatchScalar)
