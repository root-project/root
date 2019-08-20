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
#include "RooPolynomial.h"

class TestPolynomial2 : public PDFTest
{
  protected:
    TestPolynomial2() :
      PDFTest("Polynomial2", 300000)
  {
        auto x = new RooRealVar("x", "x", -10, 10);
        auto a1 = new RooRealVar("a1", "a1", 0.3, 0.01, 0.5);
        auto a2 = new RooRealVar("a2", "a2", 0.2, 0.01, 0.5);

        _pdf = std::make_unique<RooPolynomial>("polynomial2", "polynomial PDF 2 coefficients", *x, RooArgSet(*a1,*a2));


      _variables.addOwned(*x);

      //_variablesToPlot.add(*x);

      for (auto par : {a1, a2}) {
        _parameters.addOwned(*par);
      }
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestPolynomial2, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestPolynomial2, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestPolynomial2, CompareFixedNormLog)
FIT_TEST_SCALAR(TestPolynomial2, RunScalar)
FIT_TEST_BATCH(TestPolynomial2, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestPolynomial2, CompareBatchScalar)

class TestPolynomial5 : public PDFTest
{
  protected:
    TestPolynomial5() :
      PDFTest("Polynomial5", 300000)
  {
        auto x = new RooRealVar("x", "x", -100, 40);
        auto a0 = new RooRealVar("a0", "a0", 1000.0);
        auto a1 = new RooRealVar("a1", "a1", 1.0, 0.0, 3.0);
        auto a2 = new RooRealVar("a2", "a2", 10.0, 9.0, 12.0);
        auto a3 = new RooRealVar("a3", "a3", 0.09, 0.05, 0.1);
        auto a4 = new RooRealVar("a4", "a4", 0.00009, 0.00005, 0.0005);
        auto a5 = new RooRealVar("a5", "a5", 0.0000009, 0.0000005, 0.000005);
        a0->setConstant(true);
        a1->setConstant(true);
        a3->setConstant(true);

        _pdf = std::make_unique<RooPolynomial>("polynomial5", "polynomial PDF 5 coefficients", *x, RooArgSet(*a0,*a1, *a2, *a3, *a4, *a5),0);


      _variables.addOwned(*x);

      //_variablesToPlot.add(*x);

      for (auto par : { a2, a4, a5}) {
        _parameters.addOwned(*par);
      }
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestPolynomial5, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestPolynomial5, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestPolynomial5, CompareFixedNormLog)
FIT_TEST_SCALAR(TestPolynomial5, RunScalar)
FIT_TEST_BATCH(TestPolynomial5, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestPolynomial5, CompareBatchScalar)

