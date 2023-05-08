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

#include "RooLognormal.h"

class TestLognormal : public PDFTest
{
  protected:
    TestLognormal() :
      PDFTest("Lognormal", 100000)
  {
        auto x = new RooRealVar("x", "x", 1, 0.1, 10);
        auto m0 = new RooRealVar("m0", "m0", 5, 0.1, 10);
        auto k = new RooRealVar("k", "k", 0.6, 0.1, 0.9);

        _pdf = std::make_unique<RooLognormal>("Lognormal", "Lognormal PDF", *x, *m0, *k);


      _variables.addOwned(*x);

      for (auto par : {m0, k}) {
        _parameters.addOwned(*par);
      }

       //Standard of 1.E-14 is too strong.
      _toleranceCompareBatches = 8.E-14;
      _toleranceParameter = 2e-6;
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestLognormal, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestLognormal, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestLognormal, CompareFixedNormLog)
FIT_TEST_SCALAR(TestLognormal, RunScalar)
FIT_TEST_BATCH(TestLognormal, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestLognormal, CompareBatchScalar)

class TestLognormalInMeanAndX : public PDFTest
{
  protected:
    TestLognormalInMeanAndX() :
      PDFTest("Lognormal(x, mean)", 100000)
  {
        auto x = new RooRealVar("x", "x", 1, 0.1, 10);
        auto m0 = new RooRealVar("m0", "m0", 1, 0.1, 10);
        auto k = new RooRealVar("k", "k", 2, 1.1, 10);

        _pdf = std::make_unique<RooLognormal>("Lognormal", "Lognormal PDF", *x, *m0, *k);


      for (auto var : {x, m0}) {
        _variables.addOwned(*var);
      }
      //_variablesToPlot.add(*x);

      for (auto par : {k}) {
        _parameters.addOwned(*par);
      }

      // Standard of 1.E-14 is slightly too strong.
      _toleranceCompareBatches = 2.E-14;
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestLognormalInMeanAndX, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestLognormalInMeanAndX, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestLognormalInMeanAndX, CompareFixedNormLog)
FIT_TEST_SCALAR(TestLognormalInMeanAndX, RunScalar)
FIT_TEST_BATCH(TestLognormalInMeanAndX, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestLognormalInMeanAndX, CompareBatchScalar)
