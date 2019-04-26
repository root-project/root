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

#include "RooPoisson.h"


class TestPoisson : public PDFTest
{
  protected:
    TestPoisson() :
      PDFTest("Poisson", 100000)
  {
      auto x = new RooRealVar("x", "x", -10, 100);
      auto mean = new RooRealVar("mean", "Mean of Poisson", 2., 0., 50);
      _pdf = std::make_unique<RooPoisson>("Pois", "Poisson PDF", *x, *mean);

      _variables.addOwned(*x);

      for (auto par : {mean}) {
        _parameters.addOwned(*par);
      }

  }
};

FIT_TEST_BATCH(TestPoisson, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestPoisson, CompareBatchScalar)


class TestPoissonOddMean : public PDFTest
{
  protected:
    TestPoissonOddMean() :
      PDFTest("PoissonOddMean", 100000)
  {
      auto x = new RooRealVar("x", "x", -10, 50);
      auto mean = new RooRealVar("mean", "Mean of Poisson", 7.5, 0., 50);
      _pdf = std::make_unique<RooPoisson>("Pois", "Poisson PDF", *x, *mean);

      _variables.addOwned(*x);

      for (auto par : {mean}) {
        _parameters.addOwned(*par);
      }

  }
};

FIT_TEST_BATCH(TestPoissonOddMean, DISABLED_RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestPoissonOddMean, CompareBatchScalar)


class TestPoissonOddMeanNoRounding : public PDFTest
{
  protected:
    TestPoissonOddMeanNoRounding() :
      PDFTest("PoissonOddMeanNoRounding", 1000)
  {
      auto x = new RooRealVar("x", "x", 0., 100);
      auto mean = new RooRealVar("mean", "Mean of Poisson", 7.8529298854862928, 0., 10);
      _pdf = std::make_unique<RooPoisson>("Pois", "Poisson PDF", *x, *mean, true);

      _variables.addOwned(*x);

      for (auto par : {mean}) {
        _parameters.addOwned(*par);
      }

      _toleranceParameter = 1.2E-5;
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestPoissonOddMeanNoRounding, CompareFixedValuesUnnorm);
COMPARE_FIXED_VALUES_NORM(TestPoissonOddMeanNoRounding, CompareFixedValuesNorm);
COMPARE_FIXED_VALUES_NORM_LOG(TestPoissonOddMeanNoRounding, CompareFixedValuesNormLog);

// Fit tests have a small bias. Unclear why.
FIT_TEST_SCALAR(TestPoissonOddMeanNoRounding, DISABLED_RunScalar)
FIT_TEST_BATCH(TestPoissonOddMeanNoRounding, DISABLED_RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestPoissonOddMeanNoRounding, CompareBatchScalar)
