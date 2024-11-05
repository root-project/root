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

#include "RooGaussian.h"
#include "RooFormulaVar.h"

class TestGauss : public PDFTest {
protected:
   TestGauss() : PDFTest("Gauss")
   {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto x = std::make_unique<RooRealVar>("x", "x", -10, 10);
      auto mean = std::make_unique<RooRealVar>("mean", "mean of gaussian", 1, -10, 10);
      auto sigma = std::make_unique<RooRealVar>("sigma", "width of gaussian", 1, 0.1, 10);

      // Build gaussian p.d.f in terms of x,mean and sigma
      _pdf = std::make_unique<RooGaussian>("gauss", "gaussian PDF", *x, *mean, *sigma);

      //      _variablesToPlot.add(x);

      _variables.addOwned(std::move(x));

      _parameters.addOwned(std::move(mean));
      _parameters.addOwned(std::move(sigma));

      // Standard of 1.E-14 is slightly too strong.
      _toleranceCompareBatches = 3.E-14;
      _toleranceCompareLogs = 3.E-14;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestGauss, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestGauss, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestGauss, CompareFixedNormLog)

FIT_TEST_SCALAR(TestGauss, RunScalar)
FIT_TEST_BATCH(TestGauss, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestGauss, CompareBatchScalar)

#if !defined(_MSC_VER) // TODO: make TestGaussWeighted work on Windows

class TestGaussWeighted : public PDFTestWeightedData {
protected:
   TestGaussWeighted() : PDFTestWeightedData("GaussWithWeights")
   {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto x = std::make_unique<RooRealVar>("x", "x", -10, 10);
      auto mean = std::make_unique<RooRealVar>("mean", "mean of gaussian", 1, -10, 10);
      auto sigma = std::make_unique<RooRealVar>("sigma", "width of gaussian", 1, 0.1, 10);

      // Build gaussian p.d.f in terms of x,mean and sigma
      _pdf = std::make_unique<RooGaussian>("gauss", "gaussian PDF", *x, *mean, *sigma);

      _variables.addOwned(std::move(x));

      _parameters.addOwned(std::move(mean));
      _parameters.addOwned(std::move(sigma));

      _multiProcess = 4;
   }
};

FIT_TEST_BATCH(TestGaussWeighted,
               DISABLED_RunBatch) // Would need SumW2 or asymptotic error correction, but that's not in test macro.
FIT_TEST_BATCH_VS_SCALAR(TestGaussWeighted, CompareBatchScalar)

#endif // !defined(_MSC_VER)

class TestGaussInMeanAndX : public PDFTest {
protected:
   TestGaussInMeanAndX() : PDFTest("Gauss(x, mean)")
   {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto x = std::make_unique<RooRealVar>("x", "x", -10, 10);
      auto mean = std::make_unique<RooRealVar>("mean", "mean of gaussian", 1, -10, 10);
      auto sigma = std::make_unique<RooRealVar>("sigma", "width of gaussian", 1, 0.1, 10);

      // Build gaussian p.d.f in terms of x,mean and sigma
      _pdf = std::make_unique<RooGaussian>("gauss", "gaussian PDF", *x, *mean, *sigma);

      //        _variablesToPlot.add(var);

      _variables.addOwned(std::move(x));
      _variables.addOwned(std::move(mean));

      _parameters.addOwned(std::move(sigma));
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestGaussInMeanAndX, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestGaussInMeanAndX, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestGaussInMeanAndX, CompareFixedNormLog)

FIT_TEST_BATCH(TestGaussInMeanAndX, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestGaussInMeanAndX, CompareBatchScalar)

class TestGaussWithFormulaParameters : public PDFTest {
protected:
   TestGaussWithFormulaParameters() : PDFTest("Gauss(x, mean)")
   {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto x = std::make_unique<RooRealVar>("x", "x", 0, 30);
      auto a1 = std::make_unique<RooRealVar>("a1", "First coefficient", 5, 0, 10);
      auto a2 = std::make_unique<RooRealVar>("a2", "Second coefficient", 1, 0, 10);
      auto mean = std::make_unique<RooFormulaVar>("mean", "mean", "a1+a2", RooArgList(*a1, *a2));
      auto sigma = std::make_unique<RooFormulaVar>("sigma", "sigma", "1.7*mean", RooArgList(*mean));

      // Build gaussian p.d.f in terms of x,mean and sigma
      _pdf = std::make_unique<RooGaussian>("gauss", "gaussian PDF", *x, *mean, *sigma);

      //        _variablesToPlot.add(*var);

      _variables.addOwned(std::move(x));
      _variables.addOwned(std::move(a1));

      _parameters.addOwned(std::move(a2));

      _otherObjects.addOwned(std::move(mean));
      _otherObjects.addOwned(std::move(sigma));

      _toleranceCompareBatches = 2.E-14;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestGaussWithFormulaParameters, FixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestGaussWithFormulaParameters, FixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestGaussWithFormulaParameters, FixedValuesNormLog)

FIT_TEST_SCALAR(TestGaussWithFormulaParameters, RunScalar)
FIT_TEST_BATCH(TestGaussWithFormulaParameters, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestGaussWithFormulaParameters, CompareBatchScalar)
