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

class TestGauss : public PDFTest
{
  protected:
    TestGauss() :
      PDFTest("Gauss", 200000)
  {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
        auto x = new RooRealVar("x", "x", -10, 10);
        auto mean = new RooRealVar("mean", "mean of gaussian", 1, -10, 10);
        auto sigma = new RooRealVar("sigma", "width of gaussian", 1, 0.1, 10);

        // Build gaussian p.d.f in terms of x,mean and sigma
        _pdf = std::make_unique<RooGaussian>("gauss", "gaussian PDF", *x, *mean, *sigma);


      _variables.addOwned(*x);

//      _variablesToPlot.add(x);

      for (auto par : {mean, sigma}) {
        _parameters.addOwned(*par);
      }

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



class TestGaussWeighted : public PDFTestWeightedData
{
  protected:
    TestGaussWeighted() :
      PDFTestWeightedData("GaussWithWeights", 300000)
  {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto x = new RooRealVar("x", "x", -10, 10);
      auto mean = new RooRealVar("mean", "mean of gaussian", 1, -10, 10);
      auto sigma = new RooRealVar("sigma", "width of gaussian", 1, 0.1, 10);

      // Build gaussian p.d.f in terms of x,mean and sigma
      _pdf = std::make_unique<RooGaussian>("gauss", "gaussian PDF", *x, *mean, *sigma);


      _variables.addOwned(*x);

      for (auto par : {mean, sigma}) {
        _parameters.addOwned(*par);
      }
  }
};

FIT_TEST_BATCH(TestGaussWeighted, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestGaussWeighted, CompareBatchScalar)



class TestGaussInMeanAndX : public PDFTest
{
  protected:
    TestGaussInMeanAndX() :
      PDFTest("Gauss(x, mean)")
  {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto x = new RooRealVar("x", "x", -10, 10);
      auto mean = new RooRealVar("mean", "mean of gaussian", 1, -10, 10);
      auto sigma = new RooRealVar("sigma", "width of gaussian", 1, 0.1, 10);

      // Build gaussian p.d.f in terms of x,mean and sigma
      _pdf = std::make_unique<RooGaussian>("gauss", "gaussian PDF", *x, *mean, *sigma);


      for (auto var : {x, mean}) {
        _variables.addOwned(*var);
//        _variablesToPlot.add(var);
      }

      for (auto par : {sigma}) {
        _parameters.addOwned(*par);
      }
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestGaussInMeanAndX, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestGaussInMeanAndX, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestGaussInMeanAndX, CompareFixedNormLog)

FIT_TEST_BATCH(TestGaussInMeanAndX, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestGaussInMeanAndX, CompareBatchScalar)


class TestGaussWithFormulaParameters : public PDFTest
{
  protected:
    TestGaussWithFormulaParameters() :
      PDFTest("Gauss(x, mean)", 50000)
  {
      // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
      auto x = new RooRealVar("x", "x", 0, 30);
      auto a1 = new RooRealVar("a1", "First coefficient", 5, 0, 10);
      auto a2 = new RooRealVar("a2", "Second coefficient", 1, 0, 10);
      auto mean = new RooFormulaVar("mean", "mean", "a1+a2", RooArgList(*a1, *a2));
      auto sigma = new RooFormulaVar("sigma", "sigma", "1.7*mean", RooArgList(*mean));

      // Build gaussian p.d.f in terms of x,mean and sigma
      _pdf = std::make_unique<RooGaussian>("gauss", "gaussian PDF", *x, *mean, *sigma);


      for (auto var : {x, a1}) {
        _variables.addOwned(*var);
//        _variablesToPlot.add(*var);
      }

      for (auto par : {a2}) {
        _parameters.addOwned(*par);
      }

      _otherObjects.addOwned(*mean);
      _otherObjects.addOwned(*sigma);

      _toleranceCompareBatches = 2.E-14;
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestGaussWithFormulaParameters, FixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestGaussWithFormulaParameters, FixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestGaussWithFormulaParameters, FixedValuesNormLog)

FIT_TEST_SCALAR(TestGaussWithFormulaParameters, RunScalar)
FIT_TEST_BATCH(TestGaussWithFormulaParameters, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestGaussWithFormulaParameters, CompareBatchScalar)
