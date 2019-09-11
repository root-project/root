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
#include "RooVoigtian.h"

class TestVoigtian : public PDFTest
{
  protected:
    TestVoigtian() :
      PDFTest("Voigtian", 300000)
  {
        auto x = new RooRealVar("x", "x", 1, 0.1, 10);
        auto mean = new RooRealVar("mean", "mean", 1, 0.1, 10);
        auto width = new RooRealVar("width", "width", 0.5, 0.1, 0.9);
        auto sigma = new RooRealVar("sigma", "sigma", 0.5, 0.1, 0.9);

        _pdf = std::make_unique<RooVoigtian>("Voigtian", "Voigtian PDF", *x, *mean, *width, *sigma);


      _variables.addOwned(*x);

      //_variablesToPlot.add(*x);

      for (auto par : {mean, width, sigma}) {
        _parameters.addOwned(*par);
      }
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestVoigtian, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestVoigtian, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestVoigtian, CompareFixedNormLog)
FIT_TEST_SCALAR(TestVoigtian, RunScalar)
FIT_TEST_BATCH(TestVoigtian, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestVoigtian, CompareBatchScalar)

class TestVoigtianInXandMean : public PDFTest
{
  protected:
    TestVoigtianInXandMean() :
      PDFTest("Voigtian(x,m)", 300000)
  {
        auto x = new RooRealVar("x", "x", 1, 0.1, 10);
        auto mean = new RooRealVar("mean", "mean", 1, 0.1, 10);
        auto width = new RooRealVar("width", "width", 0.5, 0.1, 0.9);
        auto sigma = new RooRealVar("sigma", "sigma", 0.5, 0.1, 0.9);

        _pdf = std::make_unique<RooVoigtian>("Voigtian", "Voigtian PDF", *x, *mean, *width, *sigma);


      for (auto var: {x,mean} ) {
        _variables.addOwned(*var);
      }

      //_variablesToPlot.add(*x);

      for (auto par : {width, sigma}) {
        _parameters.addOwned(*par);
      }

  }
};

COMPARE_FIXED_VALUES_UNNORM(TestVoigtianInXandMean, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestVoigtianInXandMean, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestVoigtianInXandMean, CompareFixedNormLog)
FIT_TEST_SCALAR(TestVoigtianInXandMean, RunScalar)
FIT_TEST_BATCH(TestVoigtianInXandMean, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestVoigtianInXandMean, CompareBatchScalar)
