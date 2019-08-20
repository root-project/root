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
#include "RooBifurGauss.h"


class TestBifurGauss : public PDFTest
{
  protected:
    TestBifurGauss() :
      PDFTest("BifurGauss", 300000)
  { 
      auto x = new RooRealVar("x", "x", 300.0, 1.0, 1000.0);
      auto mean = new RooRealVar("mean", "mean", 350.0, 100.0, 900.0);
      auto sigmaL = new RooRealVar("sigmaL", "sigmaL", 10.0, 5.0, 15.0);
      auto sigmaR = new RooRealVar("sigmaR", "sigmaR", 10.0, 5.0, 15.0);
      _pdf = std::make_unique<RooBifurGauss>("bifurGauss1", "bifurGauss1", *x, *mean, *sigmaL, *sigmaR);
      for (auto var : {x}) {
        _variables.addOwned(*var);
      }

      //for (auto var : {x}) {
        //_variablesToPlot.add(*var);
      //}

      for (auto par : {mean, sigmaL, sigmaR}) {
        _parameters.addOwned(*par);
      }
      
      _toleranceCompareBatches = 1e-13;
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestBifurGauss, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestBifurGauss, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestBifurGauss, CompareFixedNormLog)
FIT_TEST_SCALAR(TestBifurGauss, RunScalar)
FIT_TEST_BATCH(TestBifurGauss, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestBifurGauss, CompareBatchScalar)
