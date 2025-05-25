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

class TestBifurGauss : public PDFTest {
protected:
   TestBifurGauss() : PDFTest("BifurGauss")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", 300.0, 100.0, 800.0);
      auto mean = std::make_unique<RooRealVar>("mean", "mean", 350.0, 250.0, 500.0);
      mean->setConstant();
      auto sigmaL = std::make_unique<RooRealVar>("sigmaL", "sigmaL", 60.0, 50.0, 150.0);
      auto sigmaR = std::make_unique<RooRealVar>("sigmaR", "sigmaR", 100.0, 50.0, 150.0);
      _pdf = std::make_unique<RooBifurGauss>("bifurGauss1", "bifurGauss1", *x, *mean, *sigmaL, *sigmaR);

      //    for (auto var : {x}) {
      //      _variablesToPlot.add(*var);
      //    }

      _variables.addOwned(std::move(x));

      _parameters.addOwned(std::move(mean));
      _parameters.addOwned(std::move(sigmaL));
      _parameters.addOwned(std::move(sigmaR));

      _toleranceParameter = 8.E-6;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestBifurGauss, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestBifurGauss, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestBifurGauss, CompareFixedNormLog)
FIT_TEST_SCALAR(TestBifurGauss, RunScalar)
FIT_TEST_BATCH(TestBifurGauss, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestBifurGauss, CompareBatchScalar)
