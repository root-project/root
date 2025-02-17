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

#include "RooGamma.h"

class TestGamma : public PDFTest {
protected:
   TestGamma() : PDFTest("Gamma")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", 5, 4, 10);
      auto gamma = std::make_unique<RooRealVar>("gamma", "N+1", 6, 4, 8);
      gamma->setConstant();
      auto beta = std::make_unique<RooRealVar>("beta", "beta", 1.5, 0.5, 10);
      auto mu = std::make_unique<RooRealVar>("mu", "mu", 0.2, -1., 1.);
      mu->setConstant();

      // Build gaussian p.d.f in terms of x,mean and sigma
      _pdf = std::make_unique<RooGamma>("Gamma", "Gamma PDF", *x, *gamma, *beta, *mu);

      _variables.addOwned(std::move(x));

      _parameters.addOwned(std::move(gamma));
      _parameters.addOwned(std::move(beta));
      _parameters.addOwned(std::move(mu));

      //    _variablesToPlot.add(*x);
      _toleranceCompareBatches = 1.2E-14;
      _toleranceCompareLogs = 6.E-12;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestGamma, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestGamma, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestGamma, CompareFixedNormLog)
FIT_TEST_SCALAR(TestGamma, RunScalar)
FIT_TEST_BATCH(TestGamma, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestGamma, CompareBatchScalar)
