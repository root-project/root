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

class TestGamma : public PDFTest
{
  protected:
    TestGamma() :
      PDFTest("Gamma", 300000)
  {
    // Declare variables x,mean,sigma with associated name, title, initial value and allowed range
    auto x = new RooRealVar("x", "x", 5, 4, 10);
    auto gamma = new RooRealVar("m0", "m0", 1.5, 0.1, 7);
    auto beta = new RooRealVar("sigma", "sigma", 1.5, 0.1, 10);
    auto mu = new RooRealVar("alpha", "alpha", 2, 1, 4);
    
    // Build gaussian p.d.f in terms of x,mean and sigma
    _pdf = std::make_unique<RooGamma>("Gamma", "Gamma PDF", *x, *gamma, *beta, *mu);
    
    for (auto var : {x}) {
    _variables.addOwned(*var);      
    }

    for (auto par : {gamma, beta, mu}) {
      _parameters.addOwned(*par);
    }
    
    _variablesToPlot.add(*x);
    _toleranceCompareLogs = 1.105735107124577e-11;

  }
};

COMPARE_FIXED_VALUES_UNNORM(TestGamma, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestGamma, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestGamma, CompareFixedNormLog)
FIT_TEST_SCALAR(TestGamma, RunScalar)
FIT_TEST_BATCH(TestGamma, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestGamma, CompareBatchScalar)

