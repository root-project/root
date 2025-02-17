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
#include "RooAddPdf.h"
#include "RooExponential.h"

class TestExponential : public PDFTest {
protected:
   TestExponential() : PDFTest("Exp(x, c1)")
   {
      // Beyond ~19, the VDT polynomials break down when c1 is very negative
      auto x = std::make_unique<RooRealVar>("x", "x", 0.001, 18.);
      auto c1 = std::make_unique<RooRealVar>("c1", "c1", -0.2, -50., -0.001);
      _pdf = std::make_unique<RooExponential>("expo1", "expo1", *x, *c1);

      _variables.addOwned(std::move(x));

      //      for (auto var : {x}) {
      //        _variablesToPlot.add(*var);
      //      }

      _parameters.addOwned(std::move(c1));

      _toleranceCompareLogs = 3E-13;
      // For i686, this needs to be a bit less strict:
      _toleranceCompareBatches = 2.E-14;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestExponential, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestExponential, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestExponential, CompareFixedValuesNormLog)
FIT_TEST_SCALAR(TestExponential, RunScalar)
FIT_TEST_BATCH(TestExponential, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestExponential, CompareBatchScalar)
