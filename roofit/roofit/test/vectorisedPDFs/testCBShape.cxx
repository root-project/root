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
#include "RooCBShape.h"

class TestCBShape : public PDFTest {
protected:
   TestCBShape() : PDFTest("CBShape")
   {
      auto m = std::make_unique<RooRealVar>("m", "m", -10, 10);
      auto m0 = std::make_unique<RooRealVar>("m0", "m0", 1, -7, 7);
      auto sigma = std::make_unique<RooRealVar>("sigma", "sigma", 1, 0.5, 2.5);
      auto alpha = std::make_unique<RooRealVar>("alpha", "alpha", 1, -3, 3);
      auto n = std::make_unique<RooRealVar>("n", "n", 1, 0.5, 2.5);

      _pdf = std::make_unique<RooCBShape>("CBShape", "CBShape PDF", *m, *m0, *sigma, *alpha, *n);

      //_variablesToPlot.add(*m);

      _variables.addOwned(std::move(m));

      _parameters.addOwned(std::move(m0));
      _parameters.addOwned(std::move(sigma));
      _parameters.addOwned(std::move(alpha));
      _parameters.addOwned(std::move(n));

      // For i686:
      _toleranceParameter = 1.E-5;
      _toleranceCompareBatches = 1.5E-14;
      _toleranceCorrelation = 1.5E-3;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestCBShape, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestCBShape, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestCBShape, CompareFixedNormLog)
FIT_TEST_SCALAR(TestCBShape, RunScalar)
FIT_TEST_BATCH(TestCBShape, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestCBShape, CompareBatchScalar)
