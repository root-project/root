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
#include "RooBreitWigner.h"

class TestBreitWigner : public PDFTest {
protected:
   TestBreitWigner() : PDFTest("BreitWigner")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", -10, 10);
      auto mean = std::make_unique<RooRealVar>("mean", "mean", 1, -7, 7);
      auto width = std::make_unique<RooRealVar>("a2", "a2", 1.8, 0.5, 2.5);

      _pdf = std::make_unique<RooBreitWigner>("breitWigner", "breitWigner PDF", *x, *mean, *width);

      //      _variablesToPlot.add(*x);

      _variables.addOwned(std::move(x));

      _parameters.addOwned(std::move(mean));
      _parameters.addOwned(std::move(width));

      _toleranceCompareLogs = 5.e-13;
      _toleranceCorrelation = 0.007;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestBreitWigner, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestBreitWigner, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestBreitWigner, CompareFixedNormLog)
FIT_TEST_SCALAR(TestBreitWigner, RunScalar)
FIT_TEST_BATCH(TestBreitWigner, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestBreitWigner, CompareBatchScalar)
