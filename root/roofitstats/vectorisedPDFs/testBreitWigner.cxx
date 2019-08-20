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

class TestBreitWigner : public PDFTest
{
  protected:
    TestBreitWigner() :
      PDFTest("BreitWigner", 300000)
  {
        auto x = new RooRealVar("x", "x", -10, 10);
        auto mean = new RooRealVar("mean", "mean", 1, -7, 7);
        auto width = new RooRealVar("a2", "a2", 1, 0.5, 2.5);

        _pdf = std::make_unique<RooBreitWigner>("breitWigner", "breitWigner PDF", *x, *mean, *width);


      _variables.addOwned(*x);

      _variablesToPlot.add(*x);

      for (auto par : {mean, width}) {
        _parameters.addOwned(*par);
      }
      _toleranceCompareLogs = 4.1774312644789e-13;
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestBreitWigner, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestBreitWigner, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestBreitWigner, CompareFixedNormLog)
FIT_TEST_SCALAR(TestBreitWigner, RunScalar)
FIT_TEST_BATCH(TestBreitWigner, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestBreitWigner, CompareBatchScalar)

