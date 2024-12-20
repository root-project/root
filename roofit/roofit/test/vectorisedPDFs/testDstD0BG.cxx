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
#include "RooDstD0BG.h"

class TestDstD0BG : public PDFTest {
protected:
   TestDstD0BG() : PDFTest("DstD0BG")
   {
      auto m = std::make_unique<RooRealVar>("m", "m", 2.0, 1.61, 3);
      auto m0 = new RooRealVar("m0", "m0", 1.6);
      auto C = std::make_unique<RooRealVar>("C", "C", 1, 0.1, 2);
      auto A = new RooRealVar("A", "A", -1.2);
      auto B = new RooRealVar("B", "B", 0.1);

      _pdf = std::make_unique<RooDstD0BG>("DstD0BG", "DstD0BG", *m, *m0, *C, *A, *B);

      _variables.addOwned(std::move(m));

      _parameters.addOwned(std::move(C));

      _toleranceCompareBatches = 3.E-14;
      _toleranceCompareLogs = 3.E-10;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestDstD0BG, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestDstD0BG, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestDstD0BG, CompareFixedNormLog)
FIT_TEST_SCALAR(TestDstD0BG, RunScalar)
FIT_TEST_BATCH(TestDstD0BG, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestDstD0BG, CompareBatchScalar)
