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
#include "RooChiSquarePdf.h"

class TestChiSquarePdfinX : public PDFTest {
protected:
   TestChiSquarePdfinX() : PDFTest("ChiSquarePdf")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", 0.1, 100);
      auto ndof = std::make_unique<RooRealVar>("ndof", "ndof of chiSquarePdf", 2, 1, 5);

      // Build chiSquarePdf p.d.f
      _pdf = std::make_unique<RooChiSquarePdf>("chiSquarePdf", "chiSquarePdf PDF", *x, *ndof);

      //      _variablesToPlot.add(x);

      _variables.addOwned(std::move(x));

      _parameters.addOwned(std::move(ndof));

      _toleranceCompareLogs = 5e-14;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestChiSquarePdfinX, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestChiSquarePdfinX, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestChiSquarePdfinX, CompareFixedNormLog)
FIT_TEST_SCALAR(TestChiSquarePdfinX, RunScalar)
FIT_TEST_BATCH(TestChiSquarePdfinX, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestChiSquarePdfinX, CompareBatchScalar)
