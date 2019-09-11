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


class TestDstD0BG : public PDFTest
{
  protected:
    TestDstD0BG() :
      PDFTest("DstD0BG", 300000)
  { 
      auto m = new RooRealVar("m", "m", 750, 500, 1000);
      auto m0 = new RooRealVar("m0", "m0", 350, 100, 450);
      auto C = new RooRealVar("C", "C", 500, 300, 800);
      auto A = new RooRealVar("A", "A", 1, 0.5, 4);
      auto B = new RooRealVar("B", "B", 1, 0.5, 2);
      
      _pdf = std::make_unique<RooDstD0BG>("DstD0BG", "DstD0BG", *m, *m0, *C, *A, *B);
      m0->setConstant(true);
      //C->setConstant(true);
      
      for (auto var : {m}) {
        _variables.addOwned(*var);
      }

      for (auto var : {m}) {
        _variablesToPlot.add(*var);
      }

      for (auto par : {C, A, B}) {
        _parameters.addOwned(*par);
      }
    _printLevel = 1;
    //_toleranceParameter = 3e-5;

  }
};

COMPARE_FIXED_VALUES_UNNORM(TestDstD0BG, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestDstD0BG, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestDstD0BG, CompareFixedNormLog)
FIT_TEST_SCALAR(TestDstD0BG, RunScalar)
FIT_TEST_BATCH(TestDstD0BG, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestDstD0BG, CompareBatchScalar)
