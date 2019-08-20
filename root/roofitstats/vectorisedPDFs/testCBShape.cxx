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

class TestCBShape : public PDFTest
{
  protected:
    TestCBShape() :
      PDFTest("CBShape", 300000)
  {
        auto m = new RooRealVar("m", "m", -10, 10);
        auto m0 = new RooRealVar("m0", "m0", 1, -7, 7);
        auto sigma = new RooRealVar("sigma", "sigma", 1, 0.5, 2.5);
        auto alpha = new RooRealVar("alpha", "alpha", 1, -3, 3);
        auto n = new RooRealVar("n", "n", 1, 0.5, 2.5);
        
        _pdf = std::make_unique<RooCBShape>("CBShape", "CBShape PDF", *m, *m0, *sigma, *alpha, *n);


      _variables.addOwned(*m);

      //_variablesToPlot.add(*m);

      for (auto par : {m0, sigma, alpha, n}) {
        _parameters.addOwned(*par);
      }
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestCBShape, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestCBShape, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestCBShape, CompareFixedNormLog)
FIT_TEST_SCALAR(TestCBShape, RunScalar)
FIT_TEST_BATCH(TestCBShape, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestCBShape, CompareBatchScalar)
