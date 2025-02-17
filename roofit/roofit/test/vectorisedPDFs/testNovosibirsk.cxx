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
#include "RooNovosibirsk.h"

class TestNovosibirsk : public PDFTest {
protected:
   TestNovosibirsk() : PDFTest("Novosibirsk")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", 0, -5, 1.1);
      auto peak = std::make_unique<RooRealVar>("peak", "peak", 0.5, 0, 1);
      auto width = std::make_unique<RooRealVar>("width", "width", 1.1, 0.5, 3.);
      auto tail = std::make_unique<RooRealVar>("tail", "tail", 1.0, 0.5, 1.1);

      _pdf = std::make_unique<RooNovosibirsk>("Novosibirsk", "Novosibirsk", *x, *peak, *width, *tail);

      //      for (auto var : {x}) {
      //        _variablesToPlot.add(*var);
      //      }

      _variables.addOwned(std::move(x));

      _parameters.addOwned(std::move(peak));
      _parameters.addOwned(std::move(width));
      _parameters.addOwned(std::move(tail));

      _toleranceParameter = 1e-4;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestNovosibirsk, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestNovosibirsk, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestNovosibirsk, CompareFixedNormLog)
FIT_TEST_SCALAR(TestNovosibirsk, RunScalar)
FIT_TEST_BATCH(TestNovosibirsk, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestNovosibirsk, CompareBatchScalar)
