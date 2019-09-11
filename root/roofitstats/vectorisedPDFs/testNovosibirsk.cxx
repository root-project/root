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


class TestNovosibirsk : public PDFTest
{
  protected:
    TestNovosibirsk() :
      PDFTest("Novosibirsk", 300000)
  { 
      auto x = new RooRealVar("x", "x", 0, -5, 2);
      auto peak = new RooRealVar("peak", "peak", 0.5, 0, 1);
      auto width = new RooRealVar("width", "width", 3, 2.5, 3.5);
      auto tail = new RooRealVar("tail", "tail", 0.8, 0.5, 1.1);
      
      _pdf = std::make_unique<RooNovosibirsk>("Novosibirsk", "Novosibirsk", *x, *peak, *width, *tail);
      
      for (auto var : {x}) {
        _variables.addOwned(*var);
      }

      //for (auto var : {x}) {
        //_variablesToPlot.add(*var);
      //}

      for (auto par : { peak, width, tail}) {
        _parameters.addOwned(*par);
      }
      
      _toleranceParameter = 1e-4;
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestNovosibirsk, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestNovosibirsk, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestNovosibirsk, CompareFixedNormLog)
FIT_TEST_SCALAR(TestNovosibirsk, RunScalar)
FIT_TEST_BATCH(TestNovosibirsk, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestNovosibirsk, CompareBatchScalar)
