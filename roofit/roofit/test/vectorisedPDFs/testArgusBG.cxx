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
#include "RooArgusBG.h"

class TestArgus : public PDFTest {
protected:
   TestArgus() : PDFTest("Argus")
   {
      auto m = std::make_unique<RooRealVar>("m", "m", 300.0, 1.0, 800.0);
      auto m0 = std::make_unique<RooRealVar>("m0", "m0", 1100.0, 800.0, 1400.0);
      auto c = std::make_unique<RooRealVar>("c", "c", 10.0, 5.0, 15.0);
      c->setConstant();
      auto p = std::make_unique<RooRealVar>("p", "p", 1.0, 0.9, 1.3);
      p->setConstant();
      _pdf = std::make_unique<RooArgusBG>("argus1", "argus1", *m, *m0, *c, *p);
      //      for (auto var : {m}) {
      //        _variablesToPlot.add(*var);
      //      }
      //      _printLevel = 2;

      _variables.addOwned(std::move(m));

      _parameters.addOwned(std::move(m0));
      _parameters.addOwned(std::move(c));
      _parameters.addOwned(std::move(p));

      _toleranceParameter = 2.E-6;
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestArgus, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestArgus, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestArgus, CompareFixedNormLog)
FIT_TEST_SCALAR(TestArgus, RunScalar)
FIT_TEST_BATCH(TestArgus, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestArgus, CompareBatchScalar)
