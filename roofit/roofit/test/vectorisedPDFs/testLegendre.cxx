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

#include "RooLegendre.h"
#include "RooRealSumPdf.h"
#include "RooUniform.h"
class TestLegendre : public PDFTest {
protected:
   TestLegendre() : PDFTest("Legendre")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", 0.5, 0.1, 1.0);
      auto coef = new RooRealVar("coef", "coef", 0.5, 0.3, 1.0);
      auto coef2 = new RooRealVar("coef2", "coef2", 100, 80, 120);

      auto uniform = new RooUniform("uniform", "uniform", *x);

      auto legendre = new RooLegendre("Legendre", "Legendre PDF", *x, 3, 2, 2, 1);
      _pdf = std::make_unique<RooRealSumPdf>("Sum", "sum", RooArgList{*legendre, *uniform}, RooArgList{*coef, *coef2},
                                             true);

      _variablesToPlot.add(*x);

      _variables.addOwned(std::move(x));
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestLegendre, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestLegendre, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestLegendre, CompareFixedNormLog)
