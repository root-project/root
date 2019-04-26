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

#include "RooProdPdf.h"
#include "RooGaussian.h"

class TestProdPdf : public PDFTest
{
  protected:
    TestProdPdf() :
      PDFTest("Gauss(x) * Gauss(y)", 75000)
  {
      auto x = new RooRealVar("x", "x", 1, -7, 7);
      auto m1 = new RooRealVar("m1", "m1", -0.3 , -5., 5.);
      auto s1 = new RooRealVar("s1", "s1", 1.5, 0.7, 5.);
      auto y = new RooRealVar("y", "y", 1, -5., 5.);
      auto m2 = new RooRealVar("m2", "m2", 0.4, -5., 5.);
      auto s2 = new RooRealVar("s2", "s2", 2., 0.7, 10.);

      //Make a 2D PDF
      auto g1 = new RooGaussian("gaus1", "gaus1", *x, *m1, *s1);
      auto g2 = new RooGaussian("gaus2", "gaus2", *y, *m2, *s2);
      _pdf = std::make_unique<RooProdPdf>("prod", "prod", RooArgSet(*g1, *g2));

      for (auto var : {x, y}) {
        _variables.addOwned(*var);
      }

      for (auto par : {m1, s1, m2, s2}) {
        _parameters.addOwned(*par);
      }

      for (auto obj : {g1, g2}) {
        _otherObjects.addOwned(*obj);
      }

//      _variablesToPlot.add(*x);
//      _variablesToPlot.add(*y);
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestProdPdf, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestProdPdf, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestProdPdf, CompareFixedValuesNormLog)

FIT_TEST_SCALAR(TestProdPdf, FitScalar)
FIT_TEST_BATCH(TestProdPdf, FitBatch)

FIT_TEST_BATCH_VS_SCALAR(TestProdPdf, FitBatchScalar)
FIT_TEST_BATCH_VS_SCALAR_CLONE_PDF(TestProdPdf, FitBatchScalarWithCloning)

