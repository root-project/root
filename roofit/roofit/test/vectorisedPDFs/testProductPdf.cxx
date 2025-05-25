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

class TestProdPdf : public PDFTest {
protected:
   TestProdPdf() : PDFTest("Gauss(x) * Gauss(y)")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", 1, -7, 7);
      auto m1 = std::make_unique<RooRealVar>("m1", "m1", -0.3, -5., 5.);
      auto s1 = std::make_unique<RooRealVar>("s1", "s1", 1.5, 0.7, 5.);
      auto y = std::make_unique<RooRealVar>("y", "y", 1, -5., 5.);
      auto m2 = std::make_unique<RooRealVar>("m2", "m2", 0.4, -5., 5.);
      auto s2 = std::make_unique<RooRealVar>("s2", "s2", 2., 0.7, 10.);

      // Make a 2D PDF
      auto g1 = std::make_unique<RooGaussian>("gaus1", "gaus1", *x, *m1, *s1);
      auto g2 = std::make_unique<RooGaussian>("gaus2", "gaus2", *y, *m2, *s2);
      _pdf = std::make_unique<RooProdPdf>("prod", "prod", RooArgSet(*g1, *g2));

      _variables.addOwned(std::move(x));
      _variables.addOwned(std::move(y));

      _parameters.addOwned(std::move(m1));
      _parameters.addOwned(std::move(s1));
      _parameters.addOwned(std::move(m2));
      _parameters.addOwned(std::move(s2));

      _otherObjects.addOwned(std::move(g1));
      _otherObjects.addOwned(std::move(g2));

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
