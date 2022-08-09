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

#include "RooPoisson.h"
#include "RooGaussian.h"
#include "RooExponential.h"
#include "RooProduct.h"
#include "RooConstVar.h"
#include "RooRealSumPdf.h"
#include "RooAddPdf.h"

class TestNestedPDFs : public PDFTest
{
  protected:
    TestNestedPDFs() :
      PDFTest("Gauss + RooRealSumPdf(pol2)", 50000)
  {
      auto x = new RooRealVar("x", "x", -5., 5.);

      // Implement a polynomial. Value ranges are chosen to keep it positive.
      // Note that even though the parameters are constant for the fit, they are still
      // varied within their ranges when testing the function at random parameter points.
      auto a0 = new RooRealVar("a0", "a0", 2., 3., 10.);
      auto a1 = new RooRealVar("a1", "a1", -2, -2.1, -1.9);
      auto a2 = new RooRealVar("a2", "a2", 1., 1., 5.);
      a0->setConstant(true);
      a1->setConstant(true);
      auto xId = new RooProduct("xId", "x", RooArgList(*x));
      auto xSq = new RooProduct("xSq", "x^2", RooArgList(*x, *x));
      auto one = new RooConstVar("one", "one", 1.);
      auto pol = new RooRealSumPdf("pol", "pol", RooArgList(*one, *xId, *xSq), RooArgList(*a0, *a1, *a2));

      auto mean = new RooRealVar("mean", "mean of gaussian", 2., 0., 20.);
      auto sigma = new RooRealVar("sigma", "width of gaussian", 0.337, 0.1, 10);
      sigma->setConstant();
      auto gauss = new RooGaussian("gauss", "gaussian PDF", *x, *mean, *sigma);

      auto nGauss = new RooRealVar("nGauss", "Fraction of Gauss component", 0.05, 0., 0.5);
      _pdf = std::make_unique<RooAddPdf>("SumGausPol", "Sum of a Gauss and a simple polynomial",
          RooArgSet(*gauss, *pol),
          RooArgSet(*nGauss));

      _variables.addOwned(*x);

//      _variablesToPlot.add(*x);

      for (auto par : std::initializer_list<RooAbsArg*>{mean, sigma, a0, a1, a2, nGauss}) {
        _parameters.addOwned(*par);
      }

      for (auto obj : std::initializer_list<RooAbsArg*>{gauss, pol}) {
        _otherObjects.addOwned(*obj);
      }

//      RooMsgService::instance().getStream(0).minLevel = RooFit::DEBUG;

//      _toleranceParameter = 1.E-4;
//      _toleranceCorrelation = 1.E-3;
      _toleranceCompareLogs = 2.E-12;
  }
};

COMPARE_FIXED_VALUES_UNNORM(TestNestedPDFs, CompareFixedUnnorm)
COMPARE_FIXED_VALUES_NORM(TestNestedPDFs, CompareFixedNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestNestedPDFs, CompareFixedNormLog)

FIT_TEST_SCALAR(TestNestedPDFs, RunScalar)
FIT_TEST_BATCH(TestNestedPDFs, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestNestedPDFs, CompareBatchScalar)





