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
#include "RooBukinPdf.h"

#include "RooNumIntConfig.h"

class TestBukin : public PDFTest {
protected:
   TestBukin() : PDFTest("Bukin")
   {
      auto x = std::make_unique<RooRealVar>("x", "x", 0.6, -15., 10.);
      auto Xp = std::make_unique<RooRealVar>("Xp", "Xp", 0.5, -3., 5.);
      auto sigp = std::make_unique<RooRealVar>("sigp", "sigp", 3., 1., 5.);
      auto xi = new RooRealVar("xi", "xi", -0.2, -0.3, 0.3);
      auto rho1 = new RooRealVar("rho1", "rho1", -0.1, -0.3, -0.05);
      auto rho2 = new RooRealVar("rho2", "rho2", 0.15, 0.05, 0.25);
      _pdf = std::make_unique<RooBukinPdf>("bukin", "bukin", *x, *Xp, *sigp, *xi, *rho1, *rho2);
      xi->setConstant(true);
      rho1->setConstant(true);
      rho2->setConstant(true);

      _variables.addOwned(std::move(x));

      // for (auto var : {x}) {
      //_variablesToPlot.add(*var);
      //}

      _parameters.addOwned(std::move(Xp));
      _parameters.addOwned(std::move(sigp));

      _toleranceParameter = 3e-5;
      _toleranceCompareBatches = 2.5e-14;
      //_toleranceCompareLogs{2.E-14};

      // make the integrator slightly more precise (1e-6 -> 1e-7) to reduce inaccuracy in fitting
      RooAbsReal::defaultIntegratorConfig()->setEpsAbs(1e-7);
      RooAbsReal::defaultIntegratorConfig()->setEpsRel(1e-7);
   }
};

COMPARE_FIXED_VALUES_UNNORM(TestBukin, CompareFixedValuesUnnorm)
COMPARE_FIXED_VALUES_NORM(TestBukin, CompareFixedValuesNorm)
COMPARE_FIXED_VALUES_NORM_LOG(TestBukin, CompareFixedNormLog)
FIT_TEST_SCALAR(TestBukin, RunScalar)
FIT_TEST_BATCH(TestBukin, RunBatch)
FIT_TEST_BATCH_VS_SCALAR(TestBukin, CompareBatchScalar)
