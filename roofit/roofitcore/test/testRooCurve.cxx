/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2024
 *
 * Copyright (c) 2024, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooCmdArg.h>
#include <RooGenericPdf.h>
#include <RooHelpers.h>
#include <RooPlot.h>
#include <RooRealVar.h>

#include "gtest_wrapper.h"

// Cross-check to make sure the integration works correctly even if there is
// only one midpoint on the RooCurve. Covers GitHub issue #9838 (the reproducer
// in that issue was translated to this test).
TEST(RooPlot, Average)
{
   // Silence the info about numeric integration because we don't care about it
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::NumericIntegration, true};

   RooRealVar x("x", "x", 0, 50);
   RooGenericPdf func("func", "Test Function", "x", x);

   std::unique_ptr<RooPlot> xframe{x.frame()};

   func.plotOn(xframe.get(), RooFit::Name("funcCurve"));

   RooCurve *funcCurve = xframe->getCurve("funcCurve");

   const double tol = 1e-10;

   for (double i = 10; i < 11; i += 0.1) {
      double avg = funcCurve->average(i, i + 0.1);

      double xFirst = funcCurve->interpolate(i, tol);
      double xLast = funcCurve->interpolate(i + 0.1, tol);

      EXPECT_NEAR(avg, 0.5 * (xLast + xFirst), tol);
   }
}
