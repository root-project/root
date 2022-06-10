/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN  12/2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooConstVar.h"
#include "RooArgusBG.h"
#include "RooAddPdf.h"
#include "RooNLLVar.h"
#include "RooNLLVarNew.h"
#include "RooFitDriver.h"
#include "RooFit/BatchModeHelpers.h"
#include "RooGaussian.h"
#include "RooMinimizer.h"
#include "RooFitResult.h"

#include <utility>
#include <chrono>

#include "gtest/gtest.h"

struct FitOutput {
   int evalCount;
   long int elapsedTime;
   std::unique_ptr<RooFitResult> result;
};

TEST(testRooFitDriver, SimpleLikelihoodFit)
{
   constexpr bool verbose = false;

   RooRealVar x("x", "x", -10, 10);

   RooRealVar mean("mean", "mean", -1, -10, 10);
   RooRealVar width("width", "width", 1., 0.01, 10);
   RooGaussian model("model", "model", x, mean, width);

   std::size_t nEvents = 2000;

   RooDataSet *data = model.generate(x, nEvents);

   auto resetGraph = [&]() {
      // reset initial fit values so both fits have the same starting condition
      mean.setVal(1);
      width.setVal(2);

      // reset the error such that they are not used as the step size in the fit
      mean.setError(0.0);
      width.setError(0.0);
   };

   auto doFit = [](RooAbsReal &absReal) {
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

      RooMinimizer minimizer(absReal);
      minimizer.setPrintLevel(-1);
      minimizer.minimize("Minuit", "minuit");
      std::unique_ptr<RooFitResult> result(minimizer.save());

      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

      long int elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

      return FitOutput{minimizer.evalCounter(), elapsedTime, std::move(result)};
   };

   resetGraph();

   // do likelihood fitting the old way...
   auto nllScalar = model.createNLL(*data, RooFit::BatchMode(false));
   auto resultScalar = doFit(*nllScalar);
   if (verbose)
      std::cout << "- scalar mode fit took " << resultScalar.elapsedTime << " ms" << std::endl;

   resetGraph();

   // ...and now the new way with RooFitDriver
   using namespace ROOT::Experimental;
   RooNLLVarNew nll("nll", "nll", model, *data->get(), false, "", false);
   auto driver = std::make_unique<RooFitDriver>(nll, x, RooFit::BatchModeOption::Cpu);
   driver->setData(*data);
   auto wrapper = RooFit::BatchModeHelpers::makeDriverAbsRealWrapper(std::move(driver), *data->get());
   auto resultBatchNew = doFit(*wrapper);
   if (verbose) {
      std::cout << "-  batch mode fit took " << resultBatchNew.elapsedTime << " ms" << std::endl;
   }

   // make sure the fit result is the same and that there were the same number
   // of evaluations
   EXPECT_EQ(resultScalar.evalCount, resultBatchNew.evalCount);
   ASSERT_TRUE(resultBatchNew.result->isIdentical(*resultScalar.result));
}
