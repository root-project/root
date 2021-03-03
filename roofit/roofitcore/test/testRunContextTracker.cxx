// Tests for the RooRunContextTracker
// Authors: Jonas Rembser, CERN  03/2020

#include "RooHelpers.h"
#include "RooDataSet.h"
#include "RooPolyVar.h"
#include "RooGaussian.h"
#include "RooRealVar.h"
#include "RooNLLVar.h"
#include "RunContext.h"
#include "RunContextTracker.h"

#include "TRandom.h"

#include "gtest/gtest.h"

#include <memory>

RooDataSet makeFakeDataXY()
{
   RooRealVar x("x", "x", -10, 10);
   RooRealVar y("y", "y", -10, 10);
   RooArgSet coord(x, y);

   RooDataSet dataSet{"dataSet", "dataSet", RooArgSet(x, y)};

   for (int i = 0; i < 10000; i++) {
      Double_t tmpy = gRandom->Gaus(0, 10);
      Double_t tmpx = gRandom->Gaus(0.5 * tmpy, 1);
      if (fabs(tmpy) < 10 && fabs(tmpx) < 10) {
         x = tmpx;
         y = tmpy;
         dataSet.add(coord);
      }
   }

   return dataSet;
}

TEST(RooRunContextTracker, Standalone)
{
   // Optionally enable message logging on FastEvaluations for debugging
   // RooMsgService::instance().addStream(RooFit::DEBUG, Topic(RooFit::FastEvaluations));

   // Create observables
   RooRealVar x("x", "x", -10, 10);
   RooRealVar y("y", "y", -10, 10);

   // Create function f(y) = a0 + a1*y
   RooRealVar a0("a0", "a0", -0.5, -5, 5);
   RooRealVar a1("a1", "a1", -0.5, -1, 1);
   RooPolyVar fy("fy", "fy", y, RooArgSet(a0, a1));

   // Create gauss(x,f(y),s)
   RooRealVar sigma("sigma", "width of gaussian", 0.5, 0.1, 2.0);
   RooGaussian model("model", "Gaussian with shifting mean", x, fy, sigma);

   auto dataSet = makeFakeDataXY();
   const std::size_t numEntries = static_cast<std::size_t>(dataSet.numEntries());

   // Evaluate model in batchMode
   RooBatchCompute::RunContext runContext{};

   auto hasKept = [](RooBatchCompute::RunContext const &context, RooAbsReal const &arg) {
      return context.spans.find(&arg) != context.spans.end();
   };
   auto changeVal = [](RooRealVar &var) { var.setVal(var.getVal() + 0.001); };

   dataSet.getBatches(runContext, 0, numEntries);
   RooArgSet nset{x, y};

   // 1st iteration
   model.getValues(runContext, &nset);

   // In the first iteration create the RunContextTracker. This has to be done
   // after this first call to `getValues` for the RunContextTracker to know
   // which arguments to track
   RunContextTracker runContextTracker{runContext};
   {
      runContextTracker.resetTrackers();

      changeVal(a0);

      // The first argument is just to have a caller for the RooMsgService
      runContextTracker.cleanRunContext(model, runContext);

      // Let's think about what the runContextTracker should have cleared
      // based on the computation graph topology
      ASSERT_FALSE(hasKept(runContext, a0));
      ASSERT_TRUE(hasKept(runContext, a1));
      ASSERT_FALSE(hasKept(runContext, fy));
      ASSERT_TRUE(hasKept(runContext, sigma));
      ASSERT_FALSE(hasKept(runContext, model));
      ASSERT_TRUE(hasKept(runContext, x));
      ASSERT_TRUE(hasKept(runContext, y));
   }
   {
      // 2nd iteration
      model.getValues(runContext, &nset);
      runContextTracker.resetTrackers();

      changeVal(a0);
      changeVal(a1);
      changeVal(sigma);

      runContextTracker.cleanRunContext(model, runContext);

      ASSERT_FALSE(hasKept(runContext, a0));
      ASSERT_FALSE(hasKept(runContext, a1));
      ASSERT_FALSE(hasKept(runContext, fy));
      ASSERT_FALSE(hasKept(runContext, sigma));
      ASSERT_FALSE(hasKept(runContext, model));
      ASSERT_TRUE(hasKept(runContext, x));
      ASSERT_TRUE(hasKept(runContext, y));
   }
   {
      // 3rd iteration
      model.getValues(runContext, &nset);
      runContextTracker.resetTrackers();

      changeVal(sigma);

      runContextTracker.cleanRunContext(model, runContext);

      ASSERT_TRUE(hasKept(runContext, a0));
      ASSERT_TRUE(hasKept(runContext, a1));
      ASSERT_TRUE(hasKept(runContext, fy));
      ASSERT_FALSE(hasKept(runContext, sigma));
      ASSERT_FALSE(hasKept(runContext, model));
      ASSERT_TRUE(hasKept(runContext, x));
      ASSERT_TRUE(hasKept(runContext, y));
   }
}

TEST(RooRunContextTracker, InsideRooNLLVar)
{
   // It is important to also test the RooRunContextTracker as it is used
   // inside RooNLLVar, because there are many things that can go wrong when
   // the operation modes are changed during the construction of the RooNLLVar

   // Create observables
   RooRealVar x("x", "x", -10, 10);
   RooRealVar y("y", "y", -10, 10);

   // Create function f(y) = a0 + a1*y
   RooRealVar a0("a0", "a0", -0.5, -5, 5);
   RooRealVar a1("a1", "a1", -0.5, -1, 1);
   RooPolyVar fy("fy", "fy", y, RooArgSet(a0, a1));

   // Create gauss(x,f(y),s)
   RooRealVar sigma("sigma", "width of gaussian", 0.5, 0.1, 2.0);
   RooGaussian model("model", "Gaussian with shifting mean", x, fy, sigma);

   auto dataSet = makeFakeDataXY();

   auto changeVal = [](RooRealVar &var) { var.setVal(var.getVal() + 0.001); };
   auto hasKept = [](std::string const &logStr, RooAbsReal const &arg) {
      return logStr.find(std::string(arg.GetName()) + " kept", 0) != std::string::npos;
   };

   // Create the NLL object
   std::unique_ptr<RooAbsReal> nll{ model.createNLL(dataSet, RooFit::BatchMode(true)) };

   nll->getVal();

   {
      RooHelpers::HijackMessageStream hijack(RooFit::DEBUG, RooFit::FastEvaluations);

      changeVal(a0);

      nll->getVal();
      auto const &logString = hijack.str();

      ASSERT_FALSE(hasKept(logString, a0));
      ASSERT_TRUE(hasKept(logString, a1));
      ASSERT_FALSE(hasKept(logString, fy));
      ASSERT_TRUE(hasKept(logString, sigma));
      ASSERT_FALSE(hasKept(logString, model));
      ASSERT_TRUE(hasKept(logString, x));
      ASSERT_TRUE(hasKept(logString, y));
   }

   {
      RooHelpers::HijackMessageStream hijack(RooFit::DEBUG, RooFit::FastEvaluations);

      changeVal(a0);
      changeVal(a1);
      changeVal(sigma);

      nll->getVal();
      auto const &logString = hijack.str();

      ASSERT_FALSE(hasKept(logString, a0));
      ASSERT_FALSE(hasKept(logString, a1));
      ASSERT_FALSE(hasKept(logString, fy));
      ASSERT_FALSE(hasKept(logString, sigma));
      ASSERT_FALSE(hasKept(logString, model));
      ASSERT_TRUE(hasKept(logString, x));
      ASSERT_TRUE(hasKept(logString, y));
   }
   {
      RooHelpers::HijackMessageStream hijack(RooFit::DEBUG, RooFit::FastEvaluations);

      changeVal(sigma);

      nll->getVal();
      auto const &logString = hijack.str();

      ASSERT_TRUE(hasKept(logString, a0));
      ASSERT_TRUE(hasKept(logString, a1));
      ASSERT_TRUE(hasKept(logString, fy));
      ASSERT_FALSE(hasKept(logString, sigma));
      ASSERT_FALSE(hasKept(logString, model));
      ASSERT_TRUE(hasKept(logString, x));
      ASSERT_TRUE(hasKept(logString, y));
   }
}
