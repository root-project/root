// Tests for the SumW2Error correction
// Author: Jonas Rembser, CERN  10/2021

#include "RooFitResult.h"
#include "RooAbsPdf.h"
#include "RooGaussian.h"
#include "RooExponential.h"
#include "RooAddPdf.h"
#include "RooRandom.h"
#include "RooDataHist.h"
#include "RooDataSet.h"
#include "RooRealVar.h"

#include "gtest/gtest.h"

// GitHub issue 9118: Problem running weighted binned fit in batch mode
TEST(SumW2Error, BatchMode)
{
   auto &msg = RooMsgService::instance();
   msg.setGlobalKillBelow(RooFit::WARNING);

   RooRealVar x{"x", "x", 0, 0, 10};
   RooRealVar mu{"mu", "mu", 3, 0, 10};
   RooRealVar s{"s", "s", 1.0, 0.1, 5};
   RooRealVar c1{"c1", "c1", -0.5, -3, -0.1};
   RooRealVar f{"f", "f", 0.2, 0.0, 1.0};

   RooGaussian sig{"sig", "sig", x, mu, s};
   RooExponential bkg{"bkg", "bkg", x, c1};
   RooAddPdf model{"model", "model", {sig, bkg}, {f}};

   auto resetParametersToInitialFitValues = [&]() {
      mu.setVal(4.0);
      mu.setError(0.0);
      s.setVal(2.0);
      s.setError(0.0);
      c1.setVal(-0.4);
      c1.setError(0.0);
      f.setVal(0.3);
      f.setError(0.0);
   };

   std::size_t nEvents = 1000;

   RooRandom::randomGenerator()->SetSeed(4357);
   std::unique_ptr<RooAbsData> dataHist{model.generateBinned(x, nEvents)};
   std::unique_ptr<RooAbsData> dataSet{model.generate(x, nEvents)};

   // these datasets will be filled with a weight that is not unity
   RooRealVar weight("weight", "weight", 0.5, 0.0, 1.0);
   RooDataHist dataHistWeighted("dataHistWeighted", "dataHistWeighted", x);
   RooDataSet dataSetWeighted("dataSetWeighted", "dataSetWeighted", {x, weight}, "weight");

   for (std::size_t i = 0; i < nEvents; ++i) {
      dataHistWeighted.add(*dataSet->get(), 0.5); // filling the histogram from a dataset is easier
      dataSetWeighted.add(*dataSet->get(), 0.5);
   }

   auto fit = [&](RooAbsData &data, bool sumw2 = false, bool batchmode = false, std::string const &minimizer = "Minuit",
                  int printLevel = -1) {
      using namespace RooFit;

      resetParametersToInitialFitValues();

      return std::unique_ptr<RooFitResult>{model.fitTo(data, Save(), SumW2Error(sumw2), Strategy(1),
                                                       BatchMode(batchmode), Minimizer(minimizer.c_str()),
                                                       PrintLevel(printLevel))};
   };

   // Compare batch mode vs. scalar mode for non-SumW2 fits on UNWEIGHTED datasets
   EXPECT_TRUE(fit(*dataSet, 0, 0, "Minuit")->isIdentical(*fit(*dataSet, 0, 1, "Minuit")))
      << " different results for Minuit fit to RooDataSet without SumW2Error correction.";
   EXPECT_TRUE(fit(*dataHist, 0, 0, "Minuit")->isIdentical(*fit(*dataHist, 0, 1, "Minuit")))
      << " different results for Minuit fit to RooDataHist without SumW2Error correction.";
   EXPECT_TRUE(fit(*dataSet, 0, 0, "Minuit2")->isIdentical(*fit(*dataSet, 0, 1, "Minuit2")))
      << " different results for Minuit2 fit to RooDataSet without SumW2Error correction.";
   EXPECT_TRUE(fit(*dataHist, 0, 0, "Minuit2")->isIdentical(*fit(*dataHist, 0, 1, "Minuit2")))
      << " different results for Minuit2 fit to RooDataHist without SumW2Error correction.";

   // We can't compare the covariance matrix in these next cases, because it is
   // externally provided. Still, it's okay because the parameter values and
   // errors are compared, where the errors are inferred from the external
   // covariance matrix.

   // Compare batch mode vs. scalar mode for SumW2 fits on UNWEIGHTED datasets
   EXPECT_TRUE(fit(*dataSet, 1, 0, "Minuit")->isIdenticalNoCov(*fit(*dataSet, 1, 1, "Minuit")))
      << " different results for Minuit fit to RooDataSet with SumW2Error correction.";
   EXPECT_TRUE(fit(*dataHist, 1, 0, "Minuit")->isIdenticalNoCov(*fit(*dataHist, 1, 1, "Minuit")))
      << " different results for Minuit fit to RooDataHist with SumW2Error correction.";
   EXPECT_TRUE(fit(*dataSet, 1, 0, "Minuit2")->isIdenticalNoCov(*fit(*dataSet, 1, 1, "Minuit2")))
      << " different results for Minuit2 fit to RooDataSet with SumW2Error correction.";
   EXPECT_TRUE(fit(*dataHist, 1, 0, "Minuit2")->isIdenticalNoCov(*fit(*dataHist, 1, 1, "Minuit2")))
      << " different results for Minuit2 fit to RooDataHist with SumW2Error correction.";

   // Compare batch mode vs. scalar mode for SumW2 fits on WEIGHTED datasets
   EXPECT_TRUE(fit(dataSetWeighted, 1, 0, "Minuit")->isIdenticalNoCov(*fit(dataSetWeighted, 1, 1, "Minuit")))
      << " different results for Minuit fit to weighted RooDataSet with SumW2Error correction.";
   EXPECT_TRUE(fit(dataHistWeighted, 1, 0, "Minuit")->isIdenticalNoCov(*fit(dataHistWeighted, 1, 1, "Minuit")))
      << " different results for Minuit fit to weighted RooDataHist with SumW2Error correction.";
   EXPECT_TRUE(fit(dataSetWeighted, 1, 0, "Minuit2")->isIdenticalNoCov(*fit(dataSetWeighted, 1, 1, "Minuit2")))
      << " different results for Minuit2 fit to weighted RooDataSet with SumW2Error correction.";
   EXPECT_TRUE(fit(dataHistWeighted, 1, 0, "Minuit2")->isIdenticalNoCov(*fit(dataHistWeighted, 1, 1, "Minuit2")))
      << " different results for Minuit2 fit to weighted RooDataHist with SumW2Error correction.";
}
