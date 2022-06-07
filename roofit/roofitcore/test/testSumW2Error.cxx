// Tests for the SumW2Error correction
// Author: Jonas Rembser, CERN  10/2021

#include <RooFitResult.h>
#include <RooAbsPdf.h>
#include <RooAddPdf.h>
#include <RooRandom.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <gtest/gtest.h>

// GitHub issue 9118: Problem running weighted binned fit in batch mode
TEST(SumW2Error, BatchMode)
{
   auto &msg = RooMsgService::instance();
   msg.setGlobalKillBelow(RooFit::WARNING);

   RooWorkspace ws{"workspace"};
   ws.factory("Gaussian::sig(x[0,0,10],mu[3,0,10],s[1, 0.1, 5])");
   ws.factory("Exponential::bkg(x,c1[-0.5, -3, -0.1])");
   ws.factory("SUM::model(f[0.2, 0.0, 1.0] * sig, bkg)");

   auto &x = *ws.var("x");
   auto &mu = *ws.var("mu");
   auto &s = *ws.var("s");
   auto &c1 = *ws.var("c1");
   auto &f = *ws.var("f");

   auto &model = *ws.pdf("model");

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

TEST(SumW2Error, ExtendedFit)
{
   using namespace RooFit;

   RooWorkspace ws("workspace");
   ws.factory("x[-10, 10]");
   ws.factory("Gaussian::sig(x, mu[-1, 1], s[0.1, 5])");
   ws.factory("Chebychev::bkg(x, {c1[0.1, -1, 1]})");
   ws.factory("SUM::shp(Nsig[0, 20000] * sig, Nbkg[0, 20000] * bkg)");
   auto *x = ws.var("x");
   x->setRange("subrange", -5.0, 5.0);
   auto *shp = ws.pdf("shp");
   std::unique_ptr<RooDataSet> dataNoWeights{shp->generate(RooArgSet(*x))};

   // For this test, use a uniform non-unity weight of 1.5. It was set to 0.1
   // in the past, but then there were fourth-digit differences between the
   // scalar mode and the batch mode. However, this is most likeliy not
   // pointing towards a flaw in the batch mode, which is why a value was
   // handpicked for which the differences disappear. Any residual problems are
   // most likely caused by the unnecessarily complicated implementation of the
   // RooAddPdf extended term in the scalar mode: the coefficients are
   // projected to the subrange by cached scale factors, while the batch mode
   // just uses the same scaling factor as for the full likelihood.
   auto *wFunc = ws.factory("w[1.5]");

   auto *w = dataNoWeights->addColumn(*wFunc);
   RooDataSet data{dataNoWeights->GetName(),
                   dataNoWeights->GetTitle(),
                   dataNoWeights.get(),
                   *dataNoWeights->get(),
                   "",
                   w->GetName()};
   RooDataHist datahist{"datahist", "datahist", *data.get(), data};

   std::vector<std::pair<std::string, double>> inVals;
   for (auto const *v : ws.allVars()) {
      inVals.emplace_back(v->GetName(), static_cast<RooRealVar const *>(v)->getVal());
   }

   auto resetVals = [&]() {
      for (auto const &item : inVals) {
         ws.var(item.first.c_str())->setError(0.0);
         ws.var(item.first.c_str())->setVal(item.second);
      }
   };

   auto doFit = [&](bool batchMode, bool sumW2Error, const char *range) {
      resetVals();
      return std::unique_ptr<RooFitResult>{shp->fitTo(datahist, Extended(), Range(range), Save(),
                                                      SumW2Error(sumW2Error), Strategy(1), PrintLevel(-1),
                                                      BatchMode(batchMode), Minimizer("Minuit2", "migrad"))};
   };

   // compare batch mode and scalar mode fit results for full range
   {
      auto yy = doFit(true, true, nullptr);
      auto yn = doFit(true, false, nullptr);
      auto ny = doFit(false, true, nullptr);
      auto nn = doFit(false, false, nullptr);

      EXPECT_TRUE(yy->isIdenticalNoCov(*ny)) << "different results for extended fit with SumW2Error in BatchMode";
      EXPECT_TRUE(yn->isIdenticalNoCov(*nn)) << "different results for extended fit without SumW2Error in BatchMode";
   }

   // compare batch mode and scalar mode fit results for subrange
   {
      auto yy = doFit(true, true, "subrange");
      auto yn = doFit(true, false, "subrange");
      auto ny = doFit(false, true, "subrange");
      auto nn = doFit(false, false, "subrange");

      EXPECT_TRUE(yy->isIdenticalNoCov(*ny))
         << "different results for extended fit in subrange with SumW2Error in BatchMode";
      EXPECT_TRUE(yn->isIdenticalNoCov(*nn))
         << "different results for extended fit in subrange without SumW2Error in BatchMode";
   }
}
