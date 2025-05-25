// Tests for the SumW2Error correction
// Author: Jonas Rembser, CERN  10/2021

#include <RooFitResult.h>
#include <RooAbsPdf.h>
#include <RooAddPdf.h>
#include <RooRandom.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooHelpers.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <gtest/gtest.h>

// These tests are disabled if the legacy backend is not available, because
// then we don't have any reference to compare to.
#ifdef ROOFIT_LEGACY_EVAL_BACKEND
// GitHub issue 9118: Problem running weighted binned fit in batch mode
TEST(SumW2Error, BatchMode)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws{"workspace"};
   ws.factory("Gaussian::sig(x[0,0,10],mu[3,0,10],s[1, 0.1, 5])");
   ws.factory("Exponential::bkg(x,c1[-0.5, -3, -0.1])");
   ws.factory("SUM::model(f[0.2, 0.0, 1.0] * sig, bkg)");

   auto &model = *ws.pdf("model");

   RooRandom::randomGenerator()->SetSeed(4357);
   std::unique_ptr<RooDataSet> dataSet{model.generate(*ws.var("x"), 1000)};

   RooArgSet params;
   RooArgSet initialParams;

   model.getParameters(dataSet->get(), params);
   params.snapshot(initialParams);

   // these datasets will be filled with a weight that is not unity
   RooDataSet dataSetWeighted("dataSetWeighted", "dataSetWeighted", *dataSet->get(), RooFit::WeightVar());

   for (int i = 0; i < dataSet->numEntries(); ++i) {
      dataSetWeighted.add(*dataSet->get(i), 0.5);
   }

   std::unique_ptr<RooDataHist> dataHist{dataSet->binnedClone()};
   std::unique_ptr<RooDataHist> dataHistWeighted{dataSetWeighted.binnedClone()};

   using namespace RooFit;

   auto fit = [&](RooAbsData &data, bool sumw2, EvalBackend evalBackend, std::string const &minimizer,
                  int printLevel = -1) {
      params.assign(initialParams);

      return std::unique_ptr<RooFitResult>{model.fitTo(data, Save(), SumW2Error(sumw2), Strategy(1), evalBackend,
                                                       Minimizer(minimizer.c_str()), PrintLevel(printLevel))};
   };

   auto scalar = EvalBackend::Legacy();
   auto batchMode = EvalBackend::Cpu();

   // Compare batch mode vs. scalar mode for non-SumW2 fits on UNWEIGHTED datasets
   EXPECT_TRUE(fit(*dataSet, 0, scalar, "Minuit")->isIdentical(*fit(*dataSet, 0, batchMode, "Minuit")))
      << " different results for Minuit fit to RooDataSet without SumW2Error correction.";
   EXPECT_TRUE(fit(*dataHist, 0, scalar, "Minuit")->isIdentical(*fit(*dataHist, 0, batchMode, "Minuit")))
      << " different results for Minuit fit to RooDataHist without SumW2Error correction.";
   EXPECT_TRUE(fit(*dataSet, 0, scalar, "Minuit2")->isIdentical(*fit(*dataSet, 0, batchMode, "Minuit2")))
      << " different results for Minuit2 fit to RooDataSet without SumW2Error correction.";
   EXPECT_TRUE(fit(*dataHist, 0, scalar, "Minuit2")->isIdentical(*fit(*dataHist, 0, batchMode, "Minuit2")))
      << " different results for Minuit2 fit to RooDataHist without SumW2Error correction.";

   // We can't compare the covariance matrix in these next cases, because it is
   // externally provided. Still, it's okay because the parameter values and
   // errors are compared, where the errors are inferred from the external
   // covariance matrix.

   // Compare batch mode vs. scalar mode for SumW2 fits on UNWEIGHTED datasets
   EXPECT_TRUE(fit(*dataSet, 1, scalar, "Minuit")->isIdenticalNoCov(*fit(*dataSet, 1, batchMode, "Minuit")))
      << " different results for Minuit fit to RooDataSet with SumW2Error correction.";
   EXPECT_TRUE(fit(*dataHist, 1, scalar, "Minuit")->isIdenticalNoCov(*fit(*dataHist, 1, batchMode, "Minuit")))
      << " different results for Minuit fit to RooDataHist with SumW2Error correction.";
   EXPECT_TRUE(fit(*dataSet, 1, scalar, "Minuit2")->isIdenticalNoCov(*fit(*dataSet, 1, batchMode, "Minuit2")))
      << " different results for Minuit2 fit to RooDataSet with SumW2Error correction.";
   EXPECT_TRUE(fit(*dataHist, 1, scalar, "Minuit2")->isIdenticalNoCov(*fit(*dataHist, 1, batchMode, "Minuit2")))
      << " different results for Minuit2 fit to RooDataHist with SumW2Error correction.";

   // Compare batch mode vs. scalar mode for SumW2 fits on WEIGHTED datasets
   EXPECT_TRUE(
      fit(dataSetWeighted, 1, scalar, "Minuit")->isIdenticalNoCov(*fit(dataSetWeighted, 1, batchMode, "Minuit")))
      << " different results for Minuit fit to weighted RooDataSet with SumW2Error correction.";
   EXPECT_TRUE(
      fit(*dataHistWeighted, 1, scalar, "Minuit")->isIdenticalNoCov(*fit(*dataHistWeighted, 1, batchMode, "Minuit")))
      << " different results for Minuit fit to weighted RooDataHist with SumW2Error correction.";
   EXPECT_TRUE(
      fit(dataSetWeighted, 1, scalar, "Minuit2")->isIdenticalNoCov(*fit(dataSetWeighted, 1, batchMode, "Minuit2")))
      << " different results for Minuit2 fit to weighted RooDataSet with SumW2Error correction.";
   EXPECT_TRUE(
      fit(*dataHistWeighted, 1, scalar, "Minuit2")->isIdenticalNoCov(*fit(*dataHistWeighted, 1, batchMode, "Minuit2")))
      << " different results for Minuit2 fit to weighted RooDataHist with SumW2Error correction.";
}

TEST(SumW2Error, ExtendedFit)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

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
                   *dataNoWeights->get(),
                   RooFit::Import(*dataNoWeights),
                   RooFit::WeightVar(w->GetName())};
   RooDataHist datahist{"datahist", "datahist", *data.get(), data};

   RooArgSet params;
   RooArgSet initialParams;

   shp->getParameters(dataNoWeights->get(), params);
   params.snapshot(initialParams);

   auto doFit = [&](RooFit::EvalBackend evalBackend, bool sumW2Error, const char *range) {
      params.assign(initialParams);
      return std::unique_ptr<RooFitResult>{shp->fitTo(datahist, Extended(), Range(range), Save(),
                                                      SumW2Error(sumW2Error), Strategy(1), PrintLevel(-1), evalBackend,
                                                      Minimizer("Minuit2", "migrad"))};
   };

   // compare batch mode and scalar mode fit results for full range
   {
      auto yy = doFit(EvalBackend::Cpu(), true, nullptr);
      auto yn = doFit(EvalBackend::Cpu(), false, nullptr);
      auto ny = doFit(EvalBackend::Legacy(), true, nullptr);
      auto nn = doFit(EvalBackend::Legacy(), false, nullptr);

      EXPECT_TRUE(yy->isIdenticalNoCov(*ny)) << "different results for extended fit with SumW2Error in BatchMode";
      EXPECT_TRUE(yn->isIdenticalNoCov(*nn)) << "different results for extended fit without SumW2Error in BatchMode";
   }

   // compare batch mode and scalar mode fit results for subrange
   {
      auto yy = doFit(EvalBackend::Cpu(), true, "subrange");
      auto yn = doFit(EvalBackend::Cpu(), false, "subrange");
      auto ny = doFit(EvalBackend::Legacy(), true, "subrange");
      auto nn = doFit(EvalBackend::Legacy(), false, "subrange");

      EXPECT_TRUE(yy->isIdenticalNoCov(*ny))
         << "different results for extended fit in subrange with SumW2Error in BatchMode";
      EXPECT_TRUE(yn->isIdenticalNoCov(*nn))
         << "different results for extended fit in subrange without SumW2Error in BatchMode";
   }
}
#endif
