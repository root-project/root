// Tests for the SumW2Error correction
// Author: Jonas Rembser, CERN  10/2021

#include "RooFitResult.h"
#include "RooAbsPdf.h"
#include "RooWorkspace.h"
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

   RooWorkspace ws{"workspace"};

   auto *x = ws.factory("x[-10, 10]");
   ws.factory("Gaussian::sig(x, mu[-1, 1], s[0.1, 5])");
   ws.factory("Chebychev::bkg(x, {c1[0.1, -1, 1]})");
   auto *shp = static_cast<RooAbsPdf *>(ws.factory("SUM::shp(Nsig[0, 20000] * sig, Nbkg[0, 20000] * bkg)"));
   auto &model = *shp;

   // parameters
   auto *mu = ws.var("mu");
   auto *s = ws.var("s");
   auto *c1 = ws.var("c1");
   auto *Nsig = ws.var("Nsig");
   auto *Nbkg = ws.var("Nbkg");

   auto resetParameters = [&]() {
      mu->setVal(0.0);
      mu->setError(0.0);
      s->setVal(2.0);
      s->setError(0.0);
      c1->setVal(0.1);
      c1->setError(0.0);
      Nsig->setVal(10000.0);
      Nsig->setError(0.0);
      Nbkg->setVal(10000.0);
      Nbkg->setError(0.0);
   };

   std::size_t nEvents = 1000;

   RooRandom::randomGenerator()->SetSeed(4357);
   std::unique_ptr<RooAbsData> dataHist{shp->generateBinned(*x, nEvents)};
   std::unique_ptr<RooAbsData> dataSet{shp->generate(*x, nEvents)};

   auto fit = [&](RooAbsData &data, bool sumw2 = false, bool batchmode = false, std::string const &minimizer = "Minuit",
                  int printLevel = -1) {
      using namespace RooFit;

      resetParameters();

      return std::unique_ptr<RooFitResult>{model.fitTo(data, Extended(), Save(), SumW2Error(sumw2), Strategy(1),
                                                       BatchMode(batchmode), Minimizer(minimizer.c_str()),
                                                       PrintLevel(printLevel))};
   };

   EXPECT_TRUE(fit(*dataSet, 0, 0, "Minuit")->isIdentical(*fit(*dataSet, 0, 1, "Minuit")))
      << " different results for Minuit fit to RooDataSet without SumW2Error correction.";
   EXPECT_TRUE(fit(*dataHist, 0, 0, "Minuit")->isIdentical(*fit(*dataHist, 0, 1, "Minuit")))
      << " different results for Minuit fit to RooDataHist without SumW2Error correction.";
   EXPECT_TRUE(fit(*dataSet, 0, 0, "Minuit2")->isIdentical(*fit(*dataSet, 0, 1, "Minuit2")))
      << " different results for Minuit2 fit to RooDataSet without SumW2Error correction.";
   EXPECT_TRUE(fit(*dataHist, 0, 0, "Minuit2")->isIdentical(*fit(*dataHist, 0, 1, "Minuit2")))
      << " different results for Minuit2 fit to RooDataHist without SumW2Error correction.";

   // We can't compare the covariance matrix in these cases, because it is
   // externally provided. Still, it's okay because the parameter values and
   // errors are compared, where the errors are inferred from the external
   // covariance matrix.
   EXPECT_TRUE(fit(*dataSet, 1, 0, "Minuit")->isIdenticalNoCov(*fit(*dataSet, 1, 1, "Minuit")))
      << " different results for Minuit fit to RooDataSet with SumW2Error correction.";
   EXPECT_TRUE(fit(*dataHist, 1, 0, "Minuit")->isIdenticalNoCov(*fit(*dataHist, 1, 1, "Minuit")))
      << " different results for Minuit fit to RooDataHist with SumW2Error correction.";
   EXPECT_TRUE(fit(*dataSet, 1, 0, "Minuit2")->isIdenticalNoCov(*fit(*dataSet, 1, 1, "Minuit2")))
      << " different results for Minuit2 fit to RooDataSet with SumW2Error correction.";
   EXPECT_TRUE(fit(*dataHist, 1, 0, "Minuit2")->isIdenticalNoCov(*fit(*dataHist, 1, 1, "Minuit2")))
      << " different results for Minuit2 fit to RooDataHist with SumW2Error correction.";
}
