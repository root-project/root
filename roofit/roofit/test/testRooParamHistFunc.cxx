// Tests for RooParamHistFunc
// Author: Jonas Rembser, CERN  03/2020

#include <RooMsgService.h>
#include <RooParamHistFunc.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooUniform.h>

#include <TH1D.h>
#include <TF1.h>

#include <gtest/gtest.h>

#include <numeric>

TEST(RooParamHistFunc, Integration)
{
   // This tests the analytical integration of RooParamHistFunc,
   // inspired by this issue on GitHub:
   // https://github.com/root-project/root/issues/7182

   auto& msg = RooMsgService::instance();
   msg.setGlobalKillBelow(RooFit::WARNING);

   constexpr int nBins = 20;
   constexpr double xMin = 0;
   constexpr double xMax = 10.;
   constexpr int nEntries = 1000;
   constexpr double binWidth = (xMax - xMin) / nBins;

   RooRealVar x("x", "x", xMin, xMax);
   TH1D h1("h1", "h1", nBins, xMin, xMax);
   for (int i = 0; i < nBins; ++i) {
      h1.SetBinContent(i + 1, static_cast<double>(nEntries) / nBins);
   }

   RooDataHist dh("dh", "dh", x, &h1);
   RooParamHistFunc phf("phf", "", x, dh);
   x.setRange("R1", 0, xMax * 0.5);

   std::unique_ptr<RooAbsReal> integral{phf.createIntegral(x, x)};
   std::unique_ptr<RooAbsReal> integralR1{phf.createIntegral(x, x, "R1")};

   EXPECT_FLOAT_EQ(integral->getVal(), nEntries * binWidth);
   EXPECT_FLOAT_EQ(integralR1->getVal(), nEntries * binWidth * 0.5);

   // Extending the code in issue 7182, we also want to make sure that the
   // integration also works if the bin scaling parameters are not just one,
   // which would be equivalent to RooHistFunc.
   //
   // Let's scale each bin content by the bin index:
   for (int i = 0; i < nBins; ++i) {
      auto *arg = phf.paramList().at(i);
      auto *realVar = dynamic_cast<RooRealVar *>(arg);
      realVar->setVal(i + 1);
   }

   // Compute reference values
   std::vector<double> phVals(nBins);
   for (int i = 0; i < nBins; ++i) {
      phVals[i] = h1.GetBinContent(i + 1) * (i + 1);
   }

   auto ref = std::accumulate(phVals.begin(), phVals.end(), 0.0) * binWidth;
   auto refR1 = std::accumulate(phVals.begin(), phVals.begin() + nBins / 2, 0.0) * binWidth;

   EXPECT_FLOAT_EQ(integral->getVal(), ref);
   EXPECT_FLOAT_EQ(integralR1->getVal(), refR1);
}

TEST(RooParamHistFunc, IntegrationAndCloning)
{
   // This tests the analytical integration of RooParamHistFunc
   // after the RooParamHistFunc has been cloned.
   // The test was inspired by this error reported on the forum:
   // https://root-forum.cern.ch/t/barlow-beeston-in-subrange/43909/5

   auto& msg = RooMsgService::instance();
   msg.setGlobalKillBelow(RooFit::WARNING);

   using namespace RooFit;

   RooRealVar x("x", "x", 0, 10);
   x.setRange("R1", 0, 5);
   TF1 f1("f1", "1");

   TH1D h1("h1", "h1", 10, 0, 10);
   h1.FillRandom("f1", 50);
   RooDataHist dh1("dh1", "dh1", x, &h1);

   RooParamHistFunc ph("ph", "", x, dh1);

   // Combine the RooParamHistFunc with something else in a RooRealSumPdf.
   // This is do make the test more similar to the Barlow-Beeston test,
   // which is where the RooParamHistFunc is primarily used.
   RooUniform uni("uni", "uni", RooArgList(x));
   RooRealVar frac("frac", "frac", 0.5, 0.0, 1.0);
   RooRealSumPdf model{"model", "model", ph, uni, frac};

   std::unique_ptr<RooAbsReal> integral{ph.createIntegral(x, x, "R1")};
   std::unique_ptr<RooAbsReal> integralClone{static_cast<RooAbsReal *>(integral->cloneTree())};

   RooArgSet nset{x};

   EXPECT_FLOAT_EQ(integralClone->getValV(&nset), integral->getValV(&nset));
}
