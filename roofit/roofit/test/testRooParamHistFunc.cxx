// Tests for RooParamHistFunc
// Author: Jonas Rembser, CERN  03/2020

#include "RooParamHistFunc.h"
#include "RooRealVar.h"

#include "TH1D.h"

#include "gtest/gtest.h"

#include <numeric>

TEST(RooParamHistFunc, Integration)
{
   // This tests the analytical integration of RooParamHistFunc,
   // inspired by this issue on GitHub:
   // https://github.com/root-project/root/issues/7182

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
   phf.Print("t");
   x.setRange("R1", 0, xMax * 0.5);

   EXPECT_FLOAT_EQ(phf.createIntegral(x, x)->getVal(), nEntries * binWidth);
   EXPECT_FLOAT_EQ(phf.createIntegral(x, x, "R1")->getVal(), nEntries * binWidth * 0.5);

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

   EXPECT_FLOAT_EQ(phf.createIntegral(x, x)->getVal(), ref);
   EXPECT_FLOAT_EQ(phf.createIntegral(x, x, "R1")->getVal(), refR1);
}
