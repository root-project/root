// Tests for the RooHistPdf
// Authors: Jonas Rembser, CERN 03/2023

#include <RooDataHist.h>
#include <RooFitResult.h>
#include <RooHelpers.h>
#include <RooHistPdf.h>
#include <RooLinearVar.h>
#include <RooRealIntegral.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <gtest/gtest.h>

// Verify that RooFit correctly uses analytic integration when having a
// RooLinearVar as the observable of a RooHistPdf.
TEST(RooHistPdf, AnalyticIntWithRooLinearVar)
{
   RooRealVar x{"x", "x", 0, -10, 10};
   x.setBins(10);

   RooDataHist dataHist("dataHist", "dataHist", x);
   for (int i = 0; i < x.numBins(); ++i) {
      dataHist.set(i, 10.0, 0.0);
   }

   RooRealVar shift{"shift", "shift", 2.0, -10, 10};
   RooRealVar slope{"slope", "slope", 1.0};
   RooLinearVar xShifted{"x_shifted", "x_shifted", x, slope, shift};

   RooHistPdf pdf{"pdf", "pdf", xShifted, x, dataHist};

   RooRealIntegral integ{"integ", "integ", pdf, x};

   EXPECT_DOUBLE_EQ(integ.getVal(), 90.);
   EXPECT_EQ(integ.anaIntVars().size(), 1);
}

// Regression test for https://github.com/root-project/root/issues/21159, which
// uncovered that the values were not clipped to be positive when evaluating a
// RooHistPdf with the new vectorizing evaluation backend.
TEST(RooHistPdf, EnsurePositiveValuesInFFTConvPdf)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   using namespace RooFit;

   RooWorkspace ws{"ws"};

   // Observable
   ws.factory("mass[5500,6100]");

   // Signal Gaussian
   ws.factory("mean[5800,5795,5805]");
   ws.factory("sigma[8.1,1,30]");
   ws.factory("Gaussian::sigpdf(mass, mean, sigma)");

   // Argus background
   ws.factory("argus_par[-55]");
   ws.factory("argus_m0[5665]"); // 5800 - 135
   ws.factory("ArgusBG::argus(mass, argus_m0, argus_par)");

   // Resolution Gaussian (mean fixed to 0)
   ws.factory("Gaussian::resolution(mass, 0, sigma)");

   // Convolution
   ws.factory("FFTConvPdf::conv(mass, argus, resolution)");

   // Combined model (signal + background)
   ws.factory("SUM::model(0.3*sigpdf, conv)");

   // Retrieve objects
   auto &mass = *ws.var("mass");
   auto &model = *ws.pdf("model");

   std::unique_ptr<RooDataSet> data{model.generate(mass, 1000)};

   std::unique_ptr<RooFitResult> fit_result{model.fitTo(*data, Save(true), PrintLevel(-1))};

   EXPECT_EQ(fit_result->status(), 0) << "The fit should succeed with status 0";
}
