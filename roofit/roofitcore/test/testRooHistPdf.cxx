// Tests for the RooHistPdf
// Authors: Jonas Rembser, CERN 03/2023

#include <RooDataHist.h>
#include <RooFitResult.h>
#include <RooFormulaVar.h>
#include <RooHelpers.h>
#include <RooHistPdf.h>
#include <RooLinearVar.h>
#include <RooRealIntegral.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <TH1D.h>

#include <gtest/gtest.h>

#include <list>
#include <memory>

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

// GitHub issue #13030: plotting a RooHistPdf whose observable is a
// transformation of the plot variable (e.g. a shift `x_shifted = x - shift`)
// produced visually broken curves because `binBoundaries` and
// `plotSamplingHint` returned the raw histogram boundaries in the histogram
// observable coordinate instead of the plot observable coordinate.
TEST(RooHistPdf, ShiftedBinBoundaries)
{
   RooRealVar x{"x", "x", 1000, 1500};
   x.setBins(5); // bin edges in x: 1000, 1100, 1200, 1300, 1400, 1500

   RooRealVar shift{"shift", "shift", 25.0, -100, 100};

   TH1D h{"h", "", x.numBins(), x.getMin(), x.getMax()};
   for (int i = 1; i <= h.GetNbinsX(); ++i) {
      h.SetBinContent(i, 1.0);
   }
   RooDataHist dh{"dh", "", x, &h};

   // Case 1: pdfObs = x - shift as a RooFormulaVar (no l-value inverse).
   {
      RooFormulaVar xShifted{"x_shifted", "x - shift", {x, shift}};
      RooHistPdf pdf{"pdf_f", "", xShifted, x, dh, 0};
      std::unique_ptr<std::list<double>> boundaries{pdf.binBoundaries(x, 1000.0, 1600.0)};
      ASSERT_TRUE(boundaries);
      // Hist bin boundary `b` satisfies `x - shift = b`, so plot_x = b + shift.
      const std::vector<double> expected{1025.0, 1125.0, 1225.0, 1325.0, 1425.0, 1525.0};
      ASSERT_EQ(boundaries->size(), expected.size());
      auto it = boundaries->begin();
      for (double e : expected) {
         EXPECT_DOUBLE_EQ(*it++, e);
      }
   }

   // Case 2: pdfObs = RooLinearVar (slope=1, offset=shift -> xShifted = x + shift).
   {
      RooRealVar slope{"slope", "slope", 1.0};
      RooLinearVar xShifted{"x_shifted_lv", "", x, slope, shift};
      RooHistPdf pdf{"pdf_lv", "", RooArgList{xShifted}, RooArgList{x}, dh, 0};
      std::unique_ptr<std::list<double>> boundaries{pdf.binBoundaries(x, 1000.0, 1500.0)};
      ASSERT_TRUE(boundaries);
      // xShifted = x + 25 -> plot_x = b - 25; only values inside [1000, 1500] kept.
      const std::vector<double> expected{1075.0, 1175.0, 1275.0, 1375.0, 1475.0};
      ASSERT_EQ(boundaries->size(), expected.size());
      auto it = boundaries->begin();
      for (double e : expected) {
         EXPECT_DOUBLE_EQ(*it++, e);
      }
   }

   // Case 3: identity (no transformation) keeps the raw boundaries.
   {
      RooHistPdf pdf{"pdf_id", "", x, dh, 0};
      std::unique_ptr<std::list<double>> boundaries{pdf.binBoundaries(x, 1000.0, 1500.0)};
      ASSERT_TRUE(boundaries);
      const std::vector<double> expected{1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0};
      ASSERT_EQ(boundaries->size(), expected.size());
      auto it = boundaries->begin();
      for (double e : expected) {
         EXPECT_DOUBLE_EQ(*it++, e);
      }
   }
}
