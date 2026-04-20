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

#include <cmath>
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

namespace {

// Narrow Gaussian peak (sigma smaller than the bin width) on a steeply falling
// exponential, so the sub-range area genuinely depends on the interpolation
// order (a shape well sampled by the binning would let the bug below slip
// through).
void fillPeakedHist(RooDataHist &dataHist, RooRealVar &xHist)
{
   double const mean = 100.;
   double const sigma = 8.;
   for (int i = 0; i < xHist.numBins(); ++i) {
      xHist.setBin(i);
      double const c = xHist.getVal();
      double const d = (c - mean) / sigma;
      dataHist.set(i, 5000. * std::exp(-0.5 * d * d) + 3000. * std::exp(-0.05 * c) + 10., 0.0);
   }
}

// Integral of the raw RooHistPdf curve over the full range of "x", sampling
// pdf.getVal() directly. Independent of the integration machinery under test and
// correct for any interpolation order.
double trapezoidalReference(RooHistPdf &pdf, RooRealVar &x)
{
   int const n = 200000;
   double const lo = x.getMin();
   double const hi = x.getMax();
   double const h = (hi - lo) / n;
   double sum = 0.0;
   for (int i = 0; i <= n; ++i) {
      x.setVal(lo + i * h);
      double const w = (i == 0 || i == n) ? 0.5 : 1.0;
      sum += w * pdf.getVal();
   }
   return sum * h;
}

} // namespace

// The analytical integral (RooDataHist::sum()) integrates the piecewise-constant
// histogram, which only matches the interpolated curve over the full range. Over
// a sub-range with intOrder > 0 RooHistPdf used to take that shortcut anyway,
// returning the intOrder == 0 area regardless of the interpolation order.
//
// This checks the integral *value* (against an independent reference) rather
// than which strategy is used, so it stays valid both for the current numerical
// fallback and for a possible future analytical sub-range integral.
TEST(RooHistPdf, SubRangeIntegralWithInterpolation)
{
   // Histogram observable, wider than the pdf observable "x" below.
   RooRealVar xHist{"xHist", "xHist", -30., 270.};
   xHist.setBins(20);

   RooDataHist dataHist{"dataHist", "dataHist", xHist};
   fillPeakedHist(dataHist, xHist);

   // The pdf observable "x" spans only a sub-range of the histogram.
   RooRealVar x{"x", "x", 0., 200.};

   double refByOrder[2] = {0., 0.};

   for (int intOrder : {0, 1}) {
      RooHistPdf pdf{"pdf", "pdf", x, xHist, dataHist, intOrder};
      double const ref = trapezoidalReference(pdf, x);
      refByOrder[intOrder] = ref;

      std::unique_ptr<RooAbsReal> integ{pdf.createIntegral(x)};
      EXPECT_NEAR(integ->getVal(), ref, 1e-3 * ref) << "Wrong sub-range integral for intOrder=" << intOrder;
   }

   // Guard against the histogram shape becoming too smooth to tell the
   // interpolation orders apart, which would make this test insensitive.
   EXPECT_GT(std::abs(refByOrder[1] - refByOrder[0]), 0.02 * refByOrder[1])
      << "Test histogram is too smooth to distinguish interpolation orders";
}

// Analytic integration is used in the cases where RooDataHist::sum() is exact:
// non-interpolated histograms (any range) and full-range integrals (any order).
TEST(RooHistPdf, AnalyticIntForStepFunctionAndFullRange)
{
   RooRealVar xHist{"xHist", "xHist", -30., 270.};
   xHist.setBins(20);

   RooDataHist dataHist{"dataHist", "dataHist", xHist};
   fillPeakedHist(dataHist, xHist);

   // Sub-range observable (narrower than the histogram).
   RooRealVar xSub{"xSub", "xSub", 0., 200.};
   // Full-range observable (matches the histogram range).
   RooRealVar xFull{"xFull", "xFull", -30., 270.};

   // Sub-range, no interpolation: analytical (step function is exact).
   {
      RooHistPdf pdf{"pdf", "pdf", xSub, xHist, dataHist, 0};
      RooRealIntegral integ{"integ", "integ", pdf, xSub};
      integ.getVal();
      EXPECT_EQ(integ.anaIntVars().size(), 1) << "intOrder=0 sub-range should be analytical";
   }

   // Full-range, with interpolation: analytical (area is conserved over the full range).
   {
      RooHistPdf pdf{"pdf", "pdf", xFull, xHist, dataHist, 1};
      RooRealIntegral integ{"integ", "integ", pdf, xFull};
      integ.getVal();
      EXPECT_EQ(integ.anaIntVars().size(), 1) << "intOrder=1 full-range should be analytical";
   }
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

// GitHub issue #13030: a non-linear transformation of the plot observable
// cannot be inverted exactly by the (linear) boundary-finding algorithm. In
// that case the boundaries are still returned, but only approximately, and the
// user must be warned that the plotted curve may be placed incorrectly.
TEST(RooHistPdf, NonLinearBinBoundariesWarn)
{
   RooRealVar x{"x", "x", 1.0, 5.0};
   x.setBins(4);

   TH1D h{"h", "", x.numBins(), x.getMin(), x.getMax()};
   for (int i = 1; i <= h.GetNbinsX(); ++i) {
      h.SetBinContent(i, 1.0);
   }
   RooDataHist dh{"dh", "", x, &h};

   // A genuinely non-linear transformation (not an l-value, so it goes through
   // the linear-approximation path).
   RooFormulaVar xSquared{"x_squared", "x * x", x};
   RooHistPdf pdf{"pdf_nl", "", xSquared, x, dh, 0};

   std::unique_ptr<std::list<double>> boundaries;
   std::string msg;
   {
      RooHelpers::HijackMessageStream hijack{RooFit::WARNING, RooFit::Plotting};
      boundaries.reset(pdf.binBoundaries(x, x.getMin(), x.getMax()));
      msg = hijack.str();
   }

   // Boundaries are still produced (approximately), and a warning was emitted.
   ASSERT_TRUE(boundaries);
   EXPECT_FALSE(boundaries->empty());
   EXPECT_TRUE(msg.find("non-linear") != std::string::npos)
      << "Expected a non-linearity warning, but got: '" << msg << "'";
}
