// Tests for the RooHistPdf
// Authors: Jonas Rembser, CERN 03/2023

#include <RooCategory.h>
#include <RooDataHist.h>
#include <RooFitResult.h>
#include <RooFormulaVar.h>
#include <RooHelpers.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooLinearVar.h>
#include <RooRealIntegral.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <TH1D.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <list>
#include <map>
#include <memory>
#include <vector>

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

// A RooHistPdf clamps negative bin contents to zero when evaluating, but the
// normalization used to be based on the raw sum of weights including the
// negative ones, so the pdf did not integrate to unity and extended fits were
// silently biased (ROOT-10825). Now, negative bins are zeroed in an
// internally-owned clone of the histogram at construction time, making the
// shape and the normalization consistent.
TEST(RooHistPdf, NegativeBinsAreClampedConsistently)
{
   RooRealVar x{"x", "x", 0, 4};
   x.setBins(4);

   RooDataHist dataHist{"dataHist", "dataHist", x};
   double const contents[4] = {4., 3., -2., 5.};
   double const sumClamped = 12.;
   for (int i = 0; i < x.numBins(); ++i) {
      dataHist.set(i, contents[i], 1.0);
   }

   RooHelpers::HijackMessageStream hijack{RooFit::WARNING, RooFit::InputArguments, "pdf"};
   RooHistPdf pdf{"pdf", "pdf", x, dataHist};
   EXPECT_FALSE(hijack.str().empty()) << "constructing from a histogram with negative bins should warn";

   // The negative bin is dropped from both the pdf value and the
   // normalization, so the pdf values are max(w, 0) / sum(max(w, 0)) and the
   // pdf integrates to unity.
   RooArgSet normSet{x};
   double integral = 0.;
   for (int i = 0; i < x.numBins(); ++i) {
      x.setBin(i);
      EXPECT_DOUBLE_EQ(pdf.getVal(normSet), std::max(contents[i], 0.) / sumClamped) << "wrong value in bin " << i;
      integral += pdf.getVal(normSet) * x.getBinWidth(i);
   }
   EXPECT_DOUBLE_EQ(integral, 1.);

   // The input histogram must not be modified.
   for (int i = 0; i < x.numBins(); ++i) {
      EXPECT_DOUBLE_EQ(dataHist.weight(i), contents[i]) << "input histogram was modified in bin " << i;
   }

   // Same behavior for the constructor where the RooHistPdf takes ownership
   // of the input histogram.
   RooHistPdf pdf2{"pdf2", "pdf2", x, std::make_unique<RooDataHist>(dataHist, "dataHist2")};
   for (int i = 0; i < x.numBins(); ++i) {
      x.setBin(i);
      EXPECT_DOUBLE_EQ(pdf2.getVal(normSet), std::max(contents[i], 0.) / sumClamped) << "wrong value in bin " << i;
   }
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

// A binned likelihood of RooHistPdf templates in a RooSimultaneous crashed,
// because the RooHistPdf didn't map the prefixed channel observable to its
// histogram observable when asked for the bin boundaries.
TEST(RooHistPdf, BinnedLikelihoodInSimultaneous)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;
   auto &x = static_cast<RooRealVar &>(*ws.factory("x[0, 10]"));
   x.setBins(10);
   auto &index = static_cast<RooCategory &>(*ws.factory("index[A]"));

   RooDataHist dhSig{"dhSig", "dhSig", RooArgSet{x}};
   RooDataHist dhBkg{"dhBkg", "dhBkg", RooArgSet{x}};
   RooDataHist dataA{"dataA", "dataA", RooArgSet{x}};
   for (int i = 0; i < x.numBins(); ++i) {
      dhSig.set(i, i, -1);
      dhBkg.set(i, 1, -1);
      dataA.set(i, i + 1, -1);
   }
   ws.import(dhSig);
   ws.import(dhBkg);
   ws.factory("HistPdf::sig(x, dhSig)");
   ws.factory("HistPdf::bkg(x, dhBkg)");
   ws.factory("ASUM::modelA(mu_sig[1, 0, 10] * sig, mu_bkg[1, 0, 10] * bkg)");
   RooAbsPdf &modelA = *ws.pdf("modelA");
   modelA.setAttribute("BinnedLikelihood");
   ws.factory("SIMUL::simPdf(index, A=modelA)");

   RooDataHist data{"data", "data", RooArgList{x}, index, std::map<std::string, RooDataHist *>{{"A", &dataA}}};

   std::unique_ptr<RooAbsReal> nllSim{ws.pdf("simPdf")->createNLL(data)};
   std::unique_ptr<RooAbsReal> nllA{modelA.createNLL(dataA)};

   EXPECT_DOUBLE_EQ(nllSim->getVal(), nllA->getVal());
}

namespace {

// Takes ownership of a list returned by binBoundaries() or plotSamplingHint().
std::vector<double> toVector(std::list<double> *list)
{
   std::unique_ptr<std::list<double>> owned{list};
   if (!owned) {
      return {};
   }
   return {owned->begin(), owned->end()};
}

} // namespace

// GitHub issue #13030: plotting a RooHistPdf whose observable is a
// transformation of the plot variable (e.g. a shift `x_shifted = x - shift`)
// produced visually broken curves because `binBoundaries` and
// `plotSamplingHint` returned the raw histogram boundaries in the histogram
// observable coordinate instead of the plot observable coordinate.
TEST(RooHistPdf, ShiftedBinBoundaries)
{
   using Vec = std::vector<double>;

   RooRealVar x{"x", "x", 1000, 1500};
   x.setBins(5); // bin edges in x: 1000, 1100, 1200, 1300, 1400, 1500

   RooRealVar shift{"shift", "shift", 25.0, -100, 100};

   TH1D h{"h", "", x.numBins(), x.getMin(), x.getMax()};
   for (int i = 1; i <= h.GetNbinsX(); ++i) {
      h.SetBinContent(i, 1.0);
   }
   RooDataHist dh{"dh", "", x, &h};

   // pdfObs = x - shift as a RooFormulaVar (no l-value inverse). Hist bin
   // boundary `b` satisfies `x - shift = b`, so plot_x = b + shift.
   RooFormulaVar xShifted{"x_shifted", "x - shift", {x, shift}};
   RooHistPdf pdfF{"pdf_f", "", xShifted, x, dh, 0};
   EXPECT_EQ(toVector(pdfF.binBoundaries(x, 1000.0, 1600.0)), (Vec{1025, 1125, 1225, 1325, 1425, 1525}));

   // pdfObs = RooLinearVar x + shift -> plot_x = b - 25; only values inside [1000, 1500] kept.
   RooRealVar slope{"slope", "slope", 1.0};
   RooLinearVar xShiftedLV{"x_shifted_lv", "", x, slope, shift};
   RooHistPdf pdfLV{"pdf_lv", "", RooArgList{xShiftedLV}, RooArgList{x}, dh, 0};
   EXPECT_EQ(toVector(pdfLV.binBoundaries(x, 1000.0, 1500.0)), (Vec{1075, 1175, 1275, 1375, 1475}));

   // Negative slope: plot_x = 2525 - b, still returned in increasing order.
   RooFormulaVar xMirrored{"x_mirrored", "2525 - x", {x}};
   RooHistPdf pdfM{"pdf_m", "", xMirrored, x, dh, 0};
   EXPECT_EQ(toVector(pdfM.binBoundaries(x, 1000.0, 1500.0)), (Vec{1025, 1125, 1225, 1325, 1425}));

   // Identity (no transformation) keeps the raw boundaries.
   RooHistPdf pdfId{"pdf_id", "", x, dh, 0};
   EXPECT_EQ(toVector(pdfId.binBoundaries(x, 1000.0, 1500.0)), (Vec{1000, 1100, 1200, 1300, 1400, 1500}));

   // RooHistFunc uses the same mapping.
   RooHistFunc funcF{"func_f", "", RooArgList{xShifted}, RooArgList{x}, dh, 0};
   RooHistFunc funcLV{"func_lv", "", RooArgList{xShiftedLV}, RooArgList{x}, dh, 0};
   EXPECT_EQ(toVector(funcF.binBoundaries(x, 1000.0, 1500.0)), (Vec{1025, 1125, 1225, 1325, 1425}));
   EXPECT_EQ(toVector(funcLV.binBoundaries(x, 1000.0, 1500.0)), (Vec{1075, 1175, 1275, 1375, 1475}));

   // The sampling hint has a point next to each plot limit and a pair of
   // points around each boundary inside the plot range.
   const Vec hint = toVector(pdfF.plotSamplingHint(x, 1000.0, 1500.0));
   ASSERT_EQ(hint.size(), 12u);
   for (std::size_t i = 0; i < 5; ++i) {
      EXPECT_NEAR(0.5 * (hint[2 * i + 1] + hint[2 * i + 2]), 1025.0 + 100.0 * i, 1e-9);
   }
}

// The plot observable can hold a value outside of its default range, e.g. set
// by an outer integral over a wider range. Mapping the bin boundaries must
// restore that value without throwing.
TEST(RooHistPdf, ShiftedBinBoundariesKeepOutOfRangeValue)
{
   RooRealVar x{"x", "x", 1000, 1500};
   x.setBins(5);
   x.setRange("wide", 0, 2000);
   RooRealVar shift{"shift", "shift", 25.0, -100, 100};

   RooDataHist dh{"dh", "", x};
   RooFormulaVar xShifted{"x_shifted", "x - shift", {x, shift}};
   RooHistPdf pdf{"pdf", "", xShifted, x, dh, 0};

   x.setVal(500, "wide");
   ASSERT_EQ(x.getVal(), 500.0);
   EXPECT_EQ(toVector(pdf.binBoundaries(x, 1000.0, 1500.0)), (std::vector<double>{1025, 1125, 1225, 1325, 1425}));
   EXPECT_EQ(x.getVal(), 500.0);
}

// The numeric integral over a shifted RooHistPdf has to follow the shift. The
// RooBinIntegrator used to keep the bin boundaries of its first evaluation.
TEST(RooHistPdf, ShiftedIntegralFollowsShift)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRealVar x{"x", "x", 1000, 1500};
   x.setBins(5);
   RooRealVar shift{"shift", "shift", 25.0, -100, 100};

   const double contents[]{1., 2., 4., 8., 16.};
   TH1D h{"h", "", x.numBins(), x.getMin(), x.getMax()};
   for (int i = 0; i < 5; ++i) {
      h.SetBinContent(i + 1, contents[i]);
   }
   RooDataHist dh{"dh", "", x, &h};

   RooFormulaVar xShifted{"x_shifted", "x - shift", {x, shift}};
   RooHistPdf pdf{"pdf", "", xShifted, x, dh, 0};

   // Overlap of each shifted bin with [1000, 1500], times the bin density.
   auto expected = [&](double s) {
      double sum = 0.;
      for (int i = 0; i < 5; ++i) {
         const double lo = std::max(1000. + 100. * i + s, 1000.);
         const double hi = std::min(1100. + 100. * i + s, 1500.);
         sum += std::max(hi - lo, 0.) * contents[i] / 100.;
      }
      return sum;
   };

   std::unique_ptr<RooAbsReal> integral{pdf.createIntegral(x)};
   for (double s : {25., -30., 0., 60.}) {
      shift.setVal(s);
      EXPECT_NEAR(integral->getVal(), expected(s), 1e-10) << "shift = " << s;
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
