// Tests for the RooHist
// Authors: Jonas Rembser, CERN 12/2022

#include <RooAbsPdf.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooGenericPdf.h>
#include <RooGlobalFunc.h>
#include <algorithm>
#include <cmath>
#include <RooHelpers.h>
#include <RooHist.h>
#include <RooHistPdf.h>
#include <RooPlot.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <TH1D.h>
#include <TH2D.h>

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

/// Check that the values returned by `RooHist::getFitRangeNEvt(double xmin,
/// double xmax)` are correct also for non-uniform binning. Covers ROOT-9649.
TEST(RooHist, GetFitRangeNEvtWithSubrange)
{
   using namespace RooFit;

   std::vector<double> binEdges{130, 140.761, 152.413, 165.03, 178.691, 193.483, 209.5, 226.842, 245.62, 265.952};

   std::size_t nBins = binEdges.size() - 1.0;
   const double xmin = binEdges.front();
   const double xmax = binEdges.back();

   std::vector<double> binCenters(nBins);

   for (std::size_t i = 0; i < binCenters.size(); i++) {
      binCenters[i] = (binEdges[i] + binEdges[i + 1]) / 2.;
   }

   std::vector<double> weights{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};

   TH1D hist{"name", "name", int(nBins), binEdges.data()};
   hist.FillN(nBins, binCenters.data(), weights.data());

   RooRealVar x("x", "x", xmin, xmax);

   RooDataHist rooDataHist("rooDataHist", "", RooArgSet(x), &hist);

   std::unique_ptr<RooPlot> frame{x.frame()};
   rooDataHist.plotOn(frame.get());

   RooHist &rooHist = *frame->getHist();

   const double nEvents = rooDataHist.sumEntries();

   EXPECT_FLOAT_EQ(rooHist.getFitRangeNEvt(), nEvents);
   EXPECT_FLOAT_EQ(rooHist.getFitRangeNEvt(x.getMin(), x.getMax()), nEvents);
}

/// RooPlot::chiSquare(), residHist() and pullHist(), validated via the
/// pull = residual / error identity and the expected chi2 behavior for an
/// exact and a deliberately distorted model, instead of comparison with stored
/// reference plots. Replaces the former stressRooFit test based on the rf109
/// tutorial.
TEST(RooHist, ResidualsAndPulls)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};
   RooRandom::randomGenerator()->SetSeed(1337);

   RooWorkspace ws;
   ws.factory("Gaussian::gauss(x[-10, 10], mean[0.0], sigma[3.0, 0.1, 10.0])");
   RooRealVar &x = *ws.var("x");
   RooRealVar &sigma = *ws.var("sigma");
   RooAbsPdf &gauss = *ws.pdf("gauss");

   std::unique_ptr<RooDataSet> data{gauss.generate(x, 10000)};

   // Frame with the model that the data was generated from
   std::unique_ptr<RooPlot> frame1{x.frame(Bins(40))};
   data->plotOn(frame1.get(), DataError(RooAbsData::SumW2));
   gauss.plotOn(frame1.get());

   const double chi2Good = frame1->chiSquare();
   EXPECT_GT(chi2Good, 0.2);
   EXPECT_LT(chi2Good, 2.0);

   // The reduced chi2 grows when the number of fit parameters is taken into
   // account in the number of degrees of freedom
   EXPECT_GT(frame1->chiSquare(2), chi2Good);

   // Frame with a slightly distorted model that must yield a larger chi2
   sigma.setVal(3.15);
   std::unique_ptr<RooPlot> frame2{x.frame(Bins(40))};
   data->plotOn(frame2.get(), DataError(RooAbsData::SumW2));
   gauss.plotOn(frame2.get());
   EXPECT_GT(frame2->chiSquare(), chi2Good);

   // The pulls must be the residuals divided by the data error of the
   // corresponding residual point
   std::unique_ptr<RooHist> hresid{frame2->residHist(nullptr, nullptr, false, false)};
   std::unique_ptr<RooHist> hpull{frame2->pullHist(nullptr, nullptr, false)};
   ASSERT_EQ(hresid->GetN(), 40);
   ASSERT_EQ(hpull->GetN(), 40);

   int nChecked = 0;
   for (int i = 0; i < hresid->GetN(); ++i) {
      EXPECT_DOUBLE_EQ(hpull->GetPointX(i), hresid->GetPointX(i));
      const double resid = hresid->GetPointY(i);
      const double err = resid > 0 ? hresid->GetErrorYlow(i) : hresid->GetErrorYhigh(i);
      if (err == 0.) {
         // Empty bins have zero sum-of-weights error, and the pull of a
         // zero-error point is defined to be zero
         EXPECT_DOUBLE_EQ(hpull->GetPointY(i), 0.) << "point " << i;
         continue;
      }
      EXPECT_NEAR(hpull->GetPointY(i), resid / err, 1e-9) << "point " << i;
      ++nChecked;
   }
   // The bulk of the Gaussian sample must have populated bins
   EXPECT_GT(nChecked, 30);
}

namespace {

// Shared setup for the exact-bin-integration tests of
// RooFit::makeResidHist()/makePullHist(): a narrowly-peaked Gaussian and a
// TH1D out of it with 50 bins

constexpr double kGaussMean = 125.;
constexpr double kGaussSigma = 1.4;
constexpr double kGaussX0 = 110.;
constexpr double kGaussX1 = 130.;
constexpr int kNBins = 50;

/// Integral of a Gaussian with (kGaussMean, kGaussSigma) between lo and hi.
double gaussIntegral(double lo, double hi)
{
   auto phi = [](double u) { return 0.5 * (1 + std::erf(u / std::sqrt(2.))); };
   return phi((hi - kGaussMean) / kGaussSigma) - phi((lo - kGaussMean) / kGaussSigma);
}

/// The Gaussian as a model plus observable; members are in construction order.
struct GaussModel {
   RooRealVar x{"x", "x", kGaussX0, kGaussX1};
   RooRealVar mean{"mean", "mean", kGaussMean};
   RooRealVar sigma{"sigma", "sigma", kGaussSigma};
   RooGenericPdf gauss{"gauss", "gauss", "exp(-0.5 * pow((x - mean) / sigma, 2))", RooArgSet{x, mean, sigma}};

   GaussModel() { x.setBins(kNBins); }
};

/// Fill the histogram with the exact Gaussian bin contents on the given
/// ranges, normalized to nEvents on those ranges. Bins outside get the
/// (constant) otherContent fill, which must not influence the in-range
/// expectations once a Range() selection is made.
void fillExactGauss(TH1D &hist, double nEvents, std::vector<std::pair<double, double>> const &ranges,
                    double otherContent = 0.)
{
   double norm = 0.;
   for (auto const &r : ranges) {
      norm += gaussIntegral(r.first, r.second);
   }
   for (int iBin = 1; iBin <= hist.GetNbinsX(); ++iBin) {
      const double lo = hist.GetBinLowEdge(iBin);
      const double hi = lo + hist.GetBinWidth(iBin);
      // epsilon in the boundary check to guard against floating point
      // arithmetic on the bin boundaries
      const bool inRange = std::any_of(ranges.begin(), ranges.end(),
                                       [&](auto const &r) { return lo >= r.first - 1e-9 && hi <= r.second + 1e-9; });
      hist.SetBinContent(iBin, inRange ? nEvents / norm * gaussIntegral(lo, hi) : otherContent);
   }
}

/// Check that all residuals in the histogram are consistent with zero, given
/// the exact per-bin contents of `hist`. The tolerance is relative to the bin
/// content, with an absolute floor to tolerate the ~1e-3 precision of the
/// numeric integration in the far tails.
void expectExactResiduals(RooHist const &resid, TH1D const &hist, double tolerance)
{
   for (int i = 0; i < resid.GetN(); ++i) {
      const int iBin = int((resid.GetPointX(i) - kGaussX0) * kNBins / (kGaussX1 - kGaussX0));
      const double content = hist.GetBinContent(iBin + 1);
      if (content == 0.)
         continue;
      EXPECT_LT(std::abs(resid.GetPointY(i)), tolerance * std::max(content, 1e-6)) << "bin " << iBin;
   }
}

} // namespace

/// RooFit::makeResidHist() and RooFit::makePullHist() must compare the data
/// with the model integrated exactly over each bin. If the model instead was
/// evaluated at the bin center, a sharply-peaked Gaussian would show large
/// residuals in the peak bins (O(10^3) events here) even for data that is
/// filled with the exact per-bin expectation values.
TEST(RooHist, MakeResidAndPullHist)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

   GaussModel model;

   TH1D hist("hist", "hist", kNBins, kGaussX0, kGaussX1);
   fillExactGauss(hist, 1e6, {{kGaussX0, kGaussX1}});
   RooDataHist binData("binData", "binData", RooArgSet(model.x), &hist);

   // The residuals of the exact model must be (numerically) zero
   std::unique_ptr<RooHist> resid{makeResidHist(model.gauss, binData)};
   std::unique_ptr<RooHist> pull{makePullHist(model.gauss, binData)};
   ASSERT_EQ(resid->GetN(), kNBins);
   ASSERT_EQ(pull->GetN(), kNBins);
   expectExactResiduals(*resid, hist, 1e-3);
   for (int i = 0; i < pull->GetN(); ++i) {
      EXPECT_LT(std::abs(pull->GetPointY(i)), 1e-3) << "bin " << i;
   }

   // The Name() and Normalization() command args
   std::unique_ptr<RooHist> scaled{makeResidHist(model.gauss, binData, Name("scaled"), Normalization(0.5))};
   ASSERT_EQ(scaled->GetN(), kNBins);
   EXPECT_STREQ(scaled->GetName(), "scaled");
   for (int i = 0; i < scaled->GetN(); ++i) {
      const double content = hist.GetBinContent(i + 1);
      if (content == 0.)
         continue;
      EXPECT_NEAR(scaled->GetPointY(i), 0.5 * content, 1e-3 * std::max(content, 1e-6)) << "bin " << i;
   }
}

/// RooFit::makePullHist() and RooFit::makeResidHist() also accept unbinned
/// input data, which is binned internally with the binning of the observable.
TEST(RooHist, MakeResidAndPullHistUnbinned)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};
   RooRandom::randomGenerator()->SetSeed(1337);

   RooRealVar x("x", "x", -10, 10);
   RooRealVar mean("mean", "mean", 0.);
   RooRealVar sigma("sigma", "sigma", 3.);
   RooGenericPdf gauss("gauss", "gauss", "exp(-0.5 * pow((x - mean) / sigma, 2))", RooArgSet{x, mean, sigma});

   std::unique_ptr<RooDataSet> data{gauss.generate(x, 10000)};

   // The Binning() argument overrides the default binning of the observable
   std::unique_ptr<RooHist> resid{makeResidHist(gauss, *data, Binning(40))};
   std::unique_ptr<RooHist> pull{makePullHist(gauss, *data, Binning(40))};
   EXPECT_EQ(resid->GetN(), 40);
   EXPECT_EQ(pull->GetN(), 40);

   // The correct model should give small pulls
   int nChecked = 0;
   for (int i = 0; i < pull->GetN(); ++i) {
      if (pull->GetErrorYhigh(i) == 0.)
         continue;
      EXPECT_LT(std::abs(pull->GetPointY(i)), 5.) << "point " << i;
      ++nChecked;
   }
   EXPECT_GT(nChecked, 30);
}

/// The Range() argument of RooFit::makeResidHist()/makePullHist(): only bins
/// inside the range get points, and the model is normalized to the data
/// weight in the range (as needed for sideband fits).
TEST(RooHist, MakeResidAndPullHistRange)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

   GaussModel model;

   // [rlo, rhi] chosen on bin boundaries: bins 24..33 of 50. The asymmetric
   // range contains only ~16% of the model, so a wrong range normalization
   // would produce huge residuals.
   const double rlo = 119.6, rhi = 123.6;

   TH1D hist("hist", "hist", kNBins, kGaussX0, kGaussX1);
   // The constant content outside of the range must not influence the
   // expectation normalization, which only counts the data weight in the range
   fillExactGauss(hist, 1e6, {{rlo, rhi}}, 2.e6);
   RooDataHist binData("binData", "binData", RooArgSet(model.x), &hist);

   // With coordinate range. RooGenericPdf is integrated numerically, whose
   // precision is limited to ~1e-3 relative in the far tails, hence the 5e-3
   // tolerance.
   std::unique_ptr<RooHist> resid{makeResidHist(model.gauss, binData, Range(rlo, rhi))};
   EXPECT_EQ(resid->GetN(), 10);
   expectExactResiduals(*resid, hist, 5e-3);

   // With named range
   model.x.setRange("central", rlo, rhi);
   std::unique_ptr<RooHist> resid2{makeResidHist(model.gauss, binData, Range("central"))};
   ASSERT_EQ(resid2->GetN(), resid->GetN());
   for (int i = 0; i < resid2->GetN(); ++i) {
      EXPECT_DOUBLE_EQ(resid2->GetPointX(i), resid->GetPointX(i));
      EXPECT_DOUBLE_EQ(resid2->GetPointY(i), resid->GetPointY(i));
   }

   // An unknown named range must throw
   EXPECT_THROW(makeResidHist(model.gauss, binData, Range("unknown")), std::invalid_argument);
}

/// The Range("name") argument of RooFit::makeResidHist() can be passed
/// multiple times for a union of named ranges (sidebands).
TEST(RooHist, MakeResidAndPullHistMultiRange)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

   GaussModel model;

   // Bins 13..16 and 33..36 (on bin boundaries)
   model.x.setRange("sb1", 115.2, 116.8);
   model.x.setRange("sb2", 123.2, 124.8);

   TH1D hist("hist", "hist", kNBins, kGaussX0, kGaussX1);
   fillExactGauss(hist, 1e6, {{115.2, 116.8}, {123.2, 124.8}}, 3.e6);
   RooDataHist binData("binData", "binData", RooArgSet(model.x), &hist);

   std::unique_ptr<RooHist> resid{makeResidHist(model.gauss, binData, Range("sb1"), Range("sb2"))};
   ASSERT_EQ(resid->GetN(), 8);
   expectExactResiduals(*resid, hist, 5e-3);
}

/// The documented failure modes of RooFit::makeResidHist() and
/// RooFit::makePullHist() that throw exceptions.
TEST(RooHist, MakeResidAndPullHistErrors)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

   RooRealVar x("x", "x", -10, 10);
   RooRealVar y("y", "y", -10, 10);
   x.setBins(40);
   RooRealVar mean("mean", "mean", 0.);
   RooRealVar sigma("sigma", "sigma", 3.);
   RooGenericPdf gauss("gauss", "gauss", "exp(-0.5 * pow((x - mean) / sigma, 2))", RooArgSet{x, mean, sigma});

   RooRandom::randomGenerator()->SetSeed(1337);
   std::unique_ptr<RooDataSet> data{gauss.generate(x, 1000)};

   // Multi-dimensional data
   TH2D hist2("hist2", "hist2", 4, -10, 10, 4, -10, 10);
   hist2.Fill(0., 0.);
   RooDataHist hist2D("hist2D", "hist2D", RooArgSet(x, y), &hist2);
   EXPECT_THROW(makeResidHist(gauss, hist2D), std::invalid_argument);

   // Empty datasets
   RooDataSet emptyData("emptyData", "emptyData", RooArgSet(x));
   EXPECT_THROW(makeResidHist(gauss, emptyData), std::invalid_argument);

   // Binning() arguments with already binned data
   RooDataHist binned("binned", "binned", RooArgSet(x), *data);
   EXPECT_THROW(makeResidHist(gauss, binned, Binning(20)), std::invalid_argument);

   // Unknown named ranges
   EXPECT_THROW(makeResidHist(gauss, *data, Range("unknown")), std::invalid_argument);

   // Unsupported Normalization scale types
   EXPECT_THROW(makeResidHist(gauss, *data, Normalization(0.5, RooAbsReal::NumEvent)), std::invalid_argument);

   // Unsupported DataError types
   EXPECT_THROW(makeResidHist(gauss, *data, DataError(99)), std::invalid_argument);

   // A model with no support in the range must throw
   RooGenericPdf zeroModel("zeroModel", "zeroModel", "x * 0.", RooArgSet{x});
   EXPECT_THROW(makeResidHist(zeroModel, *data), std::invalid_argument);
}
