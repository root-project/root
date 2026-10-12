// Tests for the RooHist
// Authors: Jonas Rembser, CERN 12/2022

#include <RooAbsPdf.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooExtendPdf.h>
#include <RooGenericPdf.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooHist.h>
#include <RooHistPdf.h>
#include <RooProdPdf.h>
#include <RooPlot.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <TH1D.h>
#include <TH2D.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

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

/// The exact Gaussian as a RooDataHist with kNBins bins, normalized to unity.
RooDataHist makeExactGaussDataHist(RooRealVar &x)
{
   TH1D hist("modelHist", "modelHist", kNBins, kGaussX0, kGaussX1);
   fillExactGauss(hist, 1., {{kGaussX0, kGaussX1}});
   return RooDataHist{"modelDataHist", "modelDataHist", RooArgSet(x), &hist};
}

/// The Gaussian as a model plus observable; members are in construction order.
/// The model exists in two variants: a RooHistPdf with the exact per-bin
/// Gaussian integrals as bin contents, whose integrals over bins and ranges
/// are analytic (partial sums of the histogram), and a RooGenericPdf that is
/// integrated numerically.
struct GaussModel {
   RooRealVar x{"x", "x", kGaussX0, kGaussX1};
   RooRealVar mean{"mean", "mean", kGaussMean};
   RooRealVar sigma{"sigma", "sigma", kGaussSigma};
   RooDataHist modelDataHist{makeExactGaussDataHist(x)};
   RooHistPdf gaussAnalytic{"gaussAnalytic", "gaussAnalytic", RooArgSet(x), modelDataHist};
   RooGenericPdf gaussNumeric{"gaussNumeric", "gaussNumeric", "exp(-0.5 * pow((x - mean) / sigma, 2))",
                              RooArgSet{x, mean, sigma}};

   GaussModel() { x.setBins(kNBins); }
};

/// Tolerance (relative to the bin content) for the residuals of the exact
/// model. With the analytic integrals of the RooHistPdf, the comparison is
/// limited only by floating-point precision.
constexpr double kTolAnalytic = 1e-10;
/// For the numerically integrated RooGenericPdf, the relative precision of
/// the per-bin integrals is limited to ~1e-3 in the far tails.
constexpr double kTolNumeric = 1e-3;

/// Check that all residuals in the histogram are consistent with zero, given
/// the exact per-bin contents of `hist`. The tolerance is relative to the bin
/// content, with an absolute floor for the nearly empty bins in the far tails.
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

   // The residuals of the exact model must be zero up to the precision of
   // the model integration: floating-point precision for the analytic
   // integrals of the RooHistPdf, and the numeric integration precision for
   // the RooGenericPdf.
   for (auto const &[pdf, tol] : {std::pair<RooAbsPdf *, double>{&model.gaussAnalytic, kTolAnalytic},
                                  std::pair<RooAbsPdf *, double>{&model.gaussNumeric, kTolNumeric}}) {
      std::unique_ptr<RooHist> resid{makeResidHist(*pdf, binData)};
      std::unique_ptr<RooHist> pull{makePullHist(*pdf, binData)};
      ASSERT_EQ(resid->GetN(), kNBins);
      ASSERT_EQ(pull->GetN(), kNBins);
      expectExactResiduals(*resid, hist, tol);
      for (int i = 0; i < pull->GetN(); ++i) {
         EXPECT_LT(std::abs(pull->GetPointY(i)), tol) << pdf->GetName() << " bin " << i;
      }
   }

   // The Name() and Normalization() command args
   std::unique_ptr<RooHist> scaled{makeResidHist(model.gaussAnalytic, binData, Name("scaled"), Normalization(0.5))};
   ASSERT_EQ(scaled->GetN(), kNBins);
   EXPECT_STREQ(scaled->GetName(), "scaled");
   for (int i = 0; i < scaled->GetN(); ++i) {
      const double content = hist.GetBinContent(i + 1);
      if (content == 0.)
         continue;
      EXPECT_NEAR(scaled->GetPointY(i), 0.5 * content, kTolAnalytic * std::max(content, 1e-6)) << "bin " << i;
   }

   // The scratch range used for the per-bin integration must not be left on
   // the observable
   for (auto const &name : model.x.getBinningNames()) {
      EXPECT_EQ(name.find("_binInt_"), std::string::npos) << "leftover range " << name;
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
   const int nBins = 40;
   std::unique_ptr<RooHist> resid{makeResidHist(gauss, *data, Binning(nBins))};
   std::unique_ptr<RooHist> pull{makePullHist(gauss, *data, Binning(nBins))};
   EXPECT_EQ(resid->GetN(), nBins);
   EXPECT_EQ(pull->GetN(), nBins);

   // The pulls must be the residuals divided by the data error of the
   // corresponding residual point
   for (int i = 0; i < pull->GetN(); ++i) {
      EXPECT_DOUBLE_EQ(pull->GetPointX(i), resid->GetPointX(i));
      const double r = resid->GetPointY(i);
      const double err = r > 0 ? resid->GetErrorYlow(i) : resid->GetErrorYhigh(i);
      if (err == 0.) {
         EXPECT_DOUBLE_EQ(pull->GetPointY(i), 0.) << "point " << i;
         continue;
      }
      EXPECT_NEAR(pull->GetPointY(i), r / err, 1e-9) << "point " << i;
   }

   // The pulls of the true model must be standard-normal like: no outliers
   // (the seed is fixed), and a chi2/ndf around one. The chi2 is a strong
   // check of the per-bin expectations, since with a wrong normalization or
   // bin-center evaluation it would be far off.
   int nChecked = 0;
   double chi2 = 0.;
   for (int i = 0; i < pull->GetN(); ++i) {
      if (pull->GetErrorYhigh(i) == 0.)
         continue;
      const double p = pull->GetPointY(i);
      EXPECT_LT(std::abs(p), 3.5) << "point " << i;
      chi2 += p * p;
      ++nChecked;
   }
   EXPECT_GT(nChecked, 30);
   EXPECT_GT(chi2 / nChecked, 0.5);
   EXPECT_LT(chi2 / nChecked, 1.5);

   // Without Binning(), the current binning of the observable is used, not
   // the binning stored in the dataset at the time of its creation
   x.setBins(25);
   std::unique_ptr<RooHist> residDefault{makeResidHist(gauss, *data)};
   EXPECT_EQ(residDefault->GetN(), 25);
}

/// With Binning(nbins, xlo, xhi) narrower than the observable range, the
/// entries outside of the binning range must be dropped (and not clamped
/// into the edge bins, as RooDataHist::add() does).
TEST(RooHist, MakeResidAndPullHistBinningRange)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};
   RooRandom::randomGenerator()->SetSeed(1337);

   RooRealVar x("x", "x", -10, 10);
   RooRealVar mean("mean", "mean", 0.);
   RooRealVar sigma("sigma", "sigma", 3.);
   RooGenericPdf gauss("gauss", "gauss", "exp(-0.5 * pow((x - mean) / sigma, 2))", RooArgSet{x, mean, sigma});

   std::unique_ptr<RooDataSet> data{gauss.generate(x, 10000)};

   const int nBins = 20;
   const double xlo = -2.;
   const double xhi = 2.;

   // With Normalization(0.), the model expectation is zero and the residuals
   // are the binned data contents
   std::unique_ptr<RooHist> resid{makeResidHist(gauss, *data, Binning(nBins, xlo, xhi), Normalization(0.))};
   ASSERT_EQ(resid->GetN(), nBins);

   TH1D ref("ref", "ref", nBins, xlo, xhi);
   for (int i = 0; i < data->numEntries(); ++i) {
      ref.Fill(static_cast<RooAbsReal const &>(*data->get(i)->find("x")).getVal());
   }
   EXPECT_LT(ref.Integral(), 0.6 * data->numEntries()); // a large fraction of the data is outside the range
   for (int i = 0; i < nBins; ++i) {
      EXPECT_DOUBLE_EQ(resid->GetPointX(i), ref.GetBinCenter(i + 1)) << "bin " << i;
      EXPECT_DOUBLE_EQ(resid->GetPointY(i), ref.GetBinContent(i + 1)) << "bin " << i;
   }
}

/// Composite models cache the integrals of their components by range name,
/// and the integration strategy of a RooHistPdf (full sum vs. partial sum)
/// is fixed when the integral is created. The per-bin integration must not
/// be affected by this: a RooHistPdf wrapped in a RooProdPdf or projected
/// with createProjection() must give the same residuals as the bare
/// RooHistPdf.
TEST(RooHist, MakeResidAndPullHistWrappedModel)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

   GaussModel model;
   RooRealVar y("y", "y", -1., 1.);
   RooGenericPdf gy("gy", "gy", "1 + y * y", RooArgSet{y});
   RooProdPdf prod("prod", "prod", RooArgList{model.gaussAnalytic, gy});
   std::unique_ptr<RooAbsPdf> proj{prod.createProjection(RooArgSet{y})};

   TH1D hist("hist", "hist", kNBins, kGaussX0, kGaussX1);
   fillExactGauss(hist, 1e6, {{kGaussX0, kGaussX1}});
   RooDataHist binData("binData", "binData", RooArgSet(model.x), &hist);

   for (RooAbsPdf *pdf : {static_cast<RooAbsPdf *>(&prod), proj.get()}) {
      std::unique_ptr<RooHist> resid{makeResidHist(*pdf, binData)};
      ASSERT_EQ(resid->GetN(), kNBins) << pdf->GetName();
      expectExactResiduals(*resid, hist, kTolAnalytic);
      // Also with a sub-range, and in a second call with the same model
      std::unique_ptr<RooHist> residRange{makeResidHist(*pdf, binData, Range(119.6, 123.6))};
      ASSERT_EQ(residRange->GetN(), 10) << pdf->GetName();
      expectExactResiduals(*residRange, hist, kTolAnalytic);
   }

   // The user's variable must not keep any scratch range
   for (auto const &name : model.x.getBinningNames()) {
      EXPECT_EQ(name.find("_binInt_"), std::string::npos) << "leftover range " << name;
   }
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

   // With coordinate range
   std::unique_ptr<RooHist> resid{makeResidHist(model.gaussAnalytic, binData, Range(rlo, rhi))};
   EXPECT_EQ(resid->GetN(), 10);
   expectExactResiduals(*resid, hist, kTolAnalytic);

   // Same with the numerically integrated model
   std::unique_ptr<RooHist> residNumeric{makeResidHist(model.gaussNumeric, binData, Range(rlo, rhi))};
   EXPECT_EQ(residNumeric->GetN(), 10);
   expectExactResiduals(*residNumeric, hist, kTolNumeric);

   // With a range that is not aligned with the bin edges, only the bins
   // fully inside the range are used, and the expectation is normalized to
   // the data in these bins (so the residuals of the exact data are still
   // zero)
   std::unique_ptr<RooHist> residUnaligned{makeResidHist(model.gaussAnalytic, binData, Range(rlo - 0.1, rhi + 0.1))};
   EXPECT_EQ(residUnaligned->GetN(), 10);
   expectExactResiduals(*residUnaligned, hist, kTolAnalytic);

   // With named range
   model.x.setRange("central", rlo, rhi);
   std::unique_ptr<RooHist> resid2{makeResidHist(model.gaussAnalytic, binData, Range("central"))};
   ASSERT_EQ(resid2->GetN(), resid->GetN());
   for (int i = 0; i < resid2->GetN(); ++i) {
      EXPECT_DOUBLE_EQ(resid2->GetPointX(i), resid->GetPointX(i));
      EXPECT_DOUBLE_EQ(resid2->GetPointY(i), resid->GetPointY(i));
   }

   // An unknown named range must throw
   EXPECT_THROW(makeResidHist(model.gaussAnalytic, binData, Range("unknown")), std::invalid_argument);
}

/// The Range() argument of RooFit::makeResidHist() accepts multiple
/// comma-separated range names for a union of named ranges (sidebands).
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

   std::unique_ptr<RooHist> resid{makeResidHist(model.gaussAnalytic, binData, Range("sb1,sb2"))};
   ASSERT_EQ(resid->GetN(), 8);
   expectExactResiduals(*resid, hist, kTolAnalytic);

   // Overlapping ranges must not double count the model expectation: bins
   // 13..18 with the overlapping ranges sb1 and [116.0, 117.6]
   model.x.setRange("sbo", 116.0, 117.6);
   TH1D hist2("hist2", "hist2", kNBins, kGaussX0, kGaussX1);
   fillExactGauss(hist2, 1e6, {{115.2, 117.6}}, 3.e6);
   RooDataHist binData2("binData2", "binData2", RooArgSet(model.x), &hist2);
   std::unique_ptr<RooHist> residOverlap{makeResidHist(model.gaussAnalytic, binData2, Range("sb1,sbo"))};
   ASSERT_EQ(residOverlap->GetN(), 6);
   expectExactResiduals(*residOverlap, hist2, kTolAnalytic);
}

/// The scale types of the Normalization() argument of
/// RooFit::makeResidHist()/makePullHist(), following RooAbsPdf::plotOn().
TEST(RooHist, MakeResidAndPullHistNormalization)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

   GaussModel model;
   const double nEvents = 1e6;
   TH1D hist("hist", "hist", kNBins, kGaussX0, kGaussX1);
   fillExactGauss(hist, nEvents, {{kGaussX0, kGaussX1}});
   RooDataHist binData("binData", "binData", RooArgSet(model.x), &hist);

   // Checks that the residuals are (-factor) times the exact bin contents,
   // i.e. the expectation is (1 + factor) times the data
   auto expectScaledResiduals = [&](RooHist const &resid, double factor, double tol) {
      for (int i = 0; i < resid.GetN(); ++i) {
         const int iBin = int((resid.GetPointX(i) - kGaussX0) * kNBins / (kGaussX1 - kGaussX0));
         const double content = hist.GetBinContent(iBin + 1);
         if (content == 0.)
            continue;
         EXPECT_NEAR(resid.GetPointY(i), -factor * content, tol * std::max(content, 1e-6)) << "bin " << iBin;
      }
   };

   // An extended pdf with a yield of 1.5 times the data
   RooRealVar nExpected("nExpected", "nExpected", 1.5 * nEvents);
   RooExtendPdf ext("ext", "ext", model.gaussAnalytic, nExpected);

   // The default Relative scale type normalizes to the data also for an
   // extended pdf
   std::unique_ptr<RooHist> resid{makeResidHist(ext, binData)};
   ASSERT_EQ(resid->GetN(), kNBins);
   expectScaledResiduals(*resid, 0., kTolAnalytic);

   // RelativeExpected: the expectation is the expected number of events
   std::unique_ptr<RooHist> residExp{makeResidHist(ext, binData, Normalization(1.0, RooAbsReal::RelativeExpected))};
   ASSERT_EQ(residExp->GetN(), kNBins);
   expectScaledResiduals(*residExp, 0.5, kTolAnalytic);

   // Also in a sub-range, the expectation is not renormalized to the data
   std::unique_ptr<RooHist> residExpRange{
      makeResidHist(ext, binData, Normalization(2.0, RooAbsReal::RelativeExpected), Range(119.6, 123.6))};
   ASSERT_EQ(residExpRange->GetN(), 10);
   expectScaledResiduals(*residExpRange, 2., kTolAnalytic);

   // NumEvent: the expectation is normalized to the given number of events
   std::unique_ptr<RooHist> residNum{
      makeResidHist(model.gaussAnalytic, binData, Normalization(2. * nEvents, RooAbsReal::NumEvent))};
   ASSERT_EQ(residNum->GetN(), kNBins);
   expectScaledResiduals(*residNum, 1., kTolAnalytic);

   // Raw: the integral of the unnormalized model. The RooGenericPdf
   // exp(-0.5 u^2) integrates to sigma * sqrt(2 pi) times the Gaussian
   // probability in each bin.
   const double rawScale = nEvents / (kGaussSigma * std::sqrt(2. * M_PI));
   std::unique_ptr<RooHist> residRaw{
      makeResidHist(model.gaussNumeric, binData, Normalization(rawScale, RooAbsReal::Raw))};
   ASSERT_EQ(residRaw->GetN(), kNBins);
   expectScaledResiduals(*residRaw, 0., kTolNumeric);
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

   // A range that doesn't fully contain any bin (bins are 0.5 wide)
   EXPECT_THROW(makeResidHist(gauss, *data, Range(0.1, 0.4)), std::invalid_argument);

   // The RelativeExpected scale type requires an extendable pdf
   EXPECT_THROW(makeResidHist(gauss, *data, Normalization(1.0, RooAbsReal::RelativeExpected)), std::invalid_argument);

   // Unsupported DataError types
   EXPECT_THROW(makeResidHist(gauss, *data, DataError(99)), std::invalid_argument);

   // A model with no support in the range must throw
   RooGenericPdf zeroModel("zeroModel", "zeroModel", "x * 0.", RooArgSet{x});
   EXPECT_THROW(makeResidHist(zeroModel, *data), std::invalid_argument);
}
