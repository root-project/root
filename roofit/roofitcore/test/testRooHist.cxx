// Tests for the RooHist
// Authors: Jonas Rembser, CERN 12/2022

#include <RooAbsPdf.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooHelpers.h>
#include <RooHist.h>
#include <RooHistPdf.h>
#include <RooPlot.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <TH1D.h>

#include <gtest/gtest.h>

#include <memory>

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
