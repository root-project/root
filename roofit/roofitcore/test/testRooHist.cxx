// Tests for the RooHist
// Authors: Jonas Rembser, CERN 12/2022

#include <RooDataHist.h>
#include <RooHist.h>
#include <RooHistPdf.h>
#include <RooPlot.h>
#include <RooRealVar.h>

#include <TH1D.h>

#include <gtest/gtest.h>

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
