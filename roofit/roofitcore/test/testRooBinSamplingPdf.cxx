// Tests for the RooBinSamplingPdf
// Authors: Jonas Rembser, CERN  03/2022

#include <RooArgSet.h>
#include <RooBinSamplingPdf.h>
#include <RooGenericPdf.h>
#include <RooAddPdf.h>
#include <RooRandom.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooRealVar.h>
#include <RooMsgService.h>

#include <gtest/gtest.h>

#include <memory>

// For a linear pdf, doing the bin sampling should make no difference because
// the integral of a linear function is the same as the central point.
TEST(RooBinSamplingPdf, LinearPdfCrossCheck)
{
   using namespace RooFit;

   auto& msg = RooMsgService::instance();
   msg.setGlobalKillBelow(RooFit::WARNING);

   RooRandom::randomGenerator()->SetSeed(1337ul);

   RooRealVar x("x", "x", 0.1, 5.1);
   x.setBins(10);

   RooGenericPdf pdf("lin", "x", RooArgSet(x));
   std::unique_ptr<RooDataHist> dataH(pdf.generateBinned(x, 10000));
   RooRealVar w("w", "weight", 0., 0., 10000.);
   RooDataSet data("data", "data", RooArgSet(x, w), RooFit::WeightVar(w));
   for (int i = 0; i < dataH->numEntries(); ++i) {
      auto coords = dataH->get(i);
      data.add(*coords, dataH->weight());
   }

   std::unique_ptr<RooAbsReal> nll1(pdf.createNLL(data));
   std::unique_ptr<RooAbsReal> nll2(pdf.createNLL(data, IntegrateBins(1.E-3)));

   EXPECT_FLOAT_EQ(nll2->getVal(), nll1->getVal());
}

// For a linear pdf, doing the bin sampling should make no difference because
// the integral of a linear function is the same as the central point.
// Similar to "LinearPdfCrossCheck", but this time for a subrange fit.
TEST(RooBinSamplingPdf, LinearPdfSubRangeCrossCheck)
{
   using namespace RooFit;

   auto& msg = RooMsgService::instance();
   msg.setGlobalKillBelow(RooFit::WARNING);

   RooRandom::randomGenerator()->SetSeed(1337ul);

   RooRealVar x("x", "x", 0.1, 5.1);
   x.setBins(10);
   x.setRange("range", 0.1, 4.1);
   x.setBins(8, "range"); // consistent binning

   RooGenericPdf pdf("lin", "x", RooArgSet(x));
   std::unique_ptr<RooDataHist> dataH(pdf.generateBinned(x, 10000));
   RooRealVar w("w", "weight", 0., 0., 10000.);
   RooDataSet data("data", "data", RooArgSet(x, w), RooFit::WeightVar(w));
   for (int i = 0; i < dataH->numEntries(); ++i) {
      auto coords = dataH->get(i);
      data.add(*coords, dataH->weight());
   }

   std::unique_ptr<RooAbsReal> nll1(pdf.createNLL(data, Range("range")));
   std::unique_ptr<RooAbsReal> nll2(pdf.createNLL(data, Range("range"), IntegrateBins(1.E-3)));

   EXPECT_FLOAT_EQ(nll2->getVal(), nll1->getVal());
}
