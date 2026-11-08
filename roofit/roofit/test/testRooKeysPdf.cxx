// Tests for the RooKeysPdf and friends
// Authors: Jonas Rembser, CERN  07/2022

#include <RooDataSet.h>
#include <RooGenericPdf.h>
#include <RooHelpers.h>
#include <RooKeysPdf.h>
#include <RooNDKeysPdf.h>
#include <RooPlot.h>
#include <RooRealVar.h>

#include <TMath.h>
#include <TRandom3.h>

#include "gtest/gtest.h"

#include <cmath>
#include <stdexcept>

// Test the support of RooKeysPdf and RooNDKeysPdf for weighted datasets.
TEST(RooKeysPdf, WeightedDataset)
{
   // We create data with 100 events at x = 5 and 400 events at x = 15. One
   // version will have 500 unweighted entries, the other will have only 2
   // entries with the weights 100 and 400 to represent the same data. The
   // resulting RooKeysPdfs should be identical for both datasets. Checking
   // this validates that dataset weights are correctly dealt with.

   RooRealVar x("x", "x", 0, 20);

   const std::size_t nEvents0 = 100;
   const std::size_t nEvents1 = 400;

   RooDataSet data1{"data1", "data1", x};
   RooDataSet data2{"data2", "data2", x, RooFit::WeightVar()};

   x.setVal(5);
   for (std::size_t i = 0; i < nEvents0; ++i) {
      data1.add(x, 1.0);
   }
   data2.add(x, double(nEvents0));
   x.setVal(15);
   for (std::size_t i = 0; i < nEvents1; ++i) {
      data1.add(x, 1.0);
   }
   data2.add(x, double(nEvents1));

   // Creating RooKeysPdf and RooNDKeysPdf with adaptive kernel and no
   // mirroring for both weighted and unweighted datasets.
   RooKeysPdf kest1("kest1", "kest1", x, data1, RooKeysPdf::NoMirror);
   RooKeysPdf kest2("kest2", "kest2", x, data2, RooKeysPdf::NoMirror);
   RooNDKeysPdf kestND1("kestND1", "kestND1", x, data1, "a");
   RooNDKeysPdf kestND2("kestND2", "kestND2", x, data2, "a");

   RooArgSet normSet{x};

   // Check if values for the unweighted and weighted datasets are the same
   double xVal = x.getMin();
   while (xVal < x.getMax()) {
      EXPECT_FLOAT_EQ(kest1.getVal(normSet), kest2.getVal(normSet));
      EXPECT_FLOAT_EQ(kestND1.getVal(normSet), kestND2.getVal(normSet));
      x.setVal(xVal);
      xVal += 0.1;
   }
}

// Test generation with proto data, covering GitHub issue #12286.
TEST(RooKeysPdf, GenerationWithProtoData)
{
   using namespace RooFit;

   RooRealVar x{"x", "", 0, 1};
   RooGenericPdf pdfX{"pdf_x", "x", {x}};

   std::unique_ptr<RooDataSet> dtBase{pdfX.generate(x, 10000)};

   RooKeysPdf pdfKeys{"pdf_keys", "", x, *dtBase, RooKeysPdf::MirrorBoth};

   RooRealVar y{"y", "", 0, 1};
   RooDataSet proto{"proto_y", "", y};
   proto.add(y);

   std::unique_ptr<RooDataSet> dtKeysWithProto{pdfKeys.generate(x, NumEvents(10000), ProtoData(proto))};

   std::unique_ptr<RooPlot> frame{x.frame()};
   dtKeysWithProto->plotOn(frame.get());
   pdfKeys.plotOn(frame.get());

   // If the dataset generation worked, the chi-square is not too terrible
   EXPECT_LE(frame->chiSquare(), 2.0);
}

// Test the support of RooKeysPdf and RooNDKeysPdf for datasets with negative
// weights, covering GitHub issue #12639. Before, the kernel widths were
// computed from the square root of the event weight or of the signed density,
// which made the whole pdf NaN as soon as some weights were negative.
TEST(RooKeysPdf, NegativeWeights)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::InputArguments, true};

   RooRealVar x("x", "x", 0, 10);
   RooRealVar w("w", "w", -10, 10);

   // Gaussian signal on top of a flat background. The background is
   // subtracted with an independent flat sample of negative weights, like in
   // a sideband subtraction, so the weighted data describe the signal only.
   RooDataSet data{"data", "data", {x, w}, RooFit::WeightVar(w)};
   TRandom3 rng{1337};
   for (int i = 0; i < 5000; ++i) {
      x.setVal(rng.Gaus(5.0, 1.0));
      data.add(x, 1.0);
   }
   for (int i = 0; i < 2000; ++i) {
      x.setVal(rng.Uniform(0.0, 10.0));
      data.add(x, 1.0);
      x.setVal(rng.Uniform(0.0, 10.0));
      data.add(x, -1.0);
   }

   RooKeysPdf kest("kest", "kest", x, data, RooKeysPdf::NoMirror);
   RooNDKeysPdf kestND("kestND", "kestND", x, data, "a");

   RooArgSet normSet{x};
   const double sigmaTrue = 1.0;
   for (double xVal = 0.25; xVal < 10.0; xVal += 0.5) {
      x.setVal(xVal);
      const double truth = TMath::Gaus(xVal, 5.0, sigmaTrue, true);
      const double val = kest.getVal(normSet);
      const double valND = kestND.getVal(normSet);
      EXPECT_TRUE(std::isfinite(val)) << "x = " << xVal;
      EXPECT_TRUE(std::isfinite(valND)) << "x = " << xVal;
      // The estimates follow the signal density, up to statistical
      // fluctuations and the smoothing by the kernels.
      EXPECT_NEAR(val, truth, 0.03) << "x = " << xVal;
      EXPECT_NEAR(valND, truth, 0.03) << "x = " << xVal;
   }
   x.setVal(5.0);
   EXPECT_NEAR(kest.getVal(normSet), TMath::Gaus(5.0, 5.0, sigmaTrue, true), 0.02);
   EXPECT_NEAR(kestND.getVal(normSet), TMath::Gaus(5.0, 5.0, sigmaTrue, true), 0.02);
}

// A dataset with a non-positive sum of weights doesn't describe a
// distribution: the pdfs must refuse it instead of returning NaNs.
TEST(RooKeysPdf, NonPositiveSumOfWeights)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::FATAL, 0u, RooFit::InputArguments, true};

   RooRealVar x("x", "x", 0, 10);
   RooRealVar w("w", "w", -10, 10);

   RooDataSet data{"data", "data", {x, w}, RooFit::WeightVar(w)};
   x.setVal(3.0);
   data.add(x, 1.0);
   x.setVal(6.0);
   data.add(x, -2.0);

   EXPECT_THROW(RooKeysPdf("kest", "kest", x, data, RooKeysPdf::NoMirror), std::invalid_argument);
   EXPECT_THROW(RooNDKeysPdf("kestND", "kestND", x, data, "a"), std::invalid_argument);
}
