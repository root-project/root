// Tests for the RooKeysPdf and friends
// Authors: Jonas Rembser, CERN  07/2022

#include <RooRealVar.h>
#include <RooKeysPdf.h>
#include <RooNDKeysPdf.h>

#include "gtest/gtest.h"

// Test the support of RooKeysPdf and RooNDKeysPdf for weighted datasets.
TEST(RooKeysPdf, WeightedDataset)
{
   // We create data with 100 events at x = 5 and 400 events at x = 15. One
   // version will have 500 unweighted entries, the other will have only 2
   // entries with the weights 100 and 400 to represent the same data. The
   // resulting RooKeysPdfs should be identical for both datasets. Checking
   // this validates that dataset weights are correctly dealt with.

   RooRealVar x("x", "x", 0, 20);
   RooRealVar weight("weight", "weight", 0, 2000);

   const std::size_t nEvents0 = 100;
   const std::size_t nEvents1 = 400;

   RooDataSet data1{"data1", "data1", x};
   RooDataSet data2{"data2", "data2", {x, weight}, "weight"};

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
