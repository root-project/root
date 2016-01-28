#include "gtest/gtest.h"
#include <ROOT/THist.h>
#include <ROOT/THistBinIter.h>

using namespace ROOT::Experimental;

// Tests the number of bins
TEST(BinIterNBins, NumBins) {
  TH2F h({10, -1., 1.}, {10, -1., 1.});
  int nBins = 0;
  for (auto &&bin: h) {
    (void)bin;
    ++nBins;
  }
  EXPECT_EQ(h.GetImpl()->GetNBins(), nBins);
}

