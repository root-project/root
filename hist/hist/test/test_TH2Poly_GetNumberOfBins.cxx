// test TH2Poly GetNumberOfBins

#include "gtest/gtest.h"

#include "TH2Poly.h"

TEST(TH2Poly, GetNumberOfBins)
{
   TH2Poly h2p;
   EXPECT_EQ(0, h2p.GetNumberOfBins());

   h2p.AddBin(1, 1, 1, 1);
   EXPECT_EQ(1, h2p.GetNumberOfBins());
}
