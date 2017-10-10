#include "gtest/gtest.h"

#include "ROOT/TColor.hxx"

// Predef
TEST(ColorTest, Predef) {
   using namespace ROOT::Experimental;
   {
      TColor col{TColor::kRed};
      EXPECT_FLOAT_EQ(col.GetRed(), 1.);
      EXPECT_FLOAT_EQ(col.GetGreen(), 0.);
      EXPECT_FLOAT_EQ(col.GetBlue(), 0.);
      EXPECT_FLOAT_EQ(col.GetAlpha(), 1.);
   }
   {
      TColor col{TColor::kBlue};
      col.SetAlpha(TColor::kTransparent);
      EXPECT_FLOAT_EQ(col.GetRed(), 0.);
      EXPECT_FLOAT_EQ(col.GetGreen(), 0.);
      EXPECT_FLOAT_EQ(col.GetBlue(), 1.);
      EXPECT_FLOAT_EQ(col.GetAlpha(), 0.);
   }
}
