#include "gtest/gtest.h"

#include "ROOT/RColor.hxx"

// Predef
TEST(ColorTest, Predef) {
   using namespace ROOT::Experimental;
   {
      RColor col{RColor::kRed};
      EXPECT_FLOAT_EQ(col.GetRed(), 1.);
      EXPECT_FLOAT_EQ(col.GetGreen(), 0.);
      EXPECT_FLOAT_EQ(col.GetBlue(), 0.);
      EXPECT_FLOAT_EQ(col.GetAlpha(), 1.);
   }
   {
      RColor col{RColor::kBlue};
      col.SetAlpha(RColor::kTransparent);
      EXPECT_FLOAT_EQ(col.GetRed(), 0.);
      EXPECT_FLOAT_EQ(col.GetGreen(), 0.);
      EXPECT_FLOAT_EQ(col.GetBlue(), 1.);
      EXPECT_FLOAT_EQ(col.GetAlpha(), 0.);
   }
}
