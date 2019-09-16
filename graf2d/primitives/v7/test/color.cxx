#include "gtest/gtest.h"

#include "ROOT/RColor.hxx"

// Predef
TEST(ColorTest, Predef) {
   using namespace ROOT::Experimental;
   {
      RColor col{RColor::kRed};
      EXPECT_EQ(col.GetRGB(), "255,0,0");
      EXPECT_EQ(col.HasAlpha(), false);
   }
   {
      RColor col{RColor::kBlue};
      col.SetAlpha(RColor::kTransparent);
      EXPECT_EQ(col.GetRGB(), "0,0,255");
      EXPECT_FLOAT_EQ(col.GetAlpha(), 0.);
   }
}
