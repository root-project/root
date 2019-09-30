#include "gtest/gtest.h"

#include "ROOT/RColor.hxx"

// Predef
TEST(ColorTest, Predef) {
   using namespace ROOT::Experimental;
   {
      RColor col{RColor::kRed};
      EXPECT_EQ(col.GetHex(), "FF0000");
      EXPECT_EQ(col.HasAlpha(), false);
   }
   {
      RColor col{RColor::kBlue};
      col.SetAlpha(RColor::kTransparent);
      EXPECT_EQ(col.GetHex(), "0000FF");
      EXPECT_FLOAT_EQ(col.GetAlpha(), 0.);
   }
}
