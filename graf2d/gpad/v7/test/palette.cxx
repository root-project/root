#include "gtest/gtest.h"

#include "ROOT/TPalette.hxx"

// Test palette interpolation
TEST(Palette, Interpolate)
{
   using namespace ROOT::Experimental;
   TPalette p1(TPalette::kDiscrete, {TColor::kWhite, TColor::kBlack});
   EXPECT_EQ(p1.GetColor(0.), TColor::kWhite);
}
