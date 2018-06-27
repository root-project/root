#include "gtest/gtest.h"

#include "ROOT/RPalette.hxx"

// Test palette interpolation
TEST(Palette, Interpolate)
{
   using namespace ROOT::Experimental;
   RPalette p1(RPalette::kDiscrete, {RColor::kWhite, RColor::kBlack});
   EXPECT_EQ(p1.GetColor(0.), RColor::kWhite);
}
