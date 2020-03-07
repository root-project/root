#include "gtest/gtest.h"

#include "ROOT/RPalette.hxx"

// Test palette interpolation
TEST(Palette, DiscreteTrivial)
{
   using namespace ROOT::Experimental;
   RPalette p1(RPalette::kDiscrete, {RColor::kWhite, RColor::kBlack});
   EXPECT_EQ(p1.GetColor(0.), RColor::kWhite);
   EXPECT_EQ(p1.GetColor(1.), RColor::kBlack);
}

TEST(Palette, DiscretePlaced)
{
   using namespace ROOT::Experimental;
   RPalette p1(RPalette::kDiscrete, {{0., RColor::kWhite}, {.3, RColor::kRed}, {.7, RColor::kBlue}, {1., RColor::kBlack}});
   EXPECT_EQ(p1.GetColor(0.), RColor::kWhite);
   EXPECT_EQ(p1.GetColor(.3), RColor::kRed);
   EXPECT_EQ(p1.GetColor(.7), RColor::kBlue);
   EXPECT_EQ(p1.GetColor(1.), RColor::kBlack);
}
