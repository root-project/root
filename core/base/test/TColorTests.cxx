#include "gtest/gtest.h"

#include "TColor.h"
#include "TROOT.h"

TEST(TColor, Hex6String)
{
   Int_t ci = TColor::GetColor("#c0c0c0");
   auto color = gROOT->GetColor(ci);
   EXPECT_NE(color, nullptr);

   EXPECT_NEAR(color->GetRed(), 0.752941, 1e-4);
   EXPECT_NEAR(color->GetGreen(), 0.752941, 1e-4);
   EXPECT_NEAR(color->GetBlue(), 0.752941, 1e-4);
   EXPECT_NEAR(color->GetAlpha(), 1., 1e-4);
}

TEST(TColor, Hex8String)
{
   Int_t ci = TColor::GetColor("#b0b0b077");
   auto color = gROOT->GetColor(ci);
   EXPECT_NE(color, nullptr);

   EXPECT_NEAR(color->GetRed(), 0.690196, 1e-4);
   EXPECT_NEAR(color->GetGreen(), 0.690196, 1e-4);
   EXPECT_NEAR(color->GetBlue(), 0.690196, 1e-4);
   EXPECT_NEAR(color->GetAlpha(), 0.466667, 1e-4);
}
