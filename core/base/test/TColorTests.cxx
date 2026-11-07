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

// ROOT-21367: On mac beta, the constructor of TColorGradient crashes with a stack corruption
TEST(TColor, Gradient)
{
   auto fcol1 = TColor::GetColor((Float_t)0.25, 0.25, 0.25, 0.55); // special frame color 1
   auto fcol2 = TColor::GetColor((Float_t)1., 1., 1., 0.05);       // special frame color 2

   auto frameGradient = TColor::GetLinearGradient(0., {fcol1, fcol2, fcol2, fcol1}, {0., 0.2, 0.8, 1.});

   // This gradient is a mixture of two standard colors.
   auto padGradient = TColor::GetLinearGradient(0., {30, 38});

   // Another gradient built from three standard colors.
   auto histGradient = TColor::GetLinearGradient(45., {kYellow, kOrange, kRed});

   EXPECT_NE(fcol1, frameGradient);
   EXPECT_NE(fcol1, padGradient);
   EXPECT_NE(fcol1, histGradient);
   EXPECT_NE(frameGradient, padGradient);
   EXPECT_NE(padGradient, histGradient);
}
