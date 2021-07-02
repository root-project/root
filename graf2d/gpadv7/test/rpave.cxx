// @(#)root/graf2d:$Id$
// Author: Sergey Linev <s.linev@gsi.de>, 2020-06-18


#include "gtest/gtest.h"

#include "ROOT/RCanvas.hxx"
#include "ROOT/RPave.hxx"

using namespace ROOT::Experimental;

// Test RPave API
TEST(Primitives, RPave)
{
   RCanvas canv;

   auto frame = canv.AddFrame();

   auto pave = canv.Add<RPave>();
   pave->border.color = RColor::kRed;
   pave->border.width = 3.;
   pave->fill.color = RColor::kBlue;
   pave->fill.style = 3003;
   pave->cornerX = 0.03_normal;
   pave->width = 0.4_normal;
   pave->cornerY = -0.03_normal;
   pave->height = 0.2_normal;

   // when adding pave, RFrame is automatically created
   EXPECT_EQ(canv.NumPrimitives(), 2u);

   EXPECT_EQ(pave->border.color, RColor::kRed);
   EXPECT_DOUBLE_EQ(pave->border.width, 3.);

   EXPECT_EQ(pave->fill.color, RColor::kBlue);
   EXPECT_EQ(pave->fill.style, 3003);

   EXPECT_EQ(pave->cornerX, 0.03_normal);
   EXPECT_EQ(pave->width, 0.4_normal);
   EXPECT_EQ(pave->cornerY, -0.03_normal);
   EXPECT_EQ(pave->height, 0.2_normal);
}

