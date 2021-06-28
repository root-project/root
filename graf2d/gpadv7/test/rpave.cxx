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
   pave->border.width = 3.f;
   pave->fill.color = RColor::kBlue;
   pave->fill.style = 3003;
   pave->SetCornerX(0.03_normal);
   pave->SetWidth(0.4_normal);
   pave->SetCornerY(-0.03_normal);
   pave->SetHeight(0.2_normal);

   // when adding pave, RFrame is automatically created
   EXPECT_EQ(canv.NumPrimitives(), 2u);

   EXPECT_EQ(pave->border.color, RColor::kRed);
   EXPECT_EQ(pave->border.width, 3.f);

   EXPECT_EQ(pave->fill.color, RColor::kBlue);
   EXPECT_EQ(pave->fill.style, 3003);

   EXPECT_EQ(pave->GetCornerX(), 0.03_normal);
   EXPECT_EQ(pave->GetWidth(), 0.4_normal);
   EXPECT_EQ(pave->GetCornerY(), -0.03_normal);
   EXPECT_EQ(pave->GetHeight(), 0.2_normal);
}

