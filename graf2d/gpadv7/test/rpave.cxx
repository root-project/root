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

   auto frame = canv.GetOrCreateFrame();

   auto pave = canv.Draw<RPave>();
   pave->AttrFill().SetColor(RColor::kRed);

   // when adding pave, RFrame is automatically created
   EXPECT_EQ(canv.NumPrimitives(), 2u);

   EXPECT_EQ(pave->AttrFill().GetColor(), RColor::kRed);
}

