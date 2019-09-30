// @(#)root/graf2d:$Id$
// Author: Sergey Linev <s.linev@gsi.de>, 2019-09-30

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 #include "gtest/gtest.h"

#include "ROOT/RColor.hxx"

using namespace ROOT::Experimental;

// Test usage of empty color
TEST(RColorTest, Empty) {
   RColor col;
   EXPECT_EQ(col.GetHex(), "");
   EXPECT_EQ(col.GetAlphaHex(), "");
   EXPECT_EQ(col.GetRGB(), "");
   EXPECT_EQ(col.GetName(), "");
}

// Test usage of empty color
TEST(RColorTest, AsHex) {
   RColor col;
   col.SetHex(0,0,0);
   EXPECT_EQ(col.GetHex(), "000000");
   col.SetHex(1,7,15);
   EXPECT_EQ(col.GetHex(), "01070F");
   col.SetHex(127,127,127);
   EXPECT_EQ(col.GetHex(), "7F7F7F");
   col.SetHex(255,255,255);
   EXPECT_EQ(col.GetHex(), "FFFFFF");
}

TEST(RColorTest, Alpha) {

   static constexpr double delta = 0.01; // approx precision of alpha storage

   RColor col;
   col.SetAlpha(0);
   EXPECT_DOUBLE_EQ(col.GetAlpha(), 0.);

   col.SetAlpha(1);
   EXPECT_DOUBLE_EQ(col.GetAlpha(), 1.);

   col.SetAlpha(0.1);
   EXPECT_NEAR(0.1,col.GetAlpha(), delta);

   col.SetAlpha(0.5);
   EXPECT_NEAR(0.5,col.GetAlpha(), delta);

   col.SetAlpha(0.8);
   EXPECT_NEAR(0.8,col.GetAlpha(), delta);
}
