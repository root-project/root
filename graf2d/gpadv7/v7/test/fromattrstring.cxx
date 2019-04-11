// @(#)root/graf2d:$Id$
// Author: Axel Naumann <axel@cern.ch>, 2018-07-22

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 #include "gtest/gtest.h"

#include "ROOT/RAttrLine.hxx"
#include "ROOT/RPadExtent.hxx"
#include "ROOT/RPadPos.hxx"

// Test reading of Extent from empty string.
TEST(ExtentFromAttrString, Empty) {
   using namespace ROOT::Experimental;
   RAttrLine l;
   
   RPadExtent cn{0.3_normal, 40_px};
   cn = FromAttributeString("", l, "FromEmpty", &cn);
   EXPECT_DOUBLE_EQ(0., cn.fHoriz.fNormal.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fHoriz.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fHoriz.fUser.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fVert.fNormal.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fVert.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fVert.fUser.fVal);
}

// Test reading of Pos from string.
TEST(PosFromAttrString, String) {
   using namespace ROOT::Experimental;
   RAttrLine l;
   
   RPadPos cn{0.3_normal, 40_px}; // NOTE: initial values are intentionally overwritten!
   cn = FromAttributeString("  -10   px    +0.1user, 0.12 normal -    -0.2  user + 22pixel - 12px", l, "One", &cn);
   EXPECT_DOUBLE_EQ(0., cn.fHoriz.fNormal.fVal);
   EXPECT_DOUBLE_EQ(-10., cn.fHoriz.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0.1, cn.fHoriz.fUser.fVal);
   EXPECT_DOUBLE_EQ(0.12, cn.fVert.fNormal.fVal);
   EXPECT_DOUBLE_EQ(10., cn.fVert.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0.2, cn.fVert.fUser.fVal);
}
