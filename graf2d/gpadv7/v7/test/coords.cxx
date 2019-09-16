// @(#)root/graf2d:$Id$
// Author: Axel Naumann <axel@cern.ch>, 2017-08-26

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "gtest/gtest.h"

#include "ROOT/RPadLength.hxx"

// Test addition / subtraction of coords
TEST(PadCoord, AddSubtract) {
   using namespace ROOT::Experimental;

   RPadLength cn{0.3_normal};
   EXPECT_DOUBLE_EQ(0.3, cn.GetNormal());
   EXPECT_DOUBLE_EQ(0., cn.GetPixel());
   EXPECT_DOUBLE_EQ(0., cn.GetUser());

   RPadLength cn1{0.4_normal};
   cn += cn1;
   EXPECT_DOUBLE_EQ(0.7, cn.GetNormal());
   EXPECT_DOUBLE_EQ(0., cn.GetPixel());
   EXPECT_DOUBLE_EQ(0., cn.GetUser());

   RPadLength cp{120_px};
   EXPECT_DOUBLE_EQ(120., cp.GetPixel());
   EXPECT_DOUBLE_EQ(0., cp.GetNormal());
   EXPECT_DOUBLE_EQ(0., cp.GetUser());

   RPadLength sum = cn + cp;
   EXPECT_DOUBLE_EQ(0.7, sum.GetNormal());
   EXPECT_DOUBLE_EQ(120., sum.GetPixel());
   EXPECT_DOUBLE_EQ(0., sum.GetUser());

   sum -= RPadLength(0.2_user);
   EXPECT_DOUBLE_EQ(0.7, sum.GetNormal());
   EXPECT_DOUBLE_EQ(120., sum.GetPixel());
   EXPECT_DOUBLE_EQ(-0.2, sum.GetUser());

   sum *= 0.1;
   EXPECT_DOUBLE_EQ(0.07, sum.GetNormal());
   EXPECT_DOUBLE_EQ(12., sum.GetPixel());
   EXPECT_DOUBLE_EQ(-0.02, sum.GetUser());

   RPadLength subtr(0.07_normal, 12_px, -0.02_user);
   EXPECT_DOUBLE_EQ(0.07, subtr.GetNormal());
   EXPECT_DOUBLE_EQ(12., subtr.GetPixel());
   EXPECT_DOUBLE_EQ(-0.02, subtr.GetUser());

   sum -= subtr;
   EXPECT_NEAR(0., sum.GetNormal(), 1e-10);
   EXPECT_NEAR(0., sum.GetPixel(), 1e-10);
   EXPECT_NEAR(0., sum.GetUser(), 1e-10);
}
