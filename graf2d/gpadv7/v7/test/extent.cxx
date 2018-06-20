// @(#)root/graf2d:$Id$
// Author: Axel Naumann <axel@cern.ch>, 2017-09-15

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 #include "gtest/gtest.h"

#include "ROOT/RPadExtent.hxx"

// Test addition of Extents
TEST(PadExtent, Add) {
   using namespace ROOT::Experimental;
   
   RPadExtent cn{0.3_normal, 40_px};
   EXPECT_DOUBLE_EQ(0.3, cn.fHoriz.fNormal.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fHoriz.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fHoriz.fUser.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fVert.fNormal.fVal);
   EXPECT_DOUBLE_EQ(40., cn.fVert.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fVert.fUser.fVal);

   RPadExtent cn1{0.4_normal, 20_px};
   cn += cn1;
   EXPECT_DOUBLE_EQ(0.7, cn.fHoriz.fNormal.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fHoriz.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fHoriz.fUser.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fVert.fNormal.fVal);
   EXPECT_DOUBLE_EQ(60., cn.fVert.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., cn.fVert.fUser.fVal);

   RPadExtent cp{120_px, 0.42_normal};
   EXPECT_DOUBLE_EQ(120., cp.fHoriz.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., cp.fHoriz.fNormal.fVal);
   EXPECT_DOUBLE_EQ(0., cp.fHoriz.fUser.fVal);
   EXPECT_DOUBLE_EQ(0.42, cp.fVert.fNormal.fVal);
   EXPECT_DOUBLE_EQ(0., cp.fVert.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., cp.fVert.fUser.fVal);

   RPadExtent sum = cn + cp;
   EXPECT_DOUBLE_EQ(120., sum.fHoriz.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0.7, sum.fHoriz.fNormal.fVal);
   EXPECT_DOUBLE_EQ(0., sum.fHoriz.fUser.fVal);
   EXPECT_DOUBLE_EQ(0.42, sum.fVert.fNormal.fVal);
   EXPECT_DOUBLE_EQ(60., sum.fVert.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., sum.fVert.fUser.fVal);

   sum -= RPadExtent(0.2_user, 12_px);
   EXPECT_DOUBLE_EQ(120., sum.fHoriz.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0.7, sum.fHoriz.fNormal.fVal);
   EXPECT_DOUBLE_EQ(-0.2, sum.fHoriz.fUser.fVal);
   EXPECT_DOUBLE_EQ(0.42, sum.fVert.fNormal.fVal);
   EXPECT_DOUBLE_EQ(48., sum.fVert.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., sum.fVert.fUser.fVal);

   sum *= {0.1, 10.};
   EXPECT_DOUBLE_EQ(12.,sum.fHoriz.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0.07, sum.fHoriz.fNormal.fVal);
   EXPECT_DOUBLE_EQ(-0.02, sum.fHoriz.fUser.fVal);
   EXPECT_DOUBLE_EQ(4.2, sum.fVert.fNormal.fVal);
   EXPECT_DOUBLE_EQ(480., sum.fVert.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., sum.fVert.fUser.fVal);

   RPadExtent subtr({0.07_normal, 12_px, -0.02_user},
                    {4.2_normal, 480_px, 0._user});
   EXPECT_DOUBLE_EQ(12., subtr.fHoriz.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0.07, subtr.fHoriz.fNormal.fVal);
   EXPECT_DOUBLE_EQ(-0.02, subtr.fHoriz.fUser.fVal);
   EXPECT_DOUBLE_EQ(4.2, subtr.fVert.fNormal.fVal);
   EXPECT_DOUBLE_EQ(480., subtr.fVert.fPixel.fVal);
   EXPECT_DOUBLE_EQ(0., subtr.fVert.fUser.fVal);
                 
   sum -= subtr;
   static constexpr double delta = 1E-15;
   EXPECT_NEAR(0.,sum.fHoriz.fPixel.fVal, delta);
   EXPECT_NEAR(0., sum.fHoriz.fNormal.fVal, delta);
   EXPECT_NEAR(0., sum.fHoriz.fUser.fVal, delta);
   EXPECT_NEAR(0., sum.fVert.fNormal.fVal, delta);
   EXPECT_NEAR(0., sum.fVert.fPixel.fVal, delta);
   EXPECT_NEAR(0., sum.fVert.fUser.fVal, delta);
}
