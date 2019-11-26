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
   EXPECT_DOUBLE_EQ(0.3, cn.Horiz().GetNormal());
   EXPECT_DOUBLE_EQ(0., cn.Horiz().GetPixel());
   EXPECT_DOUBLE_EQ(0., cn.Horiz().GetUser());
   EXPECT_DOUBLE_EQ(0., cn.Vert().GetNormal());
   EXPECT_DOUBLE_EQ(40., cn.Vert().GetPixel());
   EXPECT_DOUBLE_EQ(0., cn.Vert().GetUser());

   RPadExtent cn1{0.4_normal, 20_px};
   cn += cn1;
   EXPECT_DOUBLE_EQ(0.7, cn.Horiz().GetNormal());
   EXPECT_DOUBLE_EQ(0., cn.Horiz().GetPixel());
   EXPECT_DOUBLE_EQ(0., cn.Horiz().GetUser());
   EXPECT_DOUBLE_EQ(0., cn.Vert().GetNormal());
   EXPECT_DOUBLE_EQ(60., cn.Vert().GetPixel());
   EXPECT_DOUBLE_EQ(0., cn.Vert().GetUser());

   RPadExtent cp{120_px, 0.42_normal};
   EXPECT_DOUBLE_EQ(120., cp.Horiz().GetPixel());
   EXPECT_DOUBLE_EQ(0., cp.Horiz().GetNormal());
   EXPECT_DOUBLE_EQ(0., cp.Horiz().GetUser());
   EXPECT_DOUBLE_EQ(0.42, cp.Vert().GetNormal());
   EXPECT_DOUBLE_EQ(0., cp.Vert().GetPixel());
   EXPECT_DOUBLE_EQ(0., cp.Vert().GetUser());

   RPadExtent sum = cn + cp;
   EXPECT_DOUBLE_EQ(120., sum.Horiz().GetPixel());
   EXPECT_DOUBLE_EQ(0.7, sum.Horiz().GetNormal());
   EXPECT_DOUBLE_EQ(0., sum.Horiz().GetUser());
   EXPECT_DOUBLE_EQ(0.42, sum.Vert().GetNormal());
   EXPECT_DOUBLE_EQ(60., sum.Vert().GetPixel());
   EXPECT_DOUBLE_EQ(0., sum.Vert().GetUser());

   sum -= RPadExtent(0.2_user, 12_px);
   EXPECT_DOUBLE_EQ(120., sum.Horiz().GetPixel());
   EXPECT_DOUBLE_EQ(0.7, sum.Horiz().GetNormal());
   EXPECT_DOUBLE_EQ(-0.2, sum.Horiz().GetUser());
   EXPECT_DOUBLE_EQ(0.42, sum.Vert().GetNormal());
   EXPECT_DOUBLE_EQ(48., sum.Vert().GetPixel());
   EXPECT_DOUBLE_EQ(0., sum.Vert().GetUser());

   sum *= {0.1, 10.};
   EXPECT_DOUBLE_EQ(12., sum.Horiz().GetPixel());
   EXPECT_DOUBLE_EQ(0.07, sum.Horiz().GetNormal());
   EXPECT_DOUBLE_EQ(-0.02, sum.Horiz().GetUser());
   EXPECT_DOUBLE_EQ(4.2, sum.Vert().GetNormal());
   EXPECT_DOUBLE_EQ(480., sum.Vert().GetPixel());
   EXPECT_DOUBLE_EQ(0., sum.Vert().GetUser());

   RPadExtent subtr({0.07_normal, 12_px, -0.02_user},
                    {4.2_normal, 480_px, 0._user});
   EXPECT_DOUBLE_EQ(12., subtr.Horiz().GetPixel());
   EXPECT_DOUBLE_EQ(0.07, subtr.Horiz().GetNormal());
   EXPECT_DOUBLE_EQ(-0.02, subtr.Horiz().GetUser());
   EXPECT_DOUBLE_EQ(4.2, subtr.Vert().GetNormal());
   EXPECT_DOUBLE_EQ(480., subtr.Vert().GetPixel());
   EXPECT_DOUBLE_EQ(0., subtr.Vert().GetUser());

   sum -= subtr;
   static constexpr double delta = 1E-15;
   EXPECT_NEAR(0.,sum.Horiz().GetPixel(), delta);
   EXPECT_NEAR(0., sum.Horiz().GetNormal(), delta);
   EXPECT_NEAR(0., sum.Horiz().GetUser(), delta);
   EXPECT_NEAR(0., sum.Vert().GetNormal(), delta);
   EXPECT_NEAR(0., sum.Vert().GetPixel(), delta);
   EXPECT_NEAR(0., sum.Vert().GetUser(), delta);
}
