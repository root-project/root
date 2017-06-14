// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile2D.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TRandom2.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestProfile2Extend)
{
   TProfile2D::Approximate(true);
   TProfile2D h1("h1", "h1", 10, 0, 10, 10, 0, 10);
   TProfile2D h2("h2", "h0", 10, 0, 10, 10, 0, 20);
   h1.SetCanExtend(TH1::kYaxis);
   for (int i = 0; i < 10 * nEvents; ++i) {
      double x = r.Uniform(-1, 11);
      double y = r.Gaus(10, 3);
      double z = r.Gaus(10 + 2 * (x + y), 1);
      if (y <= 0 || y >= 20) continue; // do not want overflow in h0
      h1.Fill(x, y, z);
      h2.Fill(x, y, z);
   }
   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats, 1E-10));
   TProfile2D::Approximate(false);
}

TEST(StressHistorgram, TestProfileExtend)
{
   TProfile::Approximate(true);
   TProfile h1("h1", "h1", 10, 0, 10);
   TProfile h0("h0", "h0", 10, 0, 20);
   h1.SetCanExtend(TH1::kXaxis);
   for (int i = 0; i < nEvents; ++i) {
      double x = gRandom->Gaus(10, 3);
      double y = gRandom->Gaus(10 + 2 * x, 1);
      if (x <= 0 || x >= 20) continue; // do not want overflow in h0
      h1.Fill(x, y);
      h0.Fill(x, y);
   }
   EXPECT_TRUE(HistogramsEquals(h1, h0, cmpOptStats, 1E-10));
   TProfile::Approximate(false);
}
