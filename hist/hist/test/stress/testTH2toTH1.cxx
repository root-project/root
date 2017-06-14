// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TRandom2.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestTH2toTH1)
{
   const unsigned int binsizeX = 10;
   const unsigned int binsizeY = 11;
   static const unsigned int minbinX = 2;
   static const unsigned int maxbinX = 5;
   static const unsigned int minbinY = 3;
   static const unsigned int maxbinY = 8;
   const int lower_limit = 0;
   const int upper_limit = 10;

   r.SetSeed(10);

   TH2D *h2XY = new TH2D("h2XY", "h2XY", binsizeX, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit);

   TH1::StatOverflows(kTRUE);

   TH1D *h1X = new TH1D("h1X", "h1X", binsizeX, lower_limit, upper_limit);
   TH1D *h1Y = new TH1D("h1Y", "h1Y", binsizeY, lower_limit, upper_limit);

   TH1D *h1XOR = new TH1D("h1XOR", "h1XOR", binsizeX, lower_limit, upper_limit);
   TH1D *h1YOR = new TH1D("h1YOR", "h1YOR", binsizeY, lower_limit, upper_limit);

   TH1D *h1XR = new TH1D("h1XR", "h1XR", maxbinX - minbinX + 1, h1X->GetXaxis()->GetBinLowEdge(minbinX),
                         h1X->GetXaxis()->GetBinUpEdge(maxbinX));
   TH1D *h1YR = new TH1D("h1YR", "h1YR", maxbinY - minbinY + 1, h1Y->GetXaxis()->GetBinLowEdge(minbinY),
                         h1Y->GetXaxis()->GetBinUpEdge(maxbinY));

   TProfile *pe1XY = new TProfile("pe1XY", "pe1XY", binsizeX, lower_limit, upper_limit);
   TProfile *pe1XYOR = new TProfile("pe1XYOR", "pe1XYOR", binsizeX, lower_limit, upper_limit);
   TProfile *pe1XYR = new TProfile("pe1XYR", "pe1XYR", maxbinX - minbinX + 1, h1X->GetXaxis()->GetBinLowEdge(minbinX),
                                   h1X->GetXaxis()->GetBinUpEdge(maxbinX));

   TProfile *pe1YX = new TProfile("pe1YX", "pe1YX", binsizeY, lower_limit, upper_limit);
   TProfile *pe1YXOR = new TProfile("pe1YXOR", "pe1YXOR", binsizeY, lower_limit, upper_limit);
   TProfile *pe1YXR = new TProfile("pe1YXR", "pe1YXR", maxbinY - minbinY + 1, h1Y->GetXaxis()->GetBinLowEdge(minbinY),
                                   h1Y->GetXaxis()->GetBinUpEdge(maxbinY));

   for (int ix = 0; ix <= h2XY->GetXaxis()->GetNbins() + 1; ++ix) {
      double xc = h2XY->GetXaxis()->GetBinCenter(ix);
      double x = xc + centre_deviation * h2XY->GetXaxis()->GetBinWidth(ix);
      for (int iy = 0; iy <= h2XY->GetYaxis()->GetNbins() + 1; ++iy) {
         double yc = h2XY->GetYaxis()->GetBinCenter(iy);
         double y = yc + centre_deviation * h2XY->GetYaxis()->GetBinWidth(iy);

         Double_t w = (Double_t)r.Uniform(1, 3);

         h2XY->Fill(x, y, w);

         h1X->Fill(x, w);
         h1Y->Fill(y, w);

         pe1XY->Fill(xc, yc, w);
         pe1YX->Fill(yc, xc, w);
         if (x >= h1X->GetXaxis()->GetBinLowEdge(minbinX) && x <= h1X->GetXaxis()->GetBinUpEdge(maxbinX) &&
             y >= h1Y->GetXaxis()->GetBinLowEdge(minbinY) && y <= h1Y->GetXaxis()->GetBinUpEdge(maxbinY)) {
            h1XOR->Fill(x, w);
            h1YOR->Fill(y, w);
            h1XR->Fill(x, w);
            h1YR->Fill(y, w);
            pe1XYR->Fill(xc, yc, w);
            pe1YXR->Fill(yc, xc, w);
            pe1XYOR->Fill(xc, yc, w);
            pe1YXOR->Fill(yc, xc, w);
         }
      }
   }

   int options = cmpOptStats;

   TH1D *projection;
   // TH1 derived from h2XY
   projection = h2XY->ProjectionX("x");
   EXPECT_EQ(0, Equals("TH2XY    -> X", h1X, projection, options));
   delete projection;
   projection = h2XY->ProjectionY("y");
   EXPECT_EQ(0, Equals("TH2XY    -> Y", h1Y, projection, options));
   delete projection;

   projection = h2XY->ProjectionX("ox", 0, -1, "o");
   EXPECT_EQ(0, Equals("TH2XYO  -> X", h1X, projection, options));
   delete projection;
   projection = h2XY->ProjectionY("oy", 0, -1, "o");
   EXPECT_EQ(0, Equals("TH2XYO  -> Y", h1Y, projection, options));
   delete projection;

   projection = (TH1D *)h2XY->ProfileX("PX", 0, h2XY->GetYaxis()->GetNbins() + 1);
   EXPECT_EQ(0, Equals("TH2XY -> PX", pe1XY, projection, options));
   delete projection;

   projection = (TH1D *)h2XY->ProfileY("PY", 0, h2XY->GetXaxis()->GetNbins() + 1);
   EXPECT_EQ(0, Equals("TH2XY -> PY", pe1YX, projection, options));
   delete projection;

   projection = (TH1D *)h2XY->ProfileX("OPX", 0, h2XY->GetYaxis()->GetNbins() + 1, "o");
   EXPECT_EQ(0, Equals("TH2XYO -> PX", pe1XY, projection, options));
   delete projection;
   projection = (TH1D *)h2XY->ProfileY("OPY", 0, h2XY->GetXaxis()->GetNbins() + 1, "o");
   EXPECT_EQ(0, Equals("TH2XYO -> PY", pe1YX, projection, options));
   delete projection;

   h2XY->GetXaxis()->SetRange(minbinX, maxbinX);
   h2XY->GetYaxis()->SetRange(minbinY, maxbinY);

   h1X->GetXaxis()->SetRange(minbinX, maxbinX);
   h1Y->GetXaxis()->SetRange(minbinY, maxbinY);

   pe1XY->GetXaxis()->SetRange(minbinX, maxbinX);
   pe1YX->GetXaxis()->SetRange(minbinY, maxbinY);

   // This two, the statistics should work!
   options = 0;

   projection = h2XY->ProjectionX("x");
   EXPECT_EQ(0, Equals("TH2XYR  -> X", h1XR, projection, options));
   delete projection;
   projection = h2XY->ProjectionY("y");
   EXPECT_EQ(0, Equals("TH2XYR  -> Y", h1YR, projection, options));
   delete projection;

   projection = h2XY->ProjectionX("ox", 0, -1, "o");
   EXPECT_EQ(0, Equals("TH2XYRO -> X", h1XOR, projection, options));
   delete projection;
   projection = h2XY->ProjectionY("oy", 0, -1, "o");
   EXPECT_EQ(0, Equals("TH2XYRO -> Y", h1YOR, projection, options));
   delete projection;

   projection = (TH1D *)h2XY->ProfileX("PX");
   EXPECT_EQ(0, Equals("TH2XYR -> PX", pe1XYR, projection, options));
   delete projection;
   projection = (TH1D *)h2XY->ProfileY("PY");
   EXPECT_EQ(0, Equals("TH2XYR -> PY", pe1YXR, projection, options));
   delete projection;

   projection = (TH1D *)h2XY->ProfileX("OPX", 0, -1, "o");
   EXPECT_EQ(0, Equals("TH2XYRO -> PX", pe1XYOR, projection, options));
   delete projection;
   projection = (TH1D *)h2XY->ProfileY("OPY", 0, -1, "o");
   EXPECT_EQ(0, Equals("TH2XYRO -> PY", pe1YXOR, projection, options));
   delete projection;

   options = 0;

   delete h2XY;
   delete h1X;
   delete h1Y;
   delete h1XOR;
   delete h1YOR;

   delete h1XR;
   delete h1YR;

   delete pe1XY;
   delete pe1XYOR;
   delete pe1XYR;

   delete pe1YX;
   delete pe1YXOR;
   delete pe1YXR;
}
