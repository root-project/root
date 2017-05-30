// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TRandom2.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestTH3toTH1)
{
   const unsigned int binsizeX = 10;
   const unsigned int binsizeY = 11;
   const unsigned int binsizeZ = 12;
   static const unsigned int minbinX = 2;
   static const unsigned int maxbinX = 5;
   static const unsigned int minbinY = 3;
   static const unsigned int maxbinY = 8;
   static const unsigned int minbinZ = 4;
   static const unsigned int maxbinZ = 10;
   const int lower_limit = 0;
   const int upper_limit = 10;

   r.SetSeed(10);

   TH3D *h3 = new TH3D("h3", "h3", binsizeX, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit, binsizeZ,
                       lower_limit, upper_limit);

   TH1::StatOverflows(kTRUE);

   TH1D *h1X = new TH1D("h1X", "h1X", binsizeX, lower_limit, upper_limit);
   TH1D *h1Y = new TH1D("h1Y", "h1Y", binsizeY, lower_limit, upper_limit);
   TH1D *h1Z = new TH1D("h1Z", "h1Z", binsizeZ, lower_limit, upper_limit);

   TH1D *h1XR = new TH1D("h1XR", "h1XR", maxbinX - minbinX + 1, h1X->GetXaxis()->GetBinLowEdge(minbinX),
                         h1X->GetXaxis()->GetBinUpEdge(maxbinX));
   TH1D *h1YR = new TH1D("h1YR", "h1YR", maxbinY - minbinY + 1, h1Y->GetXaxis()->GetBinLowEdge(minbinY),
                         h1Y->GetXaxis()->GetBinUpEdge(maxbinY));
   TH1D *h1ZR = new TH1D("h1ZR", "h1ZR", maxbinZ - minbinZ + 1, h1Z->GetXaxis()->GetBinLowEdge(minbinZ),
                         h1Z->GetXaxis()->GetBinUpEdge(maxbinZ));

   TH1D *h1XOR = new TH1D("h1XOR", "h1XOR", binsizeX, lower_limit, upper_limit);
   TH1D *h1YOR = new TH1D("h1YOR", "h1YOR", binsizeY, lower_limit, upper_limit);
   TH1D *h1ZOR = new TH1D("h1ZOR", "h1ZOR", binsizeZ, lower_limit, upper_limit);

   h3->Sumw2();

   for (int ix = 0; ix <= h3->GetXaxis()->GetNbins() + 1; ++ix) {
      double x = centre_deviation * h3->GetXaxis()->GetBinWidth(ix) + h3->GetXaxis()->GetBinCenter(ix);
      for (int iy = 0; iy <= h3->GetYaxis()->GetNbins() + 1; ++iy) {
         double y = centre_deviation * h3->GetYaxis()->GetBinWidth(iy) + h3->GetYaxis()->GetBinCenter(iy);
         for (int iz = 0; iz <= h3->GetZaxis()->GetNbins() + 1; ++iz) {
            double z = centre_deviation * h3->GetZaxis()->GetBinWidth(iz) + h3->GetZaxis()->GetBinCenter(iz);
            Double_t w = (Double_t)r.Uniform(1, 3);

            h3->Fill(x, y, z, w);

            h1X->Fill(x, w);
            h1Y->Fill(y, w);
            h1Z->Fill(z, w);

            if (x >= h1X->GetXaxis()->GetBinLowEdge(minbinX) && x <= h1X->GetXaxis()->GetBinUpEdge(maxbinX) &&
                y >= h1Y->GetXaxis()->GetBinLowEdge(minbinY) && y <= h1Y->GetXaxis()->GetBinUpEdge(maxbinY) &&
                z >= h1Z->GetXaxis()->GetBinLowEdge(minbinZ) && z <= h1Z->GetXaxis()->GetBinUpEdge(maxbinZ)) {
               h1XR->Fill(x, w);
               h1YR->Fill(y, w);
               h1ZR->Fill(z, w);
               h1XOR->Fill(x, w);
               h1YOR->Fill(y, w);
               h1ZOR->Fill(z, w);
            }
         }
      }
   }

   int options = cmpOptStats;

   TH1D *tmp1 = 0;

   options = cmpOptStats;
   EXPECT_EQ(0, Equals("TH3 -> X", h1X, (TH1D *)h3->Project3D("x"), options));
   tmp1 = h3->ProjectionX("x335");
   EXPECT_EQ(0, Equals("TH3 -> X(x2)", tmp1, (TH1D *)h3->Project3D("x2"), options));
   delete tmp1;
   tmp1 = 0;
   EXPECT_EQ(0, Equals("TH3 -> Y", h1Y, (TH1D *)h3->Project3D("y"), options));
   tmp1 = h3->ProjectionY("y335");
   EXPECT_EQ(0, Equals("TH3 -> Y(x2)", tmp1, (TH1D *)h3->Project3D("y2"), options));
   delete tmp1;
   tmp1 = 0;
   EXPECT_EQ(0, Equals("TH3 -> Z", h1Z, (TH1D *)h3->Project3D("z"), options));
   tmp1 = h3->ProjectionZ("z335");
   EXPECT_EQ(0, Equals("TH3 -> Z(x2)", tmp1, (TH1D *)h3->Project3D("z2"), options));
   delete tmp1;
   tmp1 = 0;

   options = cmpOptStats;
   EXPECT_EQ(0, Equals("TH3O -> X", h1X, (TH1D *)h3->Project3D("ox"), options));
   tmp1 = h3->ProjectionX("x1335");
   EXPECT_EQ(0, Equals("TH3O -> X(x2)", tmp1, (TH1D *)h3->Project3D("ox2"), options));
   delete tmp1;
   tmp1 = 0;
   EXPECT_EQ(0, Equals("TH3O -> Y", h1Y, (TH1D *)h3->Project3D("oy"), options));
   tmp1 = h3->ProjectionY("y1335");
   EXPECT_EQ(0, Equals("TH3O -> Y(x2)", tmp1, (TH1D *)h3->Project3D("oy2"), options));
   delete tmp1;
   tmp1 = 0;
   EXPECT_EQ(0, Equals("TH3O -> Z", h1Z, (TH1D *)h3->Project3D("oz"), options));
   tmp1 = h3->ProjectionZ("z1335");
   EXPECT_EQ(0, Equals("TH3O -> Z(x2)", tmp1, (TH1D *)h3->Project3D("oz2"), options));
   delete tmp1;
   tmp1 = 0;

   h3->GetXaxis()->SetRange(minbinX, maxbinX);
   h3->GetYaxis()->SetRange(minbinY, maxbinY);
   h3->GetZaxis()->SetRange(minbinZ, maxbinZ);

   h1X->GetXaxis()->SetRange(minbinX, maxbinX);
   h1Y->GetXaxis()->SetRange(minbinY, maxbinY);
   h1Z->GetXaxis()->SetRange(minbinZ, maxbinZ);

   // Statistics are no longer conserved if the center_deviation != 0.0
   options = 0;
   EXPECT_EQ(0, Equals("TH3R -> X", h1XR, (TH1D *)h3->Project3D("x34"), options));
   tmp1 = h3->ProjectionX("x3335", minbinY, maxbinY, minbinZ, maxbinZ);
   EXPECT_EQ(0, Equals("TH3R -> X(x2)", tmp1, (TH1D *)h3->Project3D("x22"), options));
   delete tmp1;
   tmp1 = 0;
   EXPECT_EQ(0, Equals("TH3R -> Y", h1YR, (TH1D *)h3->Project3D("y34"), options));
   tmp1 = h3->ProjectionY("y3335", minbinX, maxbinX, minbinZ, maxbinZ);
   EXPECT_EQ(0, Equals("TH3R -> Y(x2)", tmp1, (TH1D *)h3->Project3D("y22"), options));
   delete tmp1;
   tmp1 = 0;
   EXPECT_EQ(0, Equals("TH3R -> Z", h1ZR, (TH1D *)h3->Project3D("z34"), options));
   tmp1 = h3->ProjectionZ("z3335", minbinX, maxbinX, minbinY, maxbinY);
   EXPECT_EQ(0, Equals("TH3R -> Z(x2)", tmp1, (TH1D *)h3->Project3D("z22"), options));
   delete tmp1;
   tmp1 = 0;

   options = 0;
   EXPECT_EQ(0, Equals("TH3RO -> X", h1XOR, (TH1D *)h3->Project3D("ox"), options));
   tmp1 = h3->ProjectionX("x1335", minbinY, maxbinY, minbinZ, maxbinZ, "o");
   EXPECT_EQ(0, Equals("TH3RO-> X(x2)", tmp1, (TH1D *)h3->Project3D("ox2"), options));
   delete tmp1;
   tmp1 = 0;
   EXPECT_EQ(0, Equals("TH3RO -> Y", h1YOR, (TH1D *)h3->Project3D("oy"), options));
   tmp1 = h3->ProjectionY("y1335", minbinX, maxbinX, minbinZ, maxbinZ, "o");
   EXPECT_EQ(0, Equals("TH3RO-> Y(x2)", tmp1, (TH1D *)h3->Project3D("oy2"), options));
   delete tmp1;
   tmp1 = 0;
   EXPECT_EQ(0, Equals("TH3RO-> Z", h1ZOR, (TH1D *)h3->Project3D("oz"), options));
   tmp1 = h3->ProjectionZ("z1335", minbinX, maxbinX, minbinY, maxbinY, "o");
   EXPECT_EQ(0, Equals("TH3RO-> Z(x2)", tmp1, (TH1D *)h3->Project3D("oz2"), options));
   delete tmp1;
   tmp1 = 0;

   options = 0;

   delete h3;

   delete h1X;
   delete h1Y;
   delete h1Z;

   delete h1XR;
   delete h1YR;
   delete h1ZR;

   delete h1XOR;
   delete h1YOR;
   delete h1ZOR;
}
