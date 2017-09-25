// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include "StressHistogramGlobal.h"

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TRandom2.h"

#include <sstream>
#include "gtest/gtest.h"

using namespace std;

TEST(StressHistogram, TestTH3toTH1)
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

   TRandom2 r;
   r.SetSeed(10);

   TH3D h3("h3", "h3", binsizeX, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit, binsizeZ, lower_limit,
           upper_limit);

   TH1::StatOverflows(kTRUE);

   TH1D h1X("h1X", "h1X", binsizeX, lower_limit, upper_limit);
   TH1D h1Y("h1Y", "h1Y", binsizeY, lower_limit, upper_limit);
   TH1D h1Z("h1Z", "h1Z", binsizeZ, lower_limit, upper_limit);

   TH1D h1XR("h1XR", "h1XR", maxbinX - minbinX + 1, h1X.GetXaxis()->GetBinLowEdge(minbinX),
             h1X.GetXaxis()->GetBinUpEdge(maxbinX));
   TH1D h1YR("h1YR", "h1YR", maxbinY - minbinY + 1, h1Y.GetXaxis()->GetBinLowEdge(minbinY),
             h1Y.GetXaxis()->GetBinUpEdge(maxbinY));
   TH1D h1ZR("h1ZR", "h1ZR", maxbinZ - minbinZ + 1, h1Z.GetXaxis()->GetBinLowEdge(minbinZ),
             h1Z.GetXaxis()->GetBinUpEdge(maxbinZ));

   TH1D h1XOR("h1XOR", "h1XOR", binsizeX, lower_limit, upper_limit);
   TH1D h1YOR("h1YOR", "h1YOR", binsizeY, lower_limit, upper_limit);
   TH1D h1ZOR("h1ZOR", "h1ZOR", binsizeZ, lower_limit, upper_limit);

   h3.Sumw2();

   for (int ix = 0; ix <= h3.GetXaxis()->GetNbins() + 1; ++ix) {
      double x = centre_deviation * h3.GetXaxis()->GetBinWidth(ix) + h3.GetXaxis()->GetBinCenter(ix);
      for (int iy = 0; iy <= h3.GetYaxis()->GetNbins() + 1; ++iy) {
         double y = centre_deviation * h3.GetYaxis()->GetBinWidth(iy) + h3.GetYaxis()->GetBinCenter(iy);
         for (int iz = 0; iz <= h3.GetZaxis()->GetNbins() + 1; ++iz) {
            double z = centre_deviation * h3.GetZaxis()->GetBinWidth(iz) + h3.GetZaxis()->GetBinCenter(iz);
            Double_t w = (Double_t)r.Uniform(1, 3);

            h3.Fill(x, y, z, w);

            h1X.Fill(x, w);
            h1Y.Fill(y, w);
            h1Z.Fill(z, w);

            if (x >= h1X.GetXaxis()->GetBinLowEdge(minbinX) && x <= h1X.GetXaxis()->GetBinUpEdge(maxbinX) &&
                y >= h1Y.GetXaxis()->GetBinLowEdge(minbinY) && y <= h1Y.GetXaxis()->GetBinUpEdge(maxbinY) &&
                z >= h1Z.GetXaxis()->GetBinLowEdge(minbinZ) && z <= h1Z.GetXaxis()->GetBinUpEdge(maxbinZ)) {
               h1XR.Fill(x, w);
               h1YR.Fill(y, w);
               h1ZR.Fill(z, w);
               h1XOR.Fill(x, w);
               h1YOR.Fill(y, w);
               h1ZOR.Fill(z, w);
            }
         }
      }
   }

   int options = cmpOptStats;

   unique_ptr<TH1D> tmp1;
   unique_ptr<TH1D> projection((TH1D *)h3.Project3D("x"));

   options = cmpOptStats;
   EXPECT_EQ(0, Equals("TH3 -> X", h1X, *projection.get(), options));
   tmp1.reset(h3.ProjectionX("x335"));
   projection.reset((TH1D *)h3.Project3D("x2"));
   EXPECT_EQ(0, Equals("TH3 -> X(x2)", *tmp1.get(), *projection.get(), options));
   projection.reset((TH1D *)h3.Project3D("y"));
   EXPECT_EQ(0, Equals("TH3 -> Y", h1Y, *projection.get(), options));
   tmp1.reset(h3.ProjectionY("y335"));
   projection.reset((TH1D *)h3.Project3D("y2"));
   EXPECT_EQ(0, Equals("TH3 -> Y(x2)", *tmp1.get(), *projection.get(), options));
   projection.reset((TH1D *)h3.Project3D("z"));
   EXPECT_EQ(0, Equals("TH3 -> Z", h1Z, *projection.get(), options));
   tmp1.reset(h3.ProjectionZ("z335"));
   projection.reset((TH1D *)h3.Project3D("z2"));
   EXPECT_EQ(0, Equals("TH3 -> Z(x2)", *tmp1.get(), *projection.get(), options));

   options = cmpOptStats;
   projection.reset((TH1D *)h3.Project3D("ox"));
   EXPECT_EQ(0, Equals("TH3O -> X", h1X, *projection.get(), options));
   tmp1.reset(h3.ProjectionX("x1335"));
   projection.reset((TH1D *)h3.Project3D("ox2"));
   EXPECT_EQ(0, Equals("TH3O -> X(x2)", *tmp1.get(), *projection.get(), options));
   projection.reset((TH1D *)h3.Project3D("oy"));
   EXPECT_EQ(0, Equals("TH3O -> Y", h1Y, *projection.get(), options));
   tmp1.reset(h3.ProjectionY("y1335"));
   projection.reset((TH1D *)h3.Project3D("oy2"));
   EXPECT_EQ(0, Equals("TH3O -> Y(x2)", *tmp1.get(), *projection.get(), options));
   projection.reset((TH1D *)h3.Project3D("oz"));
   EXPECT_EQ(0, Equals("TH3O -> Z", h1Z, *projection.get(), options));
   tmp1.reset(h3.ProjectionZ("z1335"));
   projection.reset((TH1D *)h3.Project3D("oz2"));
   EXPECT_EQ(0, Equals("TH3O -> Z(x2)", *tmp1.get(), *projection.get(), options));

   h3.GetXaxis()->SetRange(minbinX, maxbinX);
   h3.GetYaxis()->SetRange(minbinY, maxbinY);
   h3.GetZaxis()->SetRange(minbinZ, maxbinZ);

   h1X.GetXaxis()->SetRange(minbinX, maxbinX);
   h1Y.GetXaxis()->SetRange(minbinY, maxbinY);
   h1Z.GetXaxis()->SetRange(minbinZ, maxbinZ);

   // Statistics are no longer conserved if the center_deviation != 0.0
   options = 0;
   projection.reset((TH1D *)h3.Project3D("x34"));
   EXPECT_EQ(0, Equals("TH3R -> X", h1XR, *projection.get(), options));
   tmp1.reset(h3.ProjectionX("x3335", minbinY, maxbinY, minbinZ, maxbinZ));
   projection.reset((TH1D *)h3.Project3D("x22"));
   EXPECT_EQ(0, Equals("TH3R -> X(x2)", *tmp1.get(), *projection.get(), options));
   projection.reset((TH1D *)h3.Project3D("y34"));
   EXPECT_EQ(0, Equals("TH3R -> Y", h1YR, *projection.get(), options));
   tmp1.reset(h3.ProjectionY("y3335", minbinX, maxbinX, minbinZ, maxbinZ));
   projection.reset((TH1D *)h3.Project3D("y22"));
   EXPECT_EQ(0, Equals("TH3R -> Y(x2)", *tmp1.get(), *projection.get(), options));
   projection.reset((TH1D *)h3.Project3D("z34"));
   EXPECT_EQ(0, Equals("TH3R -> Z", h1ZR, *projection.get(), options));
   tmp1.reset(h3.ProjectionZ("z3335", minbinX, maxbinX, minbinY, maxbinY));
   projection.reset((TH1D *)h3.Project3D("z22"));
   EXPECT_EQ(0, Equals("TH3R -> Z(x2)", *tmp1.get(), *projection.get(), options));

   options = 0;
   projection.reset((TH1D *)h3.Project3D("ox"));
   EXPECT_EQ(0, Equals("TH3RO -> X", h1XOR, *projection.get(), options));
   tmp1.reset(h3.ProjectionX("x1335", minbinY, maxbinY, minbinZ, maxbinZ, "o"));
   projection.reset((TH1D *)h3.Project3D("ox2"));
   EXPECT_EQ(0, Equals("TH3RO-> X(x2)", *tmp1.get(), *projection.get(), options));
   projection.reset((TH1D *)h3.Project3D("oy"));
   EXPECT_EQ(0, Equals("TH3RO -> Y", h1YOR, *projection.get(), options));
   tmp1.reset(h3.ProjectionY("y1335", minbinX, maxbinX, minbinZ, maxbinZ, "o"));
   projection.reset((TH1D *)h3.Project3D("oy2"));
   EXPECT_EQ(0, Equals("TH3RO-> Y(x2)", *tmp1.get(), *projection.get(), options));
   projection.reset((TH1D *)h3.Project3D("oz"));
   EXPECT_EQ(0, Equals("TH3RO-> Z", h1ZOR, *projection.get(), options));
   tmp1.reset(h3.ProjectionZ("z1335", minbinX, maxbinX, minbinY, maxbinY, "o"));
   projection.reset((TH1D *)h3.Project3D("oz2"));
   EXPECT_EQ(0, Equals("TH3RO-> Z(x2)", *tmp1.get(), *projection.get(), options));
}
