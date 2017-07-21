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

TEST(StressHistogram, TestTH3toTH2)
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

   TRandom2 r(initialRandomSeed);
   r.SetSeed(10);

   TH3D h3("h3", "h3", binsizeX, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit, binsizeZ, lower_limit,
           upper_limit);

   TH1::StatOverflows(kTRUE);

   TH2D h2XY("h2XY", "h2XY", binsizeX, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit);
   TH2D h2XZ("h2XZ", "h2XZ", binsizeX, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit);
   TH2D h2YX("h2YX", "h2YX", binsizeY, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit);
   TH2D h2YZ("h2YZ", "h2YZ", binsizeY, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit);
   TH2D h2ZX("h2ZX", "h2ZX", binsizeZ, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit);
   TH2D h2ZY("h2ZY", "h2ZY", binsizeZ, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit);

   TH2D h2XYR("h2XYR", "h2XYR", maxbinX - minbinX + 1, h3.GetXaxis()->GetBinLowEdge(minbinX),
              h3.GetXaxis()->GetBinUpEdge(maxbinX), maxbinY - minbinY + 1, h3.GetYaxis()->GetBinLowEdge(minbinY),
              h3.GetYaxis()->GetBinUpEdge(maxbinY));
   TH2D h2XZR("h2XZR", "h2XZR", maxbinX - minbinX + 1, h3.GetXaxis()->GetBinLowEdge(minbinX),
              h3.GetXaxis()->GetBinUpEdge(maxbinX), maxbinZ - minbinZ + 1, h3.GetZaxis()->GetBinLowEdge(minbinZ),
              h3.GetZaxis()->GetBinUpEdge(maxbinZ));
   TH2D h2YXR("h2YXR", "h2YXR", maxbinY - minbinY + 1, h3.GetYaxis()->GetBinLowEdge(minbinY),
              h3.GetYaxis()->GetBinUpEdge(maxbinY), maxbinX - minbinX + 1, h3.GetXaxis()->GetBinLowEdge(minbinX),
              h3.GetXaxis()->GetBinUpEdge(maxbinX));
   TH2D h2YZR("h2YZR", "h2YZR", maxbinY - minbinY + 1, h3.GetYaxis()->GetBinLowEdge(minbinY),
              h3.GetYaxis()->GetBinUpEdge(maxbinY), maxbinZ - minbinZ + 1, h3.GetZaxis()->GetBinLowEdge(minbinZ),
              h3.GetZaxis()->GetBinUpEdge(maxbinZ));
   TH2D h2ZXR("h2ZXR", "h2ZXR", maxbinZ - minbinZ + 1, h3.GetZaxis()->GetBinLowEdge(minbinZ),
              h3.GetZaxis()->GetBinUpEdge(maxbinZ), maxbinX - minbinX + 1, h3.GetXaxis()->GetBinLowEdge(minbinX),
              h3.GetXaxis()->GetBinUpEdge(maxbinX));
   TH2D h2ZYR("h2ZYR", "h2ZYR", maxbinZ - minbinZ + 1, h3.GetZaxis()->GetBinLowEdge(minbinZ),
              h3.GetZaxis()->GetBinUpEdge(maxbinZ), maxbinY - minbinY + 1, h3.GetYaxis()->GetBinLowEdge(minbinY),
              h3.GetYaxis()->GetBinUpEdge(maxbinY));

   TH2D h2XYOR("h2XYOR", "h2XYOR", binsizeX, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit);
   TH2D h2XZOR("h2XZOR", "h2XZOR", binsizeX, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit);
   TH2D h2YXOR("h2YXOR", "h2YXOR", binsizeY, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit);
   TH2D h2YZOR("h2YZOR", "h2YZOR", binsizeY, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit);
   TH2D h2ZXOR("h2ZXOR", "h2ZXOR", binsizeZ, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit);
   TH2D h2ZYOR("h2ZYOR", "h2ZYOR", binsizeZ, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit);

   TProfile2D pe2XY("pe2XY", "pe2XY", binsizeX, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit);
   TProfile2D pe2XZ("pe2XZ", "pe2XZ", binsizeX, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit);
   TProfile2D pe2YX("pe2YX", "pe2YX", binsizeY, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit);
   TProfile2D pe2YZ("pe2YZ", "pe2YZ", binsizeY, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit);
   TProfile2D pe2ZX("pe2ZX", "pe2ZX", binsizeZ, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit);
   TProfile2D pe2ZY("pe2ZY", "pe2ZY", binsizeZ, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit);

   TProfile2D pe2XYR("pe2XYR", "pe2XYR", maxbinX - minbinX + 1, h3.GetXaxis()->GetBinLowEdge(minbinX),
                     h3.GetXaxis()->GetBinUpEdge(maxbinX), maxbinY - minbinY + 1, h3.GetYaxis()->GetBinLowEdge(minbinY),
                     h3.GetYaxis()->GetBinUpEdge(maxbinY));
   TProfile2D pe2XZR("pe2XZR", "pe2XZR", maxbinX - minbinX + 1, h3.GetXaxis()->GetBinLowEdge(minbinX),
                     h3.GetXaxis()->GetBinUpEdge(maxbinX), maxbinZ - minbinZ + 1, h3.GetZaxis()->GetBinLowEdge(minbinZ),
                     h3.GetZaxis()->GetBinUpEdge(maxbinZ));
   TProfile2D pe2YXR("pe2YXR", "pe2YXR", maxbinY - minbinY + 1, h3.GetYaxis()->GetBinLowEdge(minbinY),
                     h3.GetYaxis()->GetBinUpEdge(maxbinY), maxbinX - minbinX + 1, h3.GetXaxis()->GetBinLowEdge(minbinX),
                     h3.GetXaxis()->GetBinUpEdge(maxbinX));
   TProfile2D pe2YZR("pe2YZR", "pe2YZR", maxbinY - minbinY + 1, h3.GetYaxis()->GetBinLowEdge(minbinY),
                     h3.GetYaxis()->GetBinUpEdge(maxbinY), maxbinZ - minbinZ + 1, h3.GetZaxis()->GetBinLowEdge(minbinZ),
                     h3.GetZaxis()->GetBinUpEdge(maxbinZ));
   TProfile2D pe2ZXR("pe2ZXR", "pe2ZXR", maxbinZ - minbinZ + 1, h3.GetZaxis()->GetBinLowEdge(minbinZ),
                     h3.GetZaxis()->GetBinUpEdge(maxbinZ), maxbinX - minbinX + 1, h3.GetXaxis()->GetBinLowEdge(minbinX),
                     h3.GetXaxis()->GetBinUpEdge(maxbinX));
   TProfile2D pe2ZYR("pe2ZYR", "pe2ZYR", maxbinZ - minbinZ + 1, h3.GetZaxis()->GetBinLowEdge(minbinZ),
                     h3.GetZaxis()->GetBinUpEdge(maxbinZ), maxbinY - minbinY + 1, h3.GetYaxis()->GetBinLowEdge(minbinY),
                     h3.GetYaxis()->GetBinUpEdge(maxbinY));

   TProfile2D pe2XYOR("pe2XYOR", "pe2XYOR", binsizeX, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit);
   TProfile2D pe2XZOR("pe2XZOR", "pe2XZOR", binsizeX, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit);
   TProfile2D pe2YXOR("pe2YXOR", "pe2YXOR", binsizeY, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit);
   TProfile2D pe2YZOR("pe2YZOR", "pe2YZOR", binsizeY, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit);
   TProfile2D pe2ZXOR("pe2ZXOR", "pe2ZXOR", binsizeZ, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit);
   TProfile2D pe2ZYOR("pe2ZYOR", "pe2ZYOR", binsizeZ, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit);

   for (int ix = 0; ix <= h3.GetXaxis()->GetNbins() + 1; ++ix) {
      double xc = h3.GetXaxis()->GetBinCenter(ix);
      double x = xc + centre_deviation * h3.GetXaxis()->GetBinWidth(ix);
      for (int iy = 0; iy <= h3.GetYaxis()->GetNbins() + 1; ++iy) {
         double yc = h3.GetYaxis()->GetBinCenter(iy);
         double y = yc + centre_deviation * h3.GetYaxis()->GetBinWidth(iy);
         for (int iz = 0; iz <= h3.GetZaxis()->GetNbins() + 1; ++iz) {
            double zc = h3.GetZaxis()->GetBinCenter(iz);
            double z = zc + centre_deviation * h3.GetZaxis()->GetBinWidth(iz);

            //    for ( int ix = 0; ix <= h3.GetXaxis()->GetNbins() + 1; ++ix ) {
            //       double x = centre_deviation * h3.GetXaxis()->GetBinWidth(ix) + h3.GetXaxis()->GetBinCenter(ix);
            //       for ( int iy = 0; iy <= h3.GetYaxis()->GetNbins() + 1; ++iy ) {
            //          double y = centre_deviation * h3.GetYaxis()->GetBinWidth(iy) +
            //          h3.GetYaxis()->GetBinCenter(iy);
            //          for ( int iz = 0; iz <= h3.GetZaxis()->GetNbins() + 1; ++iz ) {
            //             double z = centre_deviation * h3.GetZaxis()->GetBinWidth(iz) +
            //             h3.GetZaxis()->GetBinCenter(iz);
            Double_t w = (Double_t)r.Uniform(1, 3);

            h3.Fill(x, y, z, w);

            h2XY.Fill(x, y, w);
            h2XZ.Fill(x, z, w);
            h2YX.Fill(y, x, w);
            h2YZ.Fill(y, z, w);
            h2ZX.Fill(z, x, w);
            h2ZY.Fill(z, y, w);

            pe2XY.Fill(xc, yc, zc, w);
            pe2XZ.Fill(xc, zc, yc, w);
            pe2YX.Fill(yc, xc, zc, w);
            pe2YZ.Fill(yc, zc, xc, w);
            pe2ZX.Fill(zc, xc, yc, w);
            pe2ZY.Fill(zc, yc, xc, w);

            if (x >= h3.GetXaxis()->GetBinLowEdge(minbinX) && x <= h3.GetXaxis()->GetBinUpEdge(maxbinX) &&
                y >= h3.GetYaxis()->GetBinLowEdge(minbinY) && y <= h3.GetYaxis()->GetBinUpEdge(maxbinY) &&
                z >= h3.GetZaxis()->GetBinLowEdge(minbinZ) && z <= h3.GetZaxis()->GetBinUpEdge(maxbinZ)) {
               h2XYR.Fill(x, y, w);
               h2XZR.Fill(x, z, w);
               h2YXR.Fill(y, x, w);
               h2YZR.Fill(y, z, w);
               h2ZXR.Fill(z, x, w);
               h2ZYR.Fill(z, y, w);

               h2XYOR.Fill(x, y, w);
               h2XZOR.Fill(x, z, w);
               h2YXOR.Fill(y, x, w);
               h2YZOR.Fill(y, z, w);
               h2ZXOR.Fill(z, x, w);
               h2ZYOR.Fill(z, y, w);

               pe2XYR.Fill(xc, yc, zc, w);
               pe2XZR.Fill(xc, zc, yc, w);
               pe2YXR.Fill(yc, xc, zc, w);
               pe2YZR.Fill(yc, zc, xc, w);
               pe2ZXR.Fill(zc, xc, yc, w);
               pe2ZYR.Fill(zc, yc, xc, w);

               pe2XYOR.Fill(xc, yc, zc, w);
               pe2XZOR.Fill(xc, zc, yc, w);
               pe2YXOR.Fill(yc, xc, zc, w);
               pe2YZOR.Fill(yc, zc, xc, w);
               pe2ZXOR.Fill(zc, xc, yc, w);
               pe2ZYOR.Fill(zc, yc, xc, w);
            }
         }
      }
   }

   int options = cmpOptStats;

   options = cmpOptStats;
   unique_ptr<TH2D> projected((TH2D *)h3.Project3D("yx"));
   EXPECT_EQ(0, Equals("TH3 -> XY", h2XY, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("zx"));
   EXPECT_EQ(0, Equals("TH3 -> XZ", h2XZ, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("XY"));
   EXPECT_EQ(0, Equals("TH3 -> YX", h2YX, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("ZY"));
   EXPECT_EQ(0, Equals("TH3 -> YZ", h2YZ, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("XZ"));
   EXPECT_EQ(0, Equals("TH3 -> ZX", h2ZX, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("YZ"));
   EXPECT_EQ(0, Equals("TH3 -> ZY", h2ZY, *projected.get(), options));

   options = cmpOptStats;
   projected.reset((TH2D *)h3.Project3D("oyx"));
   EXPECT_EQ(0, Equals("TH3O -> XY", h2XY, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("ozx"));
   EXPECT_EQ(0, Equals("TH3O -> XZ", h2XZ, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("oXY"));
   EXPECT_EQ(0, Equals("TH3O -> YX", h2YX, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("oZY"));
   EXPECT_EQ(0, Equals("TH3O -> YZ", h2YZ, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("oXZ"));
   EXPECT_EQ(0, Equals("TH3O -> ZX", h2ZX, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("oYZ"));
   EXPECT_EQ(0, Equals("TH3O -> ZY", h2ZY, *projected.get(), options));

   options = cmpOptStats;
   projected.reset((TH2D *)h3.Project3DProfile("yx  UF OF"));
   EXPECT_EQ(0, Equals("TH3 -> PXY", pe2XY, *projected.get(), options));

   projected.reset(h3.Project3DProfile("zx  UF OF"));
   EXPECT_EQ(0, Equals("TH3 -> PXZ", pe2XZ, *projected.get(), options));

   projected.reset(h3.Project3DProfile("xy  UF OF"));
   EXPECT_EQ(0, Equals("TH3 -> PYX", pe2YX, *projected.get(), options));

   projected.reset(h3.Project3DProfile("zy  UF OF"));
   EXPECT_EQ(0, Equals("TH3 -> PYZ", pe2YZ, *projected.get(), options));

   projected.reset(h3.Project3DProfile("xz  UF OF"));
   EXPECT_EQ(0, Equals("TH3 -> PZX", pe2ZX, *projected.get(), options));

   projected.reset(h3.Project3DProfile("yz  UF OF"));
   EXPECT_EQ(0, Equals("TH3 -> PZY", pe2ZY, *projected.get(), options));

   options = cmpOptStats;
   projected.reset(h3.Project3DProfile("oyx  UF OF"));
   EXPECT_EQ(0, Equals("TH3O -> PXY", pe2XY, *projected.get(), options));
   projected.reset(h3.Project3DProfile("ozx  UF OF"));
   EXPECT_EQ(0, Equals("TH3O -> PXZ", pe2XZ, *projected.get(), options));
   projected.reset(h3.Project3DProfile("oxy  UF OF"));
   EXPECT_EQ(0, Equals("TH3O -> PYX", pe2YX, *projected.get(), options));
   projected.reset(h3.Project3DProfile("ozy  UF OF"));
   EXPECT_EQ(0, Equals("TH3O -> PYZ", pe2YZ, *projected.get(), options));
   projected.reset(h3.Project3DProfile("oxz  UF OF"));
   EXPECT_EQ(0, Equals("TH3O -> PZX", pe2ZX, *projected.get(), options));
   projected.reset(h3.Project3DProfile("oyz  UF OF"));
   EXPECT_EQ(0, Equals("TH3O -> PZY", pe2ZY, *projected.get(), options));

   h3.GetXaxis()->SetRange(minbinX, maxbinX);
   h3.GetYaxis()->SetRange(minbinY, maxbinY);
   h3.GetZaxis()->SetRange(minbinZ, maxbinZ);

   // Stats won't work here, unless centre_deviation == 0.0
   options = 0;
   projected.reset((TH2D *)h3.Project3D("yx"));
   EXPECT_EQ(0, Equals("TH3R -> XY", h2XYR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("zx"));
   EXPECT_EQ(0, Equals("TH3R -> XZ", h2XZR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("XY"));
   EXPECT_EQ(0, Equals("TH3R -> YX", h2YXR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("ZY"));
   EXPECT_EQ(0, Equals("TH3R -> YZ", h2YZR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("XZ"));
   EXPECT_EQ(0, Equals("TH3R -> ZX", h2ZXR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("YZ"));
   EXPECT_EQ(0, Equals("TH3R -> ZY", h2ZYR, *projected.get(), options));

   // Stats won't work here, unless centre_deviation == 0.0
   options = 0;
   projected.reset((TH2D *)h3.Project3D("oyx"));
   EXPECT_EQ(0, Equals("TH3OR -> XY", h2XYOR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("ozx"));
   EXPECT_EQ(0, Equals("TH3OR -> XZ", h2XZOR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("oXY"));
   EXPECT_EQ(0, Equals("TH3OR -> YX", h2YXOR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("oZY"));
   EXPECT_EQ(0, Equals("TH3OR -> YZ", h2YZOR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("oXZ"));
   EXPECT_EQ(0, Equals("TH3OR -> ZX", h2ZXOR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3D("oYZ"));
   EXPECT_EQ(0, Equals("TH3OR -> ZY", h2ZYOR, *projected.get(), options));

   options = cmpOptStats;
   projected.reset((TH2D *)h3.Project3DProfile("yx  UF OF"));
   EXPECT_EQ(0, Equals("TH3R -> PXY", pe2XYR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3DProfile("zx  UF OF"));
   EXPECT_EQ(0, Equals("TH3R -> PXZ", pe2XZR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3DProfile("xy  UF OF"));
   EXPECT_EQ(0, Equals("TH3R -> PYX", pe2YXR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3DProfile("zy  UF OF"));
   EXPECT_EQ(0, Equals("TH3R -> PYZ", pe2YZR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3DProfile("xz  UF OF"));
   EXPECT_EQ(0, Equals("TH3R -> PZX", pe2ZXR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3DProfile("yz  UF OF"));
   EXPECT_EQ(0, Equals("TH3R -> PZY", pe2ZYR, *projected.get(), options));

   options = cmpOptStats;
   projected.reset((TH2D *)h3.Project3DProfile("oyx  UF OF"));
   EXPECT_EQ(0, Equals("TH3OR -> PXY", pe2XYOR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3DProfile("ozx  UF OF"));
   EXPECT_EQ(0, Equals("TH3OR -> PXZ", pe2XZOR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3DProfile("oxy  UF OF"));
   EXPECT_EQ(0, Equals("TH3OR -> PYX", pe2YXOR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3DProfile("ozy  UF OF"));
   EXPECT_EQ(0, Equals("TH3OR -> PYZ", pe2YZOR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3DProfile("oxz  UF OF"));
   EXPECT_EQ(0, Equals("TH3OR -> PZX", pe2ZXOR, *projected.get(), options));
   projected.reset((TH2D *)h3.Project3DProfile("oyz  UF OF"));
   EXPECT_EQ(0, Equals("TH3OR -> PZY", pe2ZYOR, *projected.get(), options));
}
