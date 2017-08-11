// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include "StressHistogramGlobal.h"

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"
#include "TProfile2D.h"
#include "TProfile3D.h"

#include "TF1.h"

#include "Fit/SparseData.h"
#include "HFitInterface.h"

#include "Math/IntegratorOptions.h"

#include "Riostream.h"
#include "TApplication.h"
#include "TClass.h"
#include "TFile.h"
#include "TRandom2.h"

#include "TROOT.h"

#include <cmath>
#include <sstream>
#include "gtest/gtest.h"

using namespace std;

class ProjectionTester : public ::testing::Test {
   // This class implements the tests for all types of projections of
   // all the classes tested in this file.

   static const unsigned int binsizeX = 8;
   static const unsigned int binsizeY = 10;
   static const unsigned int binsizeZ = 12;
   static const int lower_limit = 0;
   static const int upper_limit = 10;
   static const int lower_limitX = 0;
   static const int upper_limitX = 10;
   static const int lower_limitY = -5;
   static const int upper_limitY = 10;
   static const int lower_limitZ = -10;
   static const int upper_limitZ = 10;

   unique_ptr<TH3D> h3;
   unique_ptr<TH2D> h2XY;
   unique_ptr<TH2D> h2XZ;
   unique_ptr<TH2D> h2YX;
   unique_ptr<TH2D> h2YZ;
   unique_ptr<TH2D> h2ZX;
   unique_ptr<TH2D> h2ZY;
   unique_ptr<TH1D> h1X;
   unique_ptr<TH1D> h1Y;
   unique_ptr<TH1D> h1Z;

   unique_ptr<TH1D> h1XStats;
   unique_ptr<TH1D> h1YStats;
   unique_ptr<TH1D> h1ZStats;

   unique_ptr<TProfile2D> pe2XY;
   unique_ptr<TProfile2D> pe2XZ;
   unique_ptr<TProfile2D> pe2YX;
   unique_ptr<TProfile2D> pe2YZ;
   unique_ptr<TProfile2D> pe2ZX;
   unique_ptr<TProfile2D> pe2ZY;

   unique_ptr<TH2D> h2wXY;
   unique_ptr<TH2D> h2wXZ;
   unique_ptr<TH2D> h2wYX;
   unique_ptr<TH2D> h2wYZ;
   unique_ptr<TH2D> h2wZX;
   unique_ptr<TH2D> h2wZY;

   unique_ptr<TProfile> pe1XY;
   unique_ptr<TProfile> pe1XZ;
   unique_ptr<TProfile> pe1YX;
   unique_ptr<TProfile> pe1YZ;
   unique_ptr<TProfile> pe1ZX;
   unique_ptr<TProfile> pe1ZY;

   unique_ptr<TH1D> hw1XZ;
   unique_ptr<TH1D> hw1XY;
   unique_ptr<TH1D> hw1YX;
   unique_ptr<TH1D> hw1YZ;
   unique_ptr<TH1D> hw1ZX;
   unique_ptr<TH1D> hw1ZY;

   unique_ptr<TProfile3D> p3;

   unique_ptr<TProfile2D> p2XY;
   unique_ptr<TProfile2D> p2XZ;
   unique_ptr<TProfile2D> p2YX;
   unique_ptr<TProfile2D> p2YZ;
   unique_ptr<TProfile2D> p2ZX;
   unique_ptr<TProfile2D> p2ZY;

   unique_ptr<TProfile> p1X;
   unique_ptr<TProfile> p1Y;
   unique_ptr<TProfile> p1Z;

   unique_ptr<THnSparseD> s3;
   unique_ptr<THnD> n3;

   bool buildWithWeights = false;
   TRandom2 r = TRandom2();

private:
   void CreateHistograms()
   {
      h3.reset(new TH3D("h3", "h3", binsizeX, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit, binsizeZ,
                        lower_limit, upper_limit));

      h2XY.reset(new TH2D("h2XY", "h2XY", binsizeX, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit));
      h2XZ.reset(new TH2D("h2XZ", "h2XZ", binsizeX, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit));
      h2YX.reset(new TH2D("h2YX", "h2YX", binsizeY, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit));
      h2YZ.reset(new TH2D("h2YZ", "h2YZ", binsizeY, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit));
      h2ZX.reset(new TH2D("h2ZX", "h2ZX", binsizeZ, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit));
      h2ZY.reset(new TH2D("h2ZY", "h2ZY", binsizeZ, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit));

      // The bit is set for all the histograms (It's a statistic variable)
      TH1::StatOverflows(kTRUE);

      h1X.reset(new TH1D("h1X", "h1X", binsizeX, lower_limit, upper_limit));
      h1Y.reset(new TH1D("h1Y", "h1Y", binsizeY, lower_limit, upper_limit));
      h1Z.reset(new TH1D("h1Z", "h1Z", binsizeZ, lower_limit, upper_limit));

      h1XStats.reset(new TH1D("h1XStats", "h1XStats", binsizeX, lower_limit, upper_limit));
      h1YStats.reset(new TH1D("h1YStats", "h1YStats", binsizeY, lower_limit, upper_limit));
      h1ZStats.reset(new TH1D("h1ZStats", "h1ZStats", binsizeZ, lower_limit, upper_limit));

      pe2XY.reset(
         new TProfile2D("pe2XY", "pe2XY", binsizeX, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit));
      pe2XZ.reset(
         new TProfile2D("pe2XZ", "pe2XZ", binsizeX, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit));
      pe2YX.reset(
         new TProfile2D("pe2YX", "pe2YX", binsizeY, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit));
      pe2YZ.reset(
         new TProfile2D("pe2YZ", "pe2YZ", binsizeY, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit));
      pe2ZX.reset(
         new TProfile2D("pe2ZX", "pe2ZX", binsizeZ, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit));
      pe2ZY.reset(
         new TProfile2D("pe2ZY", "pe2ZY", binsizeZ, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit));

      h2wXY.reset(new TH2D("h2wXY", "h2wXY", binsizeX, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit));
      h2wXZ.reset(new TH2D("h2wXZ", "h2wXZ", binsizeX, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit));
      h2wYX.reset(new TH2D("h2wYX", "h2wYX", binsizeY, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit));
      h2wYZ.reset(new TH2D("h2wYZ", "h2wYZ", binsizeY, lower_limit, upper_limit, binsizeZ, lower_limit, upper_limit));
      h2wZX.reset(new TH2D("h2wZX", "h2wZX", binsizeZ, lower_limit, upper_limit, binsizeX, lower_limit, upper_limit));
      h2wZY.reset(new TH2D("h2wZY", "h2wZY", binsizeZ, lower_limit, upper_limit, binsizeY, lower_limit, upper_limit));

      h2wXY->Sumw2();
      h2wXZ->Sumw2();
      h2wYX->Sumw2();
      h2wYZ->Sumw2();
      h2wZX->Sumw2();
      h2wZY->Sumw2();

      pe1XY.reset(new TProfile("pe1XY", "pe1XY", binsizeX, lower_limit, upper_limit));
      pe1XZ.reset(new TProfile("pe1XZ", "pe1XZ", binsizeX, lower_limit, upper_limit));
      pe1YX.reset(new TProfile("pe1YX", "pe1YX", binsizeY, lower_limit, upper_limit));
      pe1YZ.reset(new TProfile("pe1YZ", "pe1YZ", binsizeY, lower_limit, upper_limit));
      pe1ZX.reset(new TProfile("pe1ZX", "pe1ZX", binsizeZ, lower_limit, upper_limit));
      pe1ZY.reset(new TProfile("pe1ZY", "pe1ZY", binsizeZ, lower_limit, upper_limit));

      hw1XY.reset(new TH1D("hw1XY", "hw1XY", binsizeX, lower_limit, upper_limit));
      hw1XZ.reset(new TH1D("hw1XZ", "hw1XZ", binsizeX, lower_limit, upper_limit));
      hw1YX.reset(new TH1D("hw1YX", "hw1YX", binsizeY, lower_limit, upper_limit));
      hw1YZ.reset(new TH1D("hw1YZ", "hw1YZ", binsizeY, lower_limit, upper_limit));
      hw1ZX.reset(new TH1D("hw1ZX", "hw1ZX", binsizeZ, lower_limit, upper_limit));
      hw1ZY.reset(new TH1D("hw1ZY", "hw1ZY", binsizeZ, lower_limit, upper_limit));

      hw1XZ->Sumw2();
      hw1XY->Sumw2();
      hw1YX->Sumw2();
      hw1YZ->Sumw2();
      hw1ZX->Sumw2();
      hw1ZY->Sumw2();

      Int_t bsize[] = {binsizeX, binsizeY, binsizeZ};
      Double_t xmin[] = {lower_limit, lower_limit, lower_limit};
      Double_t xmax[] = {upper_limit, upper_limit, upper_limit};
      s3.reset(new THnSparseD("s3", "s3", 3, bsize, xmin, xmax));
      n3.reset(new THnD("n3", "n3", 3, bsize, xmin, xmax));
   }

   void CreateProfiles()
   {

      // create Profile histograms
      p3.reset(new TProfile3D("p3", "p3", binsizeX, lower_limitX, upper_limitX, binsizeY, lower_limitY, upper_limitY,
                              binsizeZ, lower_limitZ, upper_limitZ));

      p2XY.reset(
         new TProfile2D("p2XY", "p2XY", binsizeX, lower_limitX, upper_limitX, binsizeY, lower_limitY, upper_limitY));
      p2XZ.reset(
         new TProfile2D("p2XZ", "p2XZ", binsizeX, lower_limitX, upper_limitX, binsizeZ, lower_limitZ, upper_limitZ));
      p2YX.reset(
         new TProfile2D("p2YX", "p2YX", binsizeY, lower_limitY, upper_limitY, binsizeX, lower_limitX, upper_limitX));
      p2YZ.reset(
         new TProfile2D("p2YZ", "p2YZ", binsizeY, lower_limitY, upper_limitY, binsizeZ, lower_limitZ, upper_limitZ));
      p2ZX.reset(
         new TProfile2D("p2ZX", "p2ZX", binsizeZ, lower_limitZ, upper_limitZ, binsizeX, lower_limitX, upper_limitX));
      p2ZY.reset(
         new TProfile2D("p2ZY", "p2ZY", binsizeZ, lower_limitZ, upper_limitZ, binsizeY, lower_limitY, upper_limitY));

      p1X.reset(new TProfile("p1X", "pe1X", binsizeX, lower_limitX, upper_limitX));
      p1Y.reset(new TProfile("p1Y", "pe1Y", binsizeY, lower_limitY, upper_limitY));
      p1Z.reset(new TProfile("p1Z", "pe1Z", binsizeZ, lower_limitZ, upper_limitZ));
   }

public:
   ProjectionTester()
   {
      CreateProfiles();
      CreateHistograms();
   }

   void BuildWithWeights() { buildWithWeights = true; }

   void buildHistograms()
   {
      if (h3->GetSumw2N()) {
         s3->Sumw2();
         n3->Sumw2();
      }

      for (int ix = 0; ix <= h3->GetXaxis()->GetNbins() + 1; ++ix) {
         double xc = h3->GetXaxis()->GetBinCenter(ix);
         double x = xc + centre_deviation * h3->GetXaxis()->GetBinWidth(ix);
         for (int iy = 0; iy <= h3->GetYaxis()->GetNbins() + 1; ++iy) {
            double yc = h3->GetYaxis()->GetBinCenter(iy);
            double y = yc + centre_deviation * h3->GetYaxis()->GetBinWidth(iy);
            for (int iz = 0; iz <= h3->GetZaxis()->GetNbins() + 1; ++iz) {
               double zc = h3->GetZaxis()->GetBinCenter(iz);
               double z = zc + centre_deviation * h3->GetZaxis()->GetBinWidth(iz);
               for (int i = 0; i < (int)r.Uniform(1, 3); ++i) {
                  h3->Fill(x, y, z);

                  Double_t points[] = {x, y, z};
                  s3->Fill(points);
                  n3->Fill(points);

                  h2XY->Fill(x, y);
                  h2XZ->Fill(x, z);
                  h2YX->Fill(y, x);
                  h2YZ->Fill(y, z);
                  h2ZX->Fill(z, x);
                  h2ZY->Fill(z, y);

                  h1X->Fill(x);
                  h1Y->Fill(y);
                  h1Z->Fill(z);

                  if (ix > 0 && ix < h3->GetXaxis()->GetNbins() + 1 && iy > 0 && iy < h3->GetYaxis()->GetNbins() + 1 &&
                      iz > 0 && iz < h3->GetZaxis()->GetNbins() + 1) {
                     h1XStats->Fill(x);
                     h1YStats->Fill(y);
                     h1ZStats->Fill(z);
                  }

                  // for filling reference profile need to use bin center
                  // because projection from histogram can use only bin center
                  pe2XY->Fill(xc, yc, zc);
                  pe2XZ->Fill(xc, zc, yc);
                  pe2YX->Fill(yc, xc, zc);
                  pe2YZ->Fill(yc, zc, xc);
                  pe2ZX->Fill(zc, xc, yc);
                  pe2ZY->Fill(zc, yc, xc);

                  // reference histogram to test with option W.
                  // need to use bin center for the weight
                  h2wXY->Fill(x, y, zc);
                  h2wXZ->Fill(x, z, yc);
                  h2wYX->Fill(y, x, zc);
                  h2wYZ->Fill(y, z, xc);
                  h2wZX->Fill(z, x, yc);
                  h2wZY->Fill(z, y, xc);

                  pe1XY->Fill(xc, yc);
                  pe1XZ->Fill(xc, zc);
                  pe1YX->Fill(yc, xc);
                  pe1YZ->Fill(yc, zc);
                  pe1ZX->Fill(zc, xc);
                  pe1ZY->Fill(zc, yc);

                  hw1XY->Fill(x, yc);
                  hw1XZ->Fill(x, zc);
                  hw1YX->Fill(y, xc);
                  hw1YZ->Fill(y, zc);
                  hw1ZX->Fill(z, xc);
                  hw1ZY->Fill(z, yc);
               }
            }
         }
      }

      buildWithWeights = false;
   }

   void buildHistogramsWithWeights()
   {

      s3->Sumw2();
      n3->Sumw2();

      for (int ix = 0; ix <= h3->GetXaxis()->GetNbins() + 1; ++ix) {
         double xc = h3->GetXaxis()->GetBinCenter(ix);
         double x = xc + centre_deviation * h3->GetXaxis()->GetBinWidth(ix);
         for (int iy = 0; iy <= h3->GetYaxis()->GetNbins() + 1; ++iy) {
            double yc = h3->GetYaxis()->GetBinCenter(iy);
            double y = yc + centre_deviation * h3->GetYaxis()->GetBinWidth(iy);
            for (int iz = 0; iz <= h3->GetZaxis()->GetNbins() + 1; ++iz) {
               double zc = h3->GetZaxis()->GetBinCenter(iz);
               double z = zc + centre_deviation * h3->GetZaxis()->GetBinWidth(iz);

               Double_t w = (Double_t)r.Uniform(1, 3);

               h3->Fill(x, y, z, w);

               Double_t points[] = {x, y, z};
               s3->Fill(points, w);
               n3->Fill(points, w);

               h2XY->Fill(x, y, w);
               h2XZ->Fill(x, z, w);
               h2YX->Fill(y, x, w);
               h2YZ->Fill(y, z, w);
               h2ZX->Fill(z, x, w);
               h2ZY->Fill(z, y, w);

               h1X->Fill(x, w);
               h1Y->Fill(y, w);
               h1Z->Fill(z, w);

               if (ix > 0 && ix < h3->GetXaxis()->GetNbins() + 1 && iy > 0 && iy < h3->GetYaxis()->GetNbins() + 1 &&
                   iz > 0 && iz < h3->GetZaxis()->GetNbins() + 1) {
                  h1XStats->Fill(x, w);
                  h1YStats->Fill(y, w);
                  h1ZStats->Fill(z, w);
               }

               pe2XY->Fill(xc, yc, zc, w);
               pe2XZ->Fill(xc, zc, yc, w);
               pe2YX->Fill(yc, xc, zc, w);
               pe2YZ->Fill(yc, zc, xc, w);
               pe2ZX->Fill(zc, xc, yc, w);
               pe2ZY->Fill(zc, yc, xc, w);

               h2wXY->Fill(x, y, zc * w);
               h2wXZ->Fill(x, z, yc * w);
               h2wYX->Fill(y, x, zc * w);
               h2wYZ->Fill(y, z, xc * w);
               h2wZX->Fill(z, x, yc * w);
               h2wZY->Fill(z, y, xc * w);

               pe1XY->Fill(xc, yc, w);
               pe1XZ->Fill(xc, zc, w);
               pe1YX->Fill(yc, xc, w);
               pe1YZ->Fill(yc, zc, w);
               pe1ZX->Fill(zc, xc, w);
               pe1ZY->Fill(zc, yc, w);

               hw1XY->Fill(x, yc * w);
               hw1XZ->Fill(x, zc * w);
               hw1YX->Fill(y, xc * w);
               hw1YZ->Fill(y, zc * w);
               hw1ZX->Fill(z, xc * w);
               hw1ZY->Fill(z, yc * w);
            }
         }
      }

      buildWithWeights = true;
   }

   void compareHistograms()
   {
      int options = 0;
      unique_ptr<TH2D> h2Projection;
      unique_ptr<TProfile2D> peProjection;

      // TH2 derived from TH3
      options = cmpOptStats;
      h2Projection.reset((TH2D *)h3->Project3D("yx"));
      EXPECT_EQ(0, Equals("TH3 -> XY", h2XY, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3D("zx"));
      EXPECT_EQ(0, Equals("TH3 -> XZ", h2XZ, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3D("XY"));
      EXPECT_EQ(0, Equals("TH3 -> YX", h2YX, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3D("ZY"));
      EXPECT_EQ(0, Equals("TH3 -> YZ", h2YZ, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3D("XZ"));
      EXPECT_EQ(0, Equals("TH3 -> ZX", h2ZX, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3D("YZ"));
      EXPECT_EQ(0, Equals("TH3 -> ZY", h2ZY, h2Projection, options));
      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;

      // TH1 derived from TH3
      options = cmpOptStats;
      unique_ptr<TH1D> tmp1;
      unique_ptr<TH1D> projectionTH1D;

      projectionTH1D.reset((TH1D *)h3->Project3D("x"));
      EXPECT_EQ(0, Equals("TH3 -> X", h1X, projectionTH1D, options));
      tmp1.reset(h3->ProjectionX("x335"));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset((TH1D *)h3->Project3D("x2"));
      EXPECT_EQ(0, Equals("TH3 -> X(x2)", tmp1, projectionTH1D, options));
      tmp1.reset(0);
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset((TH1D *)h3->Project3D("y"));
      EXPECT_EQ(0, Equals("TH3 -> Y", h1Y, projectionTH1D, options));
      tmp1.reset(h3->ProjectionY("y335"));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset((TH1D *)h3->Project3D("y2"));
      EXPECT_EQ(0, Equals("TH3 -> Y(x2)", tmp1, projectionTH1D, options));
      tmp1.reset(0);
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset((TH1D *)h3->Project3D("z"));
      EXPECT_EQ(0, Equals("TH3 -> Z", h1Z, projectionTH1D, options));
      tmp1.reset(h3->ProjectionZ("z335"));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset((TH1D *)h3->Project3D("z2"));
      EXPECT_EQ(0, Equals("TH3 -> Z(x2)", tmp1, projectionTH1D, options));
      tmp1.reset(0);

      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;

      // TH1 derived from h2XY
      options = cmpOptStats;
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XY->ProjectionX("x"));
      EXPECT_EQ(0, Equals("TH2XY -> X", h1X, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XY->ProjectionY("y"));
      EXPECT_EQ(0, Equals("TH2XY -> Y", h1Y, projectionTH1D, options));
      // TH1 derived from h2XZ
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XZ->ProjectionX("x"));
      EXPECT_EQ(0, Equals("TH2XZ -> X", h1X, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XZ->ProjectionY("z"));
      EXPECT_EQ(0, Equals("TH2XZ -> Z", h1Z, projectionTH1D, options));
      // TH1 derived from h2YX
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YX->ProjectionX("y"));
      EXPECT_EQ(0, Equals("TH2YX -> Y", h1Y, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YX->ProjectionY("x"));
      EXPECT_EQ(0, Equals("TH2YX -> X", h1X, projectionTH1D, options));
      // TH1 derived from h2YZ
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YZ->ProjectionX("y"));
      EXPECT_EQ(0, Equals("TH2YZ -> Y", h1Y, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YZ->ProjectionY("z"));
      EXPECT_EQ(0, Equals("TH2YZ -> Z", h1Z, projectionTH1D, options));
      // TH1 derived from h2ZX
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZX->ProjectionX("z"));
      EXPECT_EQ(0, Equals("TH2ZX -> Z", h1Z, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZX->ProjectionY("x"));
      EXPECT_EQ(0, Equals("TH2ZX -> X", h1X, projectionTH1D, options));
      // TH1 derived from h2ZY
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZY->ProjectionX("z"));
      EXPECT_EQ(0, Equals("TH2ZY -> Z", h1Z, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZY->ProjectionY("y"));
      EXPECT_EQ(0, Equals("TH2ZY -> Y", h1Y, projectionTH1D, options));
      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;

      // in the following comparison with profiles we need to re-calculate statistics using bin centers
      // on the reference histograms
      if (centre_deviation != 0) {
         h2XY->ResetStats();
         h2YX->ResetStats();
         h2XZ->ResetStats();
         h2ZX->ResetStats();
         h2YZ->ResetStats();
         h2ZY->ResetStats();

         h1X->ResetStats();
         h1Y->ResetStats();
         h1Z->ResetStats();
      }

      // Now the histograms coming from the Profiles!
      options = cmpOptStats;
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3DProfile("yx UF OF")->ProjectionXY("1", "B"));
      EXPECT_EQ(0, Equals("TH3 -> PBXY", h2XY, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3DProfile("zx UF OF")->ProjectionXY("2", "B"));
      EXPECT_EQ(0, Equals("TH3 -> PBXZ", h2XZ, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3DProfile("xy UF OF")->ProjectionXY("3", "B"));
      EXPECT_EQ(0, Equals("TH3 -> PBYX", h2YX, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3DProfile("zy UF OF")->ProjectionXY("4", "B"));
      EXPECT_EQ(0, Equals("TH3 -> PBYZ", h2YZ, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3DProfile("xz UF OF")->ProjectionXY("5", "B"));
      EXPECT_EQ(0, Equals("TH3 -> PBZX", h2ZX, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3DProfile("yz UF OF")->ProjectionXY("6", "B"));
      EXPECT_EQ(0, Equals("TH3 -> PBZY", h2ZY, h2Projection, options));
      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;

      // test directly project3dprofile
      options = cmpOptStats;
      peProjection.reset(h3->Project3DProfile("yx  UF OF"));
      EXPECT_EQ(0, Equals("TH3 -> PXY", pe2XY, peProjection, options));
      peProjection.reset(nullptr);
      peProjection.reset(h3->Project3DProfile("zx  UF OF"));
      EXPECT_EQ(0, Equals("TH3 -> PXZ", pe2XZ, peProjection, options));
      peProjection.reset(nullptr);
      peProjection.reset(h3->Project3DProfile("xy  UF OF"));
      EXPECT_EQ(0, Equals("TH3 -> PYX", pe2YX, peProjection, options));
      peProjection.reset(nullptr);
      peProjection.reset(h3->Project3DProfile("zy  UF OF"));
      EXPECT_EQ(0, Equals("TH3 -> PYZ", pe2YZ, peProjection, options));
      peProjection.reset(nullptr);
      peProjection.reset(h3->Project3DProfile("xz  UF OF"));
      EXPECT_EQ(0, Equals("TH3 -> PZX", pe2ZX, peProjection, options));
      peProjection.reset(nullptr);
      peProjection.reset(h3->Project3DProfile("yz  UF OF"));
      EXPECT_EQ(0, Equals("TH3 -> PZY", pe2ZY, peProjection, options));
      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;

      // test option E of ProjectionXY
      options = 0;
      h2Projection.reset(nullptr);
      h2Projection.reset(h3->Project3DProfile("yx  UF OF")->ProjectionXY("1", "E"));
      EXPECT_EQ(0, Equals("TH3 -> PEXY", pe2XY, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset(h3->Project3DProfile("zx  UF OF")->ProjectionXY("2", "E"));
      EXPECT_EQ(0, Equals("TH3 -> PEXZ", pe2XZ, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset(h3->Project3DProfile("xy  UF OF")->ProjectionXY("3", "E"));
      EXPECT_EQ(0, Equals("TH3 -> PEYX", pe2YX, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset(h3->Project3DProfile("zy  UF OF")->ProjectionXY("4", "E"));
      EXPECT_EQ(0, Equals("TH3 -> PEYZ", pe2YZ, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset(h3->Project3DProfile("xz  UF OF")->ProjectionXY("5", "E"));
      EXPECT_EQ(0, Equals("TH3 -> PEZX", pe2ZX, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset(h3->Project3DProfile("yz  UF OF")->ProjectionXY("6", "E"));
      EXPECT_EQ(0, Equals("TH3 -> PEZY", pe2ZY, h2Projection, options));
      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;

      // test option W of ProjectionXY

      // The error fails when built with weights. It is not properly calculated
      if (buildWithWeights) options = cmpOptNoError;
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3DProfile("yx  UF OF")->ProjectionXY("1", "W"));
      EXPECT_EQ(0, Equals("TH3 -> PWXY", h2wXY, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3DProfile("zx  UF OF")->ProjectionXY("2", "W"));
      EXPECT_EQ(0, Equals("TH3 -> PWXZ", h2wXZ, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3DProfile("xy  UF OF")->ProjectionXY("3", "W"));
      EXPECT_EQ(0, Equals("TH3 -> PWYX", h2wYX, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3DProfile("zy  UF OF")->ProjectionXY("4", "W"));
      EXPECT_EQ(0, Equals("TH3 -> PWYZ", h2wYZ, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3DProfile("xz  UF OF")->ProjectionXY("5", "W"));
      EXPECT_EQ(0, Equals("TH3 -> PWZX", h2wZX, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)h3->Project3DProfile("yz  UF OF")->ProjectionXY("6", "W"));
      EXPECT_EQ(0, Equals("TH3 -> PWZY", h2wZY, h2Projection, options));
      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;

      // test 1D histograms
      options = cmpOptStats;
      // ProfileX re-use the same histo if sme name is given.
      // need to give a diffrent name for each projectino (x,y,Z) otherwise we end-up in different bins
      // t.b.d: ProfileX make a new histo if non compatible
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XY->ProfileX("PBX", 0, h2XY->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "B"));
      EXPECT_EQ(0, Equals("TH2XY -> PBX", h1X, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XY->ProfileY("PBY", 0, h2XY->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "B"));
      EXPECT_EQ(0, Equals("TH2XY -> PBY", h1Y, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XZ->ProfileX("PBX", 0, h2XZ->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "B"));
      EXPECT_EQ(0, Equals("TH2XZ -> PBX", h1X, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XZ->ProfileY("PBZ", 0, h2XZ->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "B"));
      EXPECT_EQ(0, Equals("TH2XZ -> PBZ", h1Z, projectionTH1D, options, 1E-12));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YX->ProfileX("PBY", 0, h2YX->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "B"));
      EXPECT_EQ(0, Equals("TH2YX -> PBY", h1Y, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YX->ProfileY("PBX", 0, h2YX->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "B"));
      EXPECT_EQ(0, Equals("TH2YX -> PBX", h1X, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YZ->ProfileX("PBY", 0, h2YZ->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "B"));
      EXPECT_EQ(0, Equals("TH2YZ -> PBY", h1Y, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YZ->ProfileY("PBZ", 0, h2YZ->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "B"));
      EXPECT_EQ(0, Equals("TH2YZ -> PBZ", h1Z, projectionTH1D, options, 1E-12));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZX->ProfileX("PBZ", 0, h2ZX->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "B"));
      EXPECT_EQ(0, Equals("TH2ZX -> PBZ", h1Z, projectionTH1D, options, 1E-12));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZX->ProfileY("PBX", 0, h2ZX->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "B"));
      EXPECT_EQ(0, Equals("TH2ZX -> PBX", h1X, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZY->ProfileX("PBZ", 0, h2ZY->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "B"));
      EXPECT_EQ(0, Equals("TH2ZY -> PBZ", h1Z, projectionTH1D, options, 1E-12));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZY->ProfileY("PBY", 0, h2ZY->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "B"));
      EXPECT_EQ(0, Equals("TH2ZY -> PBY", h1Y, projectionTH1D, options));
      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;

      // 1D testing direct profiles
      options = cmpOptStats;
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XY->ProfileX("PX", 0, h2XY->GetYaxis()->GetNbins() + 1));
      EXPECT_EQ(0, Equals("TH2XY -> PX", pe1XY, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XY->ProfileY("PY", 0, h2XY->GetXaxis()->GetNbins() + 1));
      EXPECT_EQ(0, Equals("TH2XY -> PY", pe1YX, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XZ->ProfileX("PX", 0, h2XZ->GetYaxis()->GetNbins() + 1));
      EXPECT_EQ(0, Equals("TH2XZ -> PX", pe1XZ, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XZ->ProfileY("PZ", 0, h2XZ->GetXaxis()->GetNbins() + 1));
      EXPECT_EQ(0, Equals("TH2XZ -> PZ", pe1ZX, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YX->ProfileX("PY", 0, h2YX->GetYaxis()->GetNbins() + 1));
      EXPECT_EQ(0, Equals("TH2YX -> PY", pe1YX, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YX->ProfileY("PX", 0, h2YX->GetXaxis()->GetNbins() + 1));
      EXPECT_EQ(0, Equals("TH2YX -> PX", pe1XY, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YZ->ProfileX("PY", 0, h2YZ->GetYaxis()->GetNbins() + 1));
      EXPECT_EQ(0, Equals("TH2YZ -> PY", pe1YZ, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YZ->ProfileY("PZ", 0, h2YZ->GetXaxis()->GetNbins() + 1));
      EXPECT_EQ(0, Equals("TH2YZ -> PZ", pe1ZY, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZX->ProfileX("PZ", 0, h2ZX->GetYaxis()->GetNbins() + 1));
      EXPECT_EQ(0, Equals("TH2ZX -> PZ", pe1ZX, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZX->ProfileY("PX", 0, h2ZX->GetXaxis()->GetNbins() + 1));
      EXPECT_EQ(0, Equals("TH2ZX -> PX", pe1XZ, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZY->ProfileX("PZ", 0, h2ZY->GetYaxis()->GetNbins() + 1));
      EXPECT_EQ(0, Equals("TH2ZY -> PZ", pe1ZY, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZY->ProfileY("PY", 0, h2ZY->GetXaxis()->GetNbins() + 1));
      EXPECT_EQ(0, Equals("TH2ZY -> PY", pe1YZ, projectionTH1D, options));
      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;

      // 1D testing e profiles
      options = 0;
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XY->ProfileX("PEX", 0, h2XY->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "E"));
      EXPECT_EQ(0, Equals("TH2XY -> PEX", pe1XY, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XY->ProfileY("PEY", 0, h2XY->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "E"));
      EXPECT_EQ(0, Equals("TH2XY -> PEY", pe1YX, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XZ->ProfileX("PEX", 0, h2XZ->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "E"));
      EXPECT_EQ(0, Equals("TH2XZ -> PEX", pe1XZ, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XZ->ProfileY("PEZ", 0, h2XZ->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "E"));
      EXPECT_EQ(0, Equals("TH2XZ -> PEZ", pe1ZX, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YX->ProfileX("PEY", 0, h2YX->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "E"));
      EXPECT_EQ(0, Equals("TH2YX -> PEY", pe1YX, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YX->ProfileY("PEX", 0, h2YX->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "E"));
      EXPECT_EQ(0, Equals("TH2YX -> PEX", pe1XY, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YZ->ProfileX("PEY", 0, h2YZ->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "E"));
      EXPECT_EQ(0, Equals("TH2YZ -> PEY", pe1YZ, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YZ->ProfileY("PEZ", 0, h2YZ->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "E"));
      EXPECT_EQ(0, Equals("TH2YZ -> PEZ", pe1ZY, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZX->ProfileX("PEZ", 0, h2ZX->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "E"));
      EXPECT_EQ(0, Equals("TH2ZX -> PEZ", pe1ZX, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZX->ProfileY("PEX", 0, h2ZX->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "E"));
      EXPECT_EQ(0, Equals("TH2ZX -> PEX", pe1XZ, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZY->ProfileX("PEZ", 0, h2ZY->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "E"));
      EXPECT_EQ(0, Equals("TH2ZY -> PEZ", pe1ZY, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZY->ProfileY("PEY", 0, h2ZY->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "E"));
      EXPECT_EQ(0, Equals("TH2ZY -> PEY", pe1YZ, projectionTH1D, options));
      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;

      // 1D testing w profiles
      // The error is not properly propagated when build with weights :S
      if (buildWithWeights) options = cmpOptNoError;
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XY->ProfileX("PWX", 0, h2XY->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "W"));
      EXPECT_EQ(0, Equals("TH2XY -> PWX", hw1XY, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XY->ProfileY("PWY", 0, h2XY->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "W"));
      EXPECT_EQ(0, Equals("TH2XY -> PWY", hw1YX, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XZ->ProfileX("PWX", 0, h2XZ->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "W"));
      EXPECT_EQ(0, Equals("TH2XZ -> PWX", hw1XZ, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2XZ->ProfileY("PWZ", 0, h2XZ->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "W"));
      EXPECT_EQ(0, Equals("TH2XZ -> PWZ", hw1ZX, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YX->ProfileX("PWY", 0, h2YX->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "W"));
      EXPECT_EQ(0, Equals("TH2YX -> PWY", hw1YX, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YX->ProfileY("PWX", 0, h2YX->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "W"));
      EXPECT_EQ(0, Equals("TH2YX -> PWX", hw1XY, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YZ->ProfileX("PWY", 0, h2YZ->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "W"));
      EXPECT_EQ(0, Equals("TH2YZ -> PWY", hw1YZ, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2YZ->ProfileY("PWZ", 0, h2YZ->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "W"));
      EXPECT_EQ(0, Equals("TH2YZ -> PWZ", hw1ZY, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZX->ProfileX("PWZ", 0, h2ZX->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "W"));
      EXPECT_EQ(0, Equals("TH2ZX -> PWZ", hw1ZX, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZX->ProfileY("PWX", 0, h2ZX->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "W"));
      EXPECT_EQ(0, Equals("TH2ZX -> PWX", hw1XZ, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZY->ProfileX("PWZ", 0, h2ZY->GetYaxis()->GetNbins() + 1)->ProjectionX("1", "W"));
      EXPECT_EQ(0, Equals("TH2ZY -> PWZ", hw1ZY, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(h2ZY->ProfileY("PWY", 0, h2ZY->GetXaxis()->GetNbins() + 1)->ProjectionX("1", "W"));
      EXPECT_EQ(0, Equals("TH2ZY -> PWY", hw1YZ, projectionTH1D, options));

      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;

      // do THNsparse after Profile because reference histograms need to have a ResetStats
      // the statistics coming from a projected THNsparse has been computed using the bin centers

      // TH2 derived from STH3
      options = cmpOptStats;
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)s3->Projection(1, 0));
      EXPECT_EQ(0, Equals("STH3 -> XY", h2XY, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)s3->Projection(2, 0));
      EXPECT_EQ(0, Equals("STH3 -> XZ", h2XZ, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)s3->Projection(0, 1));
      EXPECT_EQ(0, Equals("STH3 -> YX", h2YX, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)s3->Projection(2, 1));
      EXPECT_EQ(0, Equals("STH3 -> YZ", h2YZ, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)s3->Projection(0, 2));
      EXPECT_EQ(0, Equals("STH3 -> ZX", h2ZX, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)s3->Projection(1, 2));
      EXPECT_EQ(0, Equals("STH3 -> ZY", h2ZY, h2Projection, options));
      h2Projection.reset(nullptr);

      h2Projection.reset((TH2D *)n3->Projection(1, 0));
      EXPECT_EQ(0, Equals("THn3 -> XY", h2XY, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)n3->Projection(2, 0));
      EXPECT_EQ(0, Equals("THn3 -> XZ", h2XZ, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)n3->Projection(0, 1));
      EXPECT_EQ(0, Equals("THn3 -> YX", h2YX, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)n3->Projection(2, 1));
      EXPECT_EQ(0, Equals("THn3 -> YZ", h2YZ, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)n3->Projection(0, 2));
      EXPECT_EQ(0, Equals("THn3 -> ZX", h2ZX, h2Projection, options));
      h2Projection.reset(nullptr);
      h2Projection.reset((TH2D *)n3->Projection(1, 2));
      EXPECT_EQ(0, Equals("THn3 -> ZY", h2ZY, h2Projection, options));
      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;

      // TH1 derived from STH3
      options = cmpOptStats;
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(s3->Projection(0));
      EXPECT_EQ(0, Equals("STH3 -> X", h1X, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(s3->Projection(1));
      EXPECT_EQ(0, Equals("STH3 -> Y", h1Y, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(s3->Projection(2));
      EXPECT_EQ(0, Equals("STH3 -> Z", h1Z, projectionTH1D, options));

      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(n3->Projection(0));
      EXPECT_EQ(0, Equals("THn3 -> X", h1X, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(n3->Projection(1));
      EXPECT_EQ(0, Equals("THn3 -> Y", h1Y, projectionTH1D, options));
      projectionTH1D.reset(nullptr);
      projectionTH1D.reset(n3->Projection(2));
      EXPECT_EQ(0, Equals("THn3 -> Z", h1Z, projectionTH1D, options));
      options = 0;
      if (defaultEqualOptions & cmpOptPrint) std::cout << "----------------------------------------------" << std::endl;
   }

   void buildProfiles()
   {

      if (buildWithWeights) {
         p3->Sumw2();
         p2XY->Sumw2();
         p2YX->Sumw2();
         p2YZ->Sumw2();
         p2XZ->Sumw2();
         p2ZX->Sumw2();
         p2ZY->Sumw2();
         p1X->Sumw2();
         p1Y->Sumw2();
         p1Z->Sumw2();
      }

      // use a different way to fill the histogram
      for (int i = 0; i < 100000; ++i) {

         // use in range in X but only overflow in Y and underflow/overflow in Z
         double x = gRandom->Uniform(lower_limitX, upper_limitX);
         double y = gRandom->Uniform(lower_limitY, upper_limitY + 2.);
         double z = gRandom->Uniform(lower_limitZ - 1, upper_limitZ + 1);
         double u = TMath::Gaus(x, 0, 3) * TMath::Gaus(y, 3, 5) * TMath::Gaus(z, -3, 10);

         double w = 1;
         if (buildWithWeights) w += x * x + (y - 2) * (y - 2) + (z + 2) * (z + 2);

         p3->Fill(x, y, z, u, w);

         p2XY->Fill(x, y, u, w);
         p2YX->Fill(y, x, u, w);
         p2XZ->Fill(x, z, u, w);
         p2ZX->Fill(z, x, u, w);
         p2YZ->Fill(y, z, u, w);
         p2ZY->Fill(z, y, u, w);

         p1X->Fill(x, u, w);
         p1Y->Fill(y, u, w);
         p1Z->Fill(z, u, w);
      }

      // reset the statistics to get same statistics computed from bin centers
      p1X->ResetStats();
      p1Y->ResetStats();
      p1Z->ResetStats();

      p2XY->ResetStats();
      p2YX->ResetStats();
      p2XZ->ResetStats();
      p2ZX->ResetStats();
      p2YZ->ResetStats();
      p2ZY->ResetStats();
   }

   // actual test of profile projections
   void compareProfiles()
   {
      int options = 0;

      // TProfile2d derived from TProfile3d
      options = cmpOptStats;
      // options = cmpOptPrint;
      unique_ptr<TProfile2D> profile2D;
      profile2D.reset(p3->Project3DProfile("yx"));
      EXPECT_EQ(0, Equals("TProfile3D -> XY", p2XY, profile2D, options));
      profile2D.reset(nullptr);
      profile2D.reset(p3->Project3DProfile("xy"));
      EXPECT_EQ(0, Equals("TProfile3D -> YX", p2YX, profile2D, options));
      profile2D.reset(nullptr);
      profile2D.reset(p3->Project3DProfile("zx"));
      EXPECT_EQ(0, Equals("TProfile3D -> XZ", p2XZ, profile2D, options));
      profile2D.reset(nullptr);
      profile2D.reset(p3->Project3DProfile("xz"));
      EXPECT_EQ(0, Equals("TProfile3D -> ZX", p2ZX, profile2D, options));
      profile2D.reset(nullptr);
      profile2D.reset(p3->Project3DProfile("zy"));
      EXPECT_EQ(0, Equals("TProfile3D -> YZ", p2YZ, profile2D, options));
      profile2D.reset(nullptr);
      profile2D.reset(p3->Project3DProfile("yz"));
      EXPECT_EQ(0, Equals("TProfile3D -> ZY", p2ZY, profile2D, options));
      options = 0;
      if (defaultEqualOptions & cmpOptPrint) cout << "----------------------------------------------" << endl;

      // TProfile1 derived from TProfile2D from TProfile3D
      options = cmpOptStats;
      // options = cmpOptDebug;
      unique_ptr<TProfile2D> tmp1;
      unique_ptr<TProfile> profile;
      profile.reset(p2XY->ProfileX());
      EXPECT_EQ(0, Equals("TProfile2D -> X", p1X, profile, options));
      tmp1.reset(nullptr);
      tmp1.reset(p3->Project3DProfile("xz"));
      profile.reset(tmp1->ProfileY());
      EXPECT_EQ(0, Equals("TProfile3D -> X", p1X, profile, options));
      profile.reset(p2ZY->ProfileY());
      EXPECT_EQ(0, Equals("TProfile2D -> Y", p1Y, profile, options));
      tmp1.reset(nullptr);
      tmp1.reset(p3->Project3DProfile("xy"));
      profile.reset(tmp1->ProfileX());
      EXPECT_EQ(0, Equals("TProfile3D -> X", p1Y, profile, options));
      profile.reset(p2ZX->ProfileX());
      EXPECT_EQ(0, Equals("TProfile2D -> Z", p1Z, profile, options));
      tmp1.reset(nullptr);
      tmp1.reset(p3->Project3DProfile("zy"));
      profile.reset(tmp1->ProfileY());
      EXPECT_EQ(0, Equals("TProfile3D -> Z", p1Z, profile, options));
   }

protected:
   virtual void SetUp()
   {
      r.SetSeed();

      TProfile::Approximate();
      TProfile2D::Approximate();
      TProfile3D::Approximate();
   }
};

TEST_F(ProjectionTester, HistogramProjectionsWithoutWeights)
{
   buildHistograms();
   compareHistograms();
}

TEST_F(ProjectionTester, ProfileProjectionsWithoutWeights)
{
   buildProfiles();
   compareProfiles();
}

TEST_F(ProjectionTester, HistogramProjectionsWithWeights)
{
   TH1::SetDefaultSumw2();
   buildHistogramsWithWeights();
   compareHistograms();
}

TEST_F(ProjectionTester, ProfileProjectionsWithWeights)
{
   TH1::SetDefaultSumw2();
   BuildWithWeights();
   buildProfiles();
   compareProfiles();
}
