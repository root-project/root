// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>
#include <cmath>

#include "TH2.h"
#include "THn.h"
#include "THnSparse.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "Riostream.h"
#include "TRandom2.h"
#include "TFile.h"
#include "TClass.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestInterpolation1D)
{
   // Tests interpolation method for 1D Histogram

   TH1D *h1 = new TH1D("h1", "h1", numberOfBins, minRange, maxRange);

   h1->Reset();

   for (Int_t nbinsx = 1; nbinsx <= h1->GetXaxis()->GetNbins(); ++nbinsx) {
      Double_t x = h1->GetXaxis()->GetBinCenter(nbinsx);
      h1->Fill(x, function1D(x));
   }

   for (int i = 0; i < 1000; i++) {
      double xp = r.Uniform(h1->GetXaxis()->GetBinCenter(1), h1->GetXaxis()->GetBinCenter(numberOfBins));
      double ip = h1->Interpolate(xp);

      EXPECT_FALSE(fabs(ip - function1D(xp)) > 1.E-13 * fabs(ip))
         << "x: " << xp << " h3->Inter: " << ip << " functionD: " << function1D(xp)
         << " diff: " << fabs(ip - function1D(xp)) << std::endl;
   }

   delete h1;
}

Double_t function2D(Double_t x, Double_t y)
{
   Double_t a = -2.1;
   Double_t b = 0.6;

   return a * x + b * y;
}

TEST(StressHistorgram, TestInterpolation2D)
{
   // Tests interpolation method for 2D Histogram

   TH2D *h1 = new TH2D("h1", "h1", numberOfBins, minRange, maxRange, 2 * numberOfBins, minRange, maxRange);

   h1->Reset();

   for (Int_t nbinsx = 1; nbinsx <= h1->GetXaxis()->GetNbins(); ++nbinsx)
      for (Int_t nbinsy = 1; nbinsy <= h1->GetYaxis()->GetNbins(); ++nbinsy) {
         Double_t x = h1->GetXaxis()->GetBinCenter(nbinsx);
         Double_t y = h1->GetYaxis()->GetBinCenter(nbinsy);
         h1->Fill(x, y, function2D(x, y));
      }

   for (int i = 0; i < 1000; i++) {
      double xp = r.Uniform(h1->GetXaxis()->GetBinCenter(1), h1->GetXaxis()->GetBinCenter(numberOfBins));
      double yp = r.Uniform(h1->GetYaxis()->GetBinCenter(1), h1->GetYaxis()->GetBinCenter(numberOfBins));

      double ip = h1->Interpolate(xp, yp);

      EXPECT_FALSE(fabs(ip - function2D(xp, yp)) > 1.E-13 * fabs(ip))
         << "x: " << xp << " y: " << yp << " h3->Inter: " << ip << " function: " << function2D(xp, yp)
         << " diff: " << fabs(ip - function2D(xp, yp)) << std::endl;
   }

   delete h1;
}

Double_t function3D(Double_t x, Double_t y, Double_t z)
{
   Double_t a = 0.3;
   Double_t b = 6;
   Double_t c = -2;

   return a * x + b * y + c * z;
}

TEST(StressHistorgram, TestInterpolation3D)
{
   // Tests interpolation method for 3D Histogram

   TH3D *h1 = new TH3D("h1", "h1", numberOfBins, minRange, maxRange, 2 * numberOfBins, minRange, maxRange,
                       4 * numberOfBins, minRange, maxRange);

   h1->Reset();

   for (Int_t nbinsx = 1; nbinsx <= h1->GetXaxis()->GetNbins(); ++nbinsx)
      for (Int_t nbinsy = 1; nbinsy <= h1->GetYaxis()->GetNbins(); ++nbinsy)
         for (Int_t nbinsz = 1; nbinsz <= h1->GetZaxis()->GetNbins(); ++nbinsz) {
            Double_t x = h1->GetXaxis()->GetBinCenter(nbinsx);
            Double_t y = h1->GetYaxis()->GetBinCenter(nbinsy);
            Double_t z = h1->GetZaxis()->GetBinCenter(nbinsz);
            h1->Fill(x, y, z, function3D(x, y, z));
         }

   for (int i = 0; i < 1000; i++) {
      double xp = r.Uniform(h1->GetXaxis()->GetBinCenter(1), h1->GetXaxis()->GetBinCenter(numberOfBins));
      double yp = r.Uniform(h1->GetYaxis()->GetBinCenter(1), h1->GetYaxis()->GetBinCenter(numberOfBins));
      double zp = r.Uniform(h1->GetZaxis()->GetBinCenter(1), h1->GetZaxis()->GetBinCenter(numberOfBins));

      double ip = h1->Interpolate(xp, yp, zp);

      EXPECT_FALSE(fabs(ip - function3D(xp, yp, zp)) > 1.E-15 * fabs(ip));
   }

   delete h1;
}

TEST(StressHistorgram, TestInterpolationVar1D)
{
   // Tests interpolation method for 1D Histogram with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   TH1D *h1 = new TH1D("h1", "h1", numberOfBins, v);

   h1->Reset();

   for (Int_t nbinsx = 1; nbinsx <= h1->GetXaxis()->GetNbins(); ++nbinsx) {
      Double_t x = h1->GetXaxis()->GetBinCenter(nbinsx);
      h1->Fill(x, function1D(x));
   }

   for (int i = 0; i < 1000; i++) {
      double xp = r.Uniform(h1->GetXaxis()->GetBinCenter(1), h1->GetXaxis()->GetBinCenter(numberOfBins));
      double ip = h1->Interpolate(xp);

      EXPECT_FALSE(fabs(ip - function1D(xp)) > 1.E-13 * fabs(ip))
         << "x: " << xp << " h3->Inter: " << ip << " functionD: " << function1D(xp)
         << " diff: " << fabs(ip - function1D(xp)) << std::endl;
   }

   delete h1;
}
