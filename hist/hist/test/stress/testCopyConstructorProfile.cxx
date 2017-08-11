// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include "StressHistogramGlobal.h"

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TRandom2.h"

#include <sstream>
#include "gtest/gtest.h"

using namespace std;

TEST(StressHistogram, TestCopyConstructorProfile1D)
{
   TRandom2 r;
   // Tests the copy constructor for 1D Profiles

   TProfile p1("cc1D-p1", "p1-Title", numberOfBins, minRange, maxRange);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, 1.0);
   }

   TProfile p2(p1);

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
}

TEST(StressHistogram, TestCopyConstructorProfile2D)
{
   TRandom2 r;
   // Tests the copy constructor for 2D Profiles

   TProfile2D p1("cc2D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, z, 1.0);
   }

   TProfile2D p2(p1);

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
}

TEST(StressHistogram, TestCopyConstructorProfile3D)
{
   TRandom2 r;
   // Tests the copy constructor for 3D Profiles

   TProfile3D p1("cc3D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                 numberOfBins + 2, minRange, maxRange);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, z, t, 1.0);
   }

   TProfile3D p2(p1);

   EXPECT_TRUE(HistogramsEquals(p1, p2));
}

TEST(StressHistogram, TestCopyConstructorProfileVar1D)
{
   TRandom2 r;
   // Tests the copy constructor for 1D Profiles with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   TProfile p1("cc1D-p1", "p1-Title", numberOfBins, v);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, 1.0);
   }

   TProfile p2(p1);

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
}
