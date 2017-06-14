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

TEST(StressHistorgram, TestAssignProfile1D)
{
   // Tests the operator=() method for 1D Profiles

   TProfile *p1 = new TProfile("=1D-p1", "p1-Title", numberOfBins, minRange, maxRange);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, 1.0);
   }

   TProfile *p2 = new TProfile("=1D-p2", "p2-Title", numberOfBins, minRange, maxRange);
   *p2 = *p1;

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
   delete p1;
   delete p2;
}

TEST(StressHistorgram, TestAssignProfile2D)
{
   // Tests the operator=() method for 2D Profiles

   TProfile2D *p1 =
      new TProfile2D("=2D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, 1.0);
   }

   TProfile2D *p2 =
      new TProfile2D("=2D-p2", "p2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   *p2 = *p1;

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
   delete p1;
   delete p2;
}

TEST(StressHistorgram, TestAssignProfile3D)
{
   // Tests the operator=() method for 3D Profiles

   TProfile3D *p1 = new TProfile3D("=3D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange,
                                   maxRange, numberOfBins + 2, minRange, maxRange);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, t, 1.0);
   }

   TProfile3D *p2 = new TProfile3D("=3D-p2", "p2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange,
                                   maxRange, numberOfBins + 2, minRange, maxRange);
   *p2 = *p1;

   EXPECT_TRUE(HistogramsEquals(p1, p2));
   delete p1;
   delete p2;
}

TEST(StressHistorgram, TestAssignProfileVar1D)
{
   // Tests the operator=() method for 1D Profiles with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   TProfile *p1 = new TProfile("=1D-p1", "p1-Title", numberOfBins, v);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, 1.0);
   }

   TProfile *p2 = new TProfile("=1D-p2", "p2-Title", numberOfBins, v);
   *p2 = *p1;

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
   delete p1;
   delete p2;
}
