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

TEST(StressHistorgram, TestScale1DProf)
{
   TProfile *p1 = new TProfile("scD1-p1", "p1-Title", numberOfBins, minRange, maxRange);
   TProfile *p2 = new TProfile("scD1-p2", "p2=c1*p1", numberOfBins, minRange, maxRange);

   Double_t c1 = r.Rndm();

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, 1.0);
      p2->Fill(x, c1 * y, 1.0);
   }

   p1->Scale(c1);

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
   delete p1;
   delete p2;
}

TEST(StressHistorgram, TestScale2DProf)
{
   TProfile2D *p1 =
      new TProfile2D("scD2-p1", "p1", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   TProfile2D *p2 =
      new TProfile2D("scD2-p2", "p2=c1*p1", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   Double_t c1 = r.Rndm();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, 1.0);
      p2->Fill(x, y, c1 * z, 1.0);
   }

   p1->Scale(c1);

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
   delete p1;
   delete p2;
}

TEST(StressHistorgram, TestScale3DProf)
{
   TProfile3D *p1 = new TProfile3D("scD3-p1", "p1", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange,
                                   maxRange, numberOfBins + 2, minRange, maxRange);

   TProfile3D *p2 = new TProfile3D("scD3-p2", "p2=c1*p1", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange,
                                   maxRange, numberOfBins + 2, minRange, maxRange);
   Double_t c1 = r.Rndm();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, t, 1.0);
      p2->Fill(x, y, z, c1 * t, 1.0);
   }

   p1->Scale(c1);

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
   delete p1;
   delete p2;
}
