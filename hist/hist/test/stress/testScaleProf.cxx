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

TEST(StressHistogram, TestScale1DProf)
{
   TRandom2 r;
   TProfile p1("scD1-p1", "p1-Title", numberOfBins, minRange, maxRange);
   TProfile p2("scD1-p2", "p2=c1*p1", numberOfBins, minRange, maxRange);

   Double_t c1 = r.Rndm();

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, 1.0);
      p2.Fill(x, c1 * y, 1.0);
   }

   p1.Scale(c1);

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
}

TEST(StressHistogram, TestScale2DProf)
{
   TRandom2 r;
   TProfile2D p1("scD2-p1", "p1", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TProfile2D p2("scD2-p2", "p2=c1*p1", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   Double_t c1 = r.Rndm();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, z, 1.0);
      p2.Fill(x, y, c1 * z, 1.0);
   }

   p1.Scale(c1);

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
}

TEST(StressHistogram, TestScale3DProf)
{
   TRandom2 r;
   TProfile3D p1("scD3-p1", "p1", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                 numberOfBins + 2, minRange, maxRange);
   TProfile3D p2("scD3-p2", "p2=c1*p1", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                 numberOfBins + 2, minRange, maxRange);
   Double_t c1 = r.Rndm();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, z, t, 1.0);
      p2.Fill(x, y, z, c1 * t, 1.0);
   }

   p1.Scale(c1);

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
}
