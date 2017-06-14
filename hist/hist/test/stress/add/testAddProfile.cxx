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

#include "../StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestAddProfile1)
{
   TH1::SetDefaultSumw2();

   // Tests the first Add method for 1D Profiles

   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TProfile p1("t1D1-p1", "p1-Title", numberOfBins, minRange, maxRange);
   TProfile p2("t1D1-p2", "p2-Title", numberOfBins, minRange, maxRange);
   TProfile p3("t1D1-p3", "p3=c1*p1+c2*p2", numberOfBins, minRange, maxRange);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, 1.0);
      p3.Fill(x, y, c1);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p2.Fill(x, y, 1.0);
      p3.Fill(x, y, c2);
   }

   TProfile p4("t1D1-p4", "p4=c1*p1+p2*c2", numberOfBins, minRange, maxRange);
   p4.Add(&p1, &p2, c1, c2);

   EXPECT_TRUE(HistogramsEquals(p3, p4, cmpOptStats, 1E-13));
}

TEST(StressHistorgram, TestAddProfile2)
{
   TH1::SetDefaultSumw2();

   // Tests the second Add method for 1D Profiles

   Double_t c2 = r.Rndm();

   TProfile p5("t1D2-p5", "p5=   p6+c2*p7", numberOfBins, minRange, maxRange);
   TProfile p6("t1D2-p6", "p6-Title", numberOfBins, minRange, maxRange);
   TProfile p7("t1D2-p7", "p7-Title", numberOfBins, minRange, maxRange);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p6.Fill(x, y, 1.0);
      p5.Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p7.Fill(x, y, 1.0);
      p5.Fill(x, y, c2);
   }

   p6.Add(&p7, c2);

   EXPECT_TRUE(HistogramsEquals(p5, p6, cmpOptStats, 1E-13));
}
