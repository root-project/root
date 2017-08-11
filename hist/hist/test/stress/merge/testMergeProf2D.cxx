// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include "../StressHistogramGlobal.h"

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile2D.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TApplication.h"
#include "TFile.h"
#include "TRandom2.h"

#include <sstream>
#include "gtest/gtest.h"

using namespace std;

TEST(StressHistogram, TestMergeProf2D)
{
   TRandom2 r;
   // Tests the merge method for 2D Profiles

   TProfile2D p1("merge2D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TProfile2D p2("merge2D-p2", "p2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TProfile2D p3("merge2D-p3", "p3-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TProfile2D p4("merge2D-p4", "p4-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, z, 1.0);
      p4.Fill(x, y, z, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p2.Fill(x, y, z, 1.0);
      p4.Fill(x, y, z, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p3.Fill(x, y, z, 1.0);
      p4.Fill(x, y, z, 1.0);
   }

   TList list;
   list.Add(&p2);
   list.Add(&p3);

   p1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(p1, p4, cmpOptStats, 1E-10));
}

TEST(StressHistogram, TestMergeProf2DDiff)
{
   TRandom2 r;
   // Tests the merge method with different binned 2D Profile

   // This tests fails! It should not!
   TProfile2D p1("merge2DDiff-p1", "p1-Title", 11, -110, 0, 11, -110, 0);
   TProfile2D p2("merge2DDiff-p2", "p2-Title", 22, 0, 110, 22, 0, 110);
   TProfile2D p3("merge2DDiff-p3", "p3-Title", 44, -55, 55, 44, -55, 55);
   TProfile2D p4("merge2DDiff-p4", "p4-Title", 22, -110, 110, 22, -110, 110);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(-110, 0);
      Double_t y = r.Uniform(-110, 0);
      Double_t z = r.Gaus(5, 2);
      p1.Fill(x, y, z, 1.0);
      p4.Fill(x, y, z, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0, 110);
      Double_t y = r.Uniform(0, 110);
      Double_t z = r.Gaus(10, 3);
      p2.Fill(x, y, z, 1.0);
      p4.Fill(x, y, z, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(-55, 55);
      Double_t y = r.Uniform(-55, 55);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p3.Fill(x, y, z, 1.0);
      p4.Fill(x, y, z, 1.0);
   }

   TList list;
   list.Add(&p2);
   list.Add(&p3);

   p1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(p1, p4, cmpOptStats, 1E-8));
}
