// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile3D.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TApplication.h"
#include "TRandom2.h"
#include "TFile.h"

#include "gtest/gtest.h"

#include "../StressHistogramGlobal.h"

using namespace std;

TEST(StressHistogram, TestMergeProf3D)
{
   // Tests the merge method for 3D Profiles

   TProfile3D p1("merge3D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                 numberOfBins + 2, minRange, maxRange);
   TProfile3D p2("merge3D-p2", "p2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                 numberOfBins + 2, minRange, maxRange);
   TProfile3D p3("merge3D-p3", "p3-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                 numberOfBins + 2, minRange, maxRange);
   TProfile3D p4("merge3D-p4", "p4-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                 numberOfBins + 2, minRange, maxRange);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, z, t, 1.0);
      p4.Fill(x, y, z, t, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p2.Fill(x, y, z, t, 1.0);
      p4.Fill(x, y, z, t, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p3.Fill(x, y, z, t, 1.0);
      p4.Fill(x, y, z, t, 1.0);
   }

   TList list;
   list.Add(&p2);
   list.Add(&p3);

   p1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(p1, p4, cmpOptStats, 1E-10));
}

TEST(StressHistogram, TestMergeProf3DDiff)
{
   // Tests the merge method with different binned 3D Profile

   // This tests fails! Segmentation Fault!!It should not!
   TProfile3D p1("merge3DDiff-p1", "p1-Title", 11, -110, 0, 11, -110, 0, 11, -110, 0);
   TProfile3D p2("merge3DDiff-p2", "p2-Title", 22, 0, 110, 22, 0, 110, 22, 0, 110);
   TProfile3D p3("merge3DDiff-p3", "p3-Title", 44, -55, 55, 44, -55, 55, 44, -55, 55);
   TProfile3D p4("merge3DDiff-p4", "p4-Title", 22, -110, 110, 22, -110, 110, 22, -110, 110);

   for (Int_t e = 0; e < 10 * nEvents; ++e) {
      Double_t x = r.Uniform(-110, 0);
      Double_t y = r.Uniform(-110, 0);
      Double_t z = r.Uniform(-110, 0);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, z, t, 1.0);
      p4.Fill(x, y, z, t, 1.0);
   }

   for (Int_t e = 0; e < 10 * nEvents; ++e) {
      Double_t x = r.Uniform(0, 110);
      Double_t y = r.Uniform(0, 110);
      Double_t z = r.Uniform(0, 110);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p2.Fill(x, y, z, t, 1.0);
      p4.Fill(x, y, z, t, 1.0);
   }

   for (Int_t e = 0; e < 10 * nEvents; ++e) {
      Double_t x = r.Uniform(-55, 55);
      Double_t y = r.Uniform(-55, 55);
      Double_t z = r.Uniform(-55, 55);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p3.Fill(x, y, z, t, 1.0);
      p4.Fill(x, y, z, t, 1.0);
   }

   TList list;
   list.Add(&p2);
   list.Add(&p3);

   p1.Merge(&list);

   // exclude statistics in comparison since chi2 test will fail with low
   // bin statistics
   EXPECT_TRUE(HistogramsEquals(p1, p4, cmpOptNone, 1E-10));
}
