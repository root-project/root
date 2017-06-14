// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>
#include <cmath>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TApplication.h"
#include "Riostream.h"
#include "TRandom2.h"
#include "TFile.h"
#include "TClass.h"

#include "gtest/gtest.h"

#include "../StressHistogramGlobal.h"

using namespace std;

class MergeTest : public ::testing::TestWithParam<bool> {
};

TEST_P(MergeTest, TestMerge1D_Diff)
{
   bool testEmpty = GetParam();
   // Tests the merge method with different binned 1D Histograms
   // test also case when the first histogram is empty (bug Savannah 95190)

   TH1D h1("h1", "h1-Title", 100, -100, 0);
   TH1D h2("h2", "h2-Title", 200, 0, 100);
   TH1D h3("h3", "h3-Title", 25, -50, 50);
   // resulting histogram will have the bigger range and the larger bin width
   // eventually range is extended by half bin width to have correct bin boundaries
   // of largest bin width histogram
   TH1D h4("h4", "h4-Title", 51, -102, 102);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();
   h4.Sumw2();

   if (!testEmpty) {
      for (Int_t e = 0; e < nEvents; ++e) {
         Double_t value = r.Gaus(-50, 10);
         h1.Fill(value, 1.0);
         h4.Fill(value, 1.0);
      }
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Gaus(50, 10);
      h2.Fill(value, 1.0);
      h4.Fill(value, 1.0);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Gaus(0, 10);
      h3.Fill(value, 1.0);
      h4.Fill(value, 1.0);
   }

   TList list;
   list.Add(&h2);
   list.Add(&h3);

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}

TEST_P(MergeTest, TestMergeProf1D_Diff)
{
   bool testEmpty = GetParam();
   // Tests the merge method with different binned 1D Profile

   // Stats fail, for a reason I do not know :S

   TProfile p1("merge1DDiff-p1", "p1-Title", 110, -110, 0);
   TProfile p2("merge1DDiff-p2", "p2-Title", 220, 0, 110);
   TProfile p3("merge1DDiff-p3", "p3-Title", 330, -55, 55);
   TProfile p4("merge1DDiff-p4", "p4-Title", 220, -110, 110);

   if (!testEmpty) {
      for (Int_t e = 0; e < nEvents; ++e) {
         Double_t x = r.Gaus(-55, 10);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p1.Fill(x, y, 1.0);
         p4.Fill(x, y, 1.0);
      }
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Gaus(55, 10);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p2.Fill(x, y, 1.0);
      p4.Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Gaus(0, 10);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p3.Fill(x, y, 1.0);
      p4.Fill(x, y, 1.0);
   }

   TList list;
   list.Add(&p2);
   list.Add(&p3);

   p1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(p1, p4, cmpOptNone, 1E-10));
}

TEST_P(MergeTest, TestMerge3DDiff)
{
   bool testEmpty = GetParam();
   // Tests the merge method with different binned 3D Histograms

   TH3D h1("merge3DDiff-h1", "h1-Title", 11, -110, 0, 11, -110, 0, 11, -110, 0);
   TH3D h2("merge3DDiff-h2", "h2-Title", 22, 0, 110, 22, 0, 110, 22, 0, 110);
   TH3D h3("merge3DDiff-h3", "h3-Title", 44, -55, 55, 44, -55, 55, 44, -55, 55);
   TH3D h4("merge3DDiff-h4", "h4-Title", 22, -110, 110, 22, -110, 110, 22, -110, 110);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();
   h4.Sumw2();

   if (!testEmpty) {
      for (Int_t e = 0; e < nEvents; ++e) {
         Double_t x = r.Gaus(-55, 10);
         Double_t y = r.Gaus(-55, 10);
         Double_t z = r.Gaus(-55, 10);
         h1.Fill(x, y, z, 1.0);
         h4.Fill(x, y, z, 1.0);
      }
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Gaus(55, 10);
      Double_t y = r.Gaus(55, 10);
      Double_t z = r.Gaus(55, 10);
      h2.Fill(x, y, z, 1.0);
      h4.Fill(x, y, z, 1.0);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Gaus(0, 10);
      Double_t y = r.Gaus(0, 10);
      Double_t z = r.Gaus(0, 10);
      h3.Fill(x, y, z, 1.0);
      h4.Fill(x, y, z, 1.0);
   }

   TList list;
   list.Add(&h2);
   list.Add(&h3);

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}

TEST_P(MergeTest, TestMerge2D_Diff)
{
   bool testEmpty = GetParam();
   // Tests the merge method with different binned 2D Histograms

   // LM. t.b.u.: for 1D can make h3 with 330 bins , while in 2D if I make h3 with 33 bins
   //  routine which check axis fails. Needs to be improved ???

   TH2D h1("merge2DDiff-h1", "h1-Title", 11, -110, 0, 11, -110, 0);
   TH2D h2("merge2DDiff-h2", "h2-Title", 22, 0, 110, 22, 0, 110);
   TH2D h3("merge2DDiff-h3", "h3-Title", 44, -55, 55, 44, -55, 55);
   TH2D h4("merge2DDiff-h4", "h4-Title", 22, -110, 110, 22, -110, 110);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();
   h4.Sumw2();

   if (!testEmpty) {
      for (Int_t e = 0; e < nEvents; ++e) {
         Double_t x = r.Gaus(-55, 10);
         Double_t y = r.Gaus(-55, 10);
         h1.Fill(x, y, 1.0);
         h4.Fill(x, y, 1.0);
      }
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Gaus(55, 10);
      Double_t y = r.Gaus(55, 10);
      h2.Fill(x, y, 1.0);
      h4.Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Gaus(0, 10);
      Double_t y = r.Gaus(0, 10);
      h3.Fill(x, y, 1.0);
      h4.Fill(x, y, 1.0);
   }

   TList list;
   list.Add(&h2);
   list.Add(&h3);

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}

INSTANTIATE_TEST_CASE_P(StressHistogram, MergeTest, ::testing::Values(true, false));
