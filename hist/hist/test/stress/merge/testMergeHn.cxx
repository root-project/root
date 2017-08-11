// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TApplication.h"
#include "TFile.h"
#include "TRandom2.h"

#include "gtest/gtest.h"

#include "../StressHistogramGlobal.h"
#include "../TypedHistTest.h"

using namespace std;

typedef ::testing::Types<THnSparseD, THnD> HistTestTypes_t;
TYPED_TEST_CASE(HistTest, HistTestTypes_t);

TYPED_TEST(HistTest, TestMergeHn)
{
   TRandom2 r;
   // Tests the merge method for n-dim Histograms

   Int_t bsize[] = {TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5))};
   Double_t xmin[] = {minRange, minRange, minRange};
   Double_t xmax[] = {maxRange, maxRange, maxRange};

   auto s1 = this->ProduceHist("mergeS-s1", "s1-Title", 3, bsize, xmin, xmax);
   auto s2 = this->ProduceHist("mergeS-s2", "s2-Title", 3, bsize, xmin, xmax);
   auto s3 = this->ProduceHist("mergeS-s3", "s3-Title", 3, bsize, xmin, xmax);
   auto s4 = this->ProduceHist("mergeS-s4", "s4-Title", 3, bsize, xmin, xmax);

   s1->Sumw2();
   s2->Sumw2();
   s3->Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t points[3];
      points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
      s1->Fill(points, 1.0);
      s4->Fill(points, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t points[3];
      points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
      s2->Fill(points, 1.0);
      s4->Fill(points, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t points[3];
      points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
      s3->Fill(points, 1.0);
      s4->Fill(points, 1.0);
   }

   TList list;
   list.Add(s2.get());
   list.Add(s3.get());

   s1->Merge(&list);

   EXPECT_TRUE(HistogramsEquals(*(THnBase *)s1.get(), *(THnBase *)s4.get(), cmpOptNone, 1E-10));
}
