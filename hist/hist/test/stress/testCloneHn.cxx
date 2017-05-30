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

#include "TRandom2.h"
#include "TFile.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"
#include "TypedHistTest.h"

using namespace std;

typedef ::testing::Types<THnSparseD, THnD> HistTestTypes_t;
TYPED_TEST_CASE(HistTest, HistTestTypes_t);

TYPED_TEST(HistTest, TestCloneHn)
{
   // Tests the clone method for Sparse histograms

   Int_t bsize[] = {TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5))};
   Double_t xmin[] = {minRange, minRange, minRange};
   Double_t xmax[] = {maxRange, maxRange, maxRange};

   auto s1 = this->ProduceHist("clS-s1", "s1-Title", 3, bsize, xmin, xmax);

   for (Int_t i = 0; i < nEvents * nEvents; ++i) {
      Double_t points[3];
      points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
      s1->Fill(points);
   }

   auto s2 = (TypeParam *)s1->Clone();

   EXPECT_TRUE(HistogramsEquals(s1, s2));
   delete s1;
}
