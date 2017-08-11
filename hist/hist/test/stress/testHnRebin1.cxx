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

#include "TFile.h"
#include "TRandom2.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

#include "TypedHistTest.h"

typedef ::testing::Types<THnSparseD, THnD> HistTestTypes_t;
TYPED_TEST_CASE(HistTest, HistTestTypes_t);

TYPED_TEST(HistTest, TestHnRebin1)
{
   TRandom2 r;
   // Tests rebin method for n-dim Histogram

   const int rebin = TMath::Nint(r.Uniform(minRebin, maxRebin));

   Int_t bsizeRebin[] = {TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5))};

   Int_t bsize[] = {bsizeRebin[0] * rebin, bsizeRebin[1] * rebin, bsizeRebin[2] * rebin};

   Double_t xmin[] = {minRange, minRange, minRange};
   Double_t xmax[] = {maxRange, maxRange, maxRange};
   auto s1 = this->ProduceHist("rebin1-s1", "s1-Title", 3, bsize, xmin, xmax);
   auto s2 = this->ProduceHist("rebin1-s2", "s2-Title", 3, bsizeRebin, xmin, xmax);

   for (Int_t i = 0; i < nEvents; ++i) {
      Double_t points[3];
      points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
      s1->Fill(points);
      s2->Fill(points);
   }

   auto s3 = s1->Rebin(rebin);

   EXPECT_TRUE(HistogramsEquals(*((THnBase *)s2.get()), *((THnBase *)s3)));

   delete s3;
}
