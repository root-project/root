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

#include "../StressHistogramGlobal.h"
#include "../TypedHistTest.h"

using namespace std;

typedef ::testing::Types<THnSparseD, THnD> HistTestTypes_t;
TYPED_TEST_CASE(HistTest, HistTestTypes_t);

TYPED_TEST(HistTest, TestAddHn)
{
   TRandom2 r;
   // Tests the Add method for n-dimensional Histograms

   Double_t c = r.Rndm();

   Int_t bsize[] = {TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5))};
   Double_t xmin[] = {minRange, minRange, minRange};
   Double_t xmax[] = {maxRange, maxRange, maxRange};

   auto s1 = this->ProduceHist("tS-s1", "s1", 3, bsize, xmin, xmax);
   auto s2 = this->ProduceHist("tS-s2", "s2", 3, bsize, xmin, xmax);
   auto s3 = this->ProduceHist("tS-s3", "s3=s1+c*s2", 3, bsize, xmin, xmax);

   s1->Sumw2();
   s2->Sumw2();
   s3->Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t points[3];
      points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
      s1->Fill(points);
      s3->Fill(points);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t points[3];
      points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
      s2->Fill(points);
      s3->Fill(points, c);
   }

   s1->Add(s2.get(), c);

   EXPECT_TRUE(HistogramsEquals(*(THnBase *)s3.get(), *(THnBase *)s1.get(), cmpOptStats, 1E-10));
}
