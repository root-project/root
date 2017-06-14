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

TYPED_TEST(HistTest, TestMulHn)
{
   // Tests the Multiply method for Sparse Histograms

   Int_t bsize[] = {TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5))};
   Double_t xmin[] = {minRange, minRange, minRange};
   Double_t xmax[] = {maxRange, maxRange, maxRange};

   auto s1 = this->ProduceHist("m3D2-s1", "s1-Title", 3, bsize, xmin, xmax);
   auto s2 = this->ProduceHist("m3D2-s2", "s2-Title", 3, bsize, xmin, xmax);
   auto s3 = this->ProduceHist("m3D2-s3", "s3=s1*s2", 3, bsize, xmin, xmax);

   s1->Sumw2();
   s2->Sumw2();
   s3->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t points[3];
      points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
      s1->Fill(points, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t points[3];
      points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
      s2->Fill(points, 1.0);
      Int_t points_s1[3];
      points_s1[0] = s1->GetAxis(0)->FindBin(points[0]);
      points_s1[1] = s1->GetAxis(1)->FindBin(points[1]);
      points_s1[2] = s1->GetAxis(2)->FindBin(points[2]);
      s3->Fill(points, s1->GetBinContent(points_s1));
   }

   // s3 has to be filled again so that the errors are properly calculated
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t points[3];
      points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
      Int_t points_s2[3];
      points_s2[0] = s2->GetAxis(0)->FindBin(points[0]);
      points_s2[1] = s2->GetAxis(1)->FindBin(points[1]);
      points_s2[2] = s2->GetAxis(2)->FindBin(points[2]);
      s3->Fill(points, s2->GetBinContent(points_s2));
   }

   // No the bin contents has to be reduced, as it was filled twice!
   for (Long64_t i = 0; i < s3->GetNbins(); ++i) {
      Int_t bin[3];
      Double_t v = s3->GetBinContent(i, bin);
      s3->SetBinContent(bin, v / 2);
   }

   s1->Multiply(s2.get());

   EXPECT_TRUE(HistogramsEquals(*(THnBase*)s3.get(), *(THnBase*)s1.get(), cmpOptNone, 1E-10));
}
