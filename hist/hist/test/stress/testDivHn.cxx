// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include "StressHistogramGlobal.h"
#include "TypedHistTest.h"

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TFile.h"
#include "TRandom2.h"

#include <cmath>
#include <sstream>
#include "gtest/gtest.h"

using namespace std;

typedef ::testing::Types<THnSparseD, THnD> HistTestTypes_t;
TYPED_TEST_CASE(HistTest, HistTestTypes_t);

TYPED_TEST(HistTest, TestDivHn1)
{
   TRandom2 r;
   // Tests the first Divide method for 3D Histograms

   Int_t bsize[] = {TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5))};
   Double_t xmin[] = {minRange, minRange, minRange};
   Double_t xmax[] = {maxRange, maxRange, maxRange};

   // There is no multiply with coefficients!
   const Double_t c1 = 1;
   const Double_t c2 = 1;

   auto s1 = this->ProduceHist("dND1-s1", "s1-Title", 3, bsize, xmin, xmax);
   auto s2 = this->ProduceHist("dND1-s2", "s2-Title", 3, bsize, xmin, xmax);
   auto s4 = this->ProduceHist("dND1-s4", "s4=s3*s2)", 3, bsize, xmin, xmax);

   s1->Sumw2();
   s2->Sumw2();
   s4->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t points[3];
      points[0] = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      points[1] = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      points[2] = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      s1->Fill(points, 1.0);
      points[0] = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      points[1] = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      points[2] = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      s2->Fill(points, 1.0);
      s4->Fill(points, 1.0);
   }

   auto s3 = this->ProduceHist("dND1-s3", "s3=(c1*s1)/(c2*s2)", 3, bsize, xmin, xmax);
   s3->Divide(s1.get(), s2.get(), c1, c2);

   s4->Multiply(s3.get());

   // No the bin contents has to be reduced, as it was filled twice!
   for (Long64_t i = 0; i < s3->GetNbins(); ++i) {
      Int_t coord[3];
      s3->GetBinContent(i, coord);
      Double_t s4BinError = s4->GetBinError(coord);
      Double_t s2BinError = s2->GetBinError(coord);
      Double_t s3BinContent = s3->GetBinContent(coord);
      Double_t error = s4BinError * s4BinError;
      error -= (2 * (c2 * c2) / (c1 * c1)) * s3BinContent * s3BinContent * s2BinError * s2BinError;
      s4->SetBinError(coord, sqrt(error));
   }

   EXPECT_TRUE(HistogramsEquals(*(THnBase *)s1.get(), *(THnBase *)s4.get(), cmpOptStats, 1E-6));
}

TYPED_TEST(HistTest, TestDivHn2)
{
   TRandom2 r;
   // Tests the second Divide method for 3D Histograms

   Int_t bsize[] = {TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5))};
   Double_t xmin[] = {minRange, minRange, minRange};
   Double_t xmax[] = {maxRange, maxRange, maxRange};

   // There is no multiply with coefficients!
   const Double_t c1 = 1;
   const Double_t c2 = 1;

   auto s1 = this->ProduceHist("dND2-s1", "s1-Title", 3, bsize, xmin, xmax);
   auto s2 = this->ProduceHist("dND2-s2", "s2-Title", 3, bsize, xmin, xmax);
   auto s4 = this->ProduceHist("dND2-s4", "s4=s3*s2)", 3, bsize, xmin, xmax);

   s1->Sumw2();
   s2->Sumw2();
   s4->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t points[3];
      points[0] = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      points[1] = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      points[2] = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      s1->Fill(points, 1.0);
      points[0] = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      points[1] = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      points[2] = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      s2->Fill(points, 1.0);
      s4->Fill(points, 1.0);
   }

   unique_ptr<TypeParam> s3(static_cast<TypeParam *>(s1->Clone()));
   s3->Divide(s2.get());

   auto s5 = this->ProduceHist("dND2-s5", "s5=(c1*s1)/(c2*s2)", 3, bsize, xmin, xmax);
   s5->Divide(s1.get(), s2.get());

   s4->Multiply(s3.get());

   // No the bin contents has to be reduced, as it was filled twice!
   for (Long64_t i = 0; i < s3->GetNbins(); ++i) {
      Int_t coord[3];
      s3->GetBinContent(i, coord);
      Double_t s4BinError = s4->GetBinError(coord);
      Double_t s2BinError = s2->GetBinError(coord);
      Double_t s3BinContent = s3->GetBinContent(coord);
      Double_t error = s4BinError * s4BinError;
      error -= (2 * (c2 * c2) / (c1 * c1)) * s3BinContent * s3BinContent * s2BinError * s2BinError;
      s4->SetBinError(coord, sqrt(error));
   }

   EXPECT_TRUE(HistogramsEquals(*(THnBase *)s1.get(), *(THnBase *)s4.get(), cmpOptStats, 1E-6));
}
