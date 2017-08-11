// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include "../StressHistogramGlobal.h"
#include "../TypedHistTest.h"

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TFile.h"
#include "TRandom2.h"

#include <sstream>
#include "gtest/gtest.h"

using namespace std;

typedef ::testing::Types<THnSparseD, THnD> HistTestTypes_t;
TYPED_TEST_CASE(HistTest, HistTestTypes_t);

TYPED_TEST(HistTest, TestMulFND)
{
   TRandom2 r;
   const UInt_t nDims = 3;
   Double_t c1 = r.Rndm();

   Int_t bsize[] = {TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5))};
   Double_t xmin[] = {minRange, minRange, minRange};
   Double_t xmax[] = {maxRange, maxRange, maxRange};

   auto s1 = this->ProduceHist("mfND-s1", "s1-Title", nDims, bsize, xmin, xmax);
   auto s2 = this->ProduceHist("mfND-s2", "s2=f*s2", nDims, bsize, xmin, xmax);

   TF1 f("sin", "sin(x)", minRange - 2, maxRange + 2);

   s1->Sumw2();
   s2->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t points[nDims];
      for (UInt_t i = 0; i < nDims; ++i) points[i] = r.Uniform(minRange * .9, maxRange * 1.1);
      s1->Fill(points, 1.0);
      s2->Fill(points, f.Eval(s2->GetAxis(0)->GetBinCenter(s2->GetAxis(0)->FindBin(points[0]))) * c1);
   }

   s1->Multiply(&f, c1);

   EXPECT_TRUE(HistogramsEquals(*(THnBase *)s1.get(), *(THnBase *)s2.get()));
}

TYPED_TEST(HistTest, TestMulFND2)
{
   TRandom2 r;
   const UInt_t nDims = 3;
   Double_t c1 = r.Rndm();

   Int_t bsize[] = {TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5))};
   Double_t xmin[] = {minRange, minRange, minRange};
   Double_t xmax[] = {maxRange, maxRange, maxRange};

   auto s1 = this->ProduceHist("mfND-s1", "s1-Title", nDims, bsize, xmin, xmax);
   auto s2 = this->ProduceHist("mfND-s2", "s2=f*s2", nDims, bsize, xmin, xmax);

   TF2 f("sin2", "sin(x)*cos(y)", minRange - 2, maxRange + 2, minRange - 2, maxRange + 2);

   s1->Sumw2();
   s2->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t points[nDims];
      for (UInt_t i = 0; i < nDims; ++i) points[i] = r.Uniform(minRange * .9, maxRange * 1.1);
      s1->Fill(points, 1.0);
      s2->Fill(points,
               f.Eval(s2->GetAxis(0)->GetBinCenter(s2->GetAxis(0)->FindBin(points[0])),
                      s2->GetAxis(1)->GetBinCenter(s2->GetAxis(1)->FindBin(points[1]))) *
                  c1);
   }

   s1->Multiply(&f, c1);

   EXPECT_TRUE(HistogramsEquals(*(THnBase *)s1.get(), *(THnBase *)s2.get()));
}
