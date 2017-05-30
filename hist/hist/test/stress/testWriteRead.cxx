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

#include "TROOT.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"
#include "TypedHistTest.h"

using namespace std;

TEST(StressHistorgram, TestWriteRead1D)
{
   // Tests the write and read methods for 1D Histograms

   TH1D *h1 = new TH1D("wr1D-h1", "h1-Title", numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(value, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   h1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TH1D *h2 = static_cast<TH1D *>(f2.Get("wr1D-h1"));

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats));
   delete h1;
}

TEST(StressHistorgram, TestWriteRead2D)
{
   // Tests the write and read methods for 2D Histograms

   TH2D *h1 = new TH2D("wr2D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   h1->Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   h1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TH2D *h2 = static_cast<TH2D *>(f2.Get("wr2D-h1"));

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats));
   delete h1;
}

TEST(StressHistorgram, TestWriteRead3D)
{
   // Tests the write and read methods for 3D Histograms

   TH3D *h1 = new TH3D("wr3D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);

   h1->Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, z, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   h1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TH3D *h2 = static_cast<TH3D *>(f2.Get("wr3D-h1"));

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats));
   delete h1;
}

typedef ::testing::Types<THnSparseD, THnD> HistTestTypes_t;
TYPED_TEST_CASE(HistTest, HistTestTypes_t);

TYPED_TEST(HistTest, TestWriteReadHn)
{
   // Tests the write and read methods for n-dim Histograms

   Int_t bsize[] = {TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5))};
   Double_t xmin[] = {minRange, minRange, minRange};
   Double_t xmax[] = {maxRange, maxRange, maxRange};

   auto s1 = this->ProduceHist("wrS-s1", "s1-Title", 3, bsize, xmin, xmax);
   s1->Sumw2();

   for (Int_t i = 0; i < nEvents * nEvents; ++i) {
      Double_t points[3];
      points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
      points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
      s1->Fill(points);
   }

   TFile f("tmpHist.root", "RECREATE");
   s1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   auto s2 = static_cast<TypeParam *>(f2.Get("wrS-s1"));

   EXPECT_TRUE(HistogramsEquals(s1, s2, cmpOptStats));
   delete s1;
}

TEST(StressHistorgram, TestWriteReadVar1D)
{
   // Tests the write and read methods for 1D Histograms with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   TH1D *h1 = new TH1D("wr1D-h1", "h1-Title", numberOfBins, v);

   h1->Sumw2();

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(value, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   h1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TH1D *h2 = static_cast<TH1D *>(f2.Get("wr1D-h1"));

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats));
   delete h1;
}
