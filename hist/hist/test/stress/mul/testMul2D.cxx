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

#include "gtest/gtest.h"

#include "../StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestMul2D1)
{
   // Tests the first Multiply method for 2D Histograms

   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TH2D *h1 = new TH2D("m2D1-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D *h2 = new TH2D("m2D1-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D *h3 =
      new TH2D("m2D1-h3", "h3=c1*h1*c2*h2", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   h1->Sumw2();
   h2->Sumw2();
   h3->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y, 1.0);
      h3->Fill(x, y, c1 * c2 * h1->GetBinContent(h1->GetXaxis()->FindBin(x), h1->GetYaxis()->FindBin(y)));
   }

   // h3 has to be filled again so that the erros are properly calculated
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3->Fill(x, y, c1 * c2 * h2->GetBinContent(h2->GetXaxis()->FindBin(x), h2->GetYaxis()->FindBin(y)));
   }

   // No the bin contents has to be reduced, as it was filled twice!
   for (Int_t i = 0; i <= h3->GetNbinsX() + 1; ++i) {
      for (Int_t j = 0; j <= h3->GetNbinsY() + 1; ++j) {
         h3->SetBinContent(i, j, h3->GetBinContent(i, j) / 2);
      }
   }

   TH2D *h4 = new TH2D("m2D1-h4", "h4=h1*h2", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   h4->Multiply(h1, h2, c1, c2);

   EXPECT_TRUE(HistogramsEquals(h3, h4, cmpOptStats, 1E-12));
   delete h1;
   delete h2;
   delete h3;
   delete h4;
}

TEST(StressHistorgram, TestMul2D2)
{
   // Tests the second Multiply method for 2D Histograms

   TH2D *h1 = new TH2D("m2D2-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D *h2 = new TH2D("m2D2-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D *h3 = new TH2D("m2D2-h3", "h3=h1*h2", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   h1->Sumw2();
   h2->Sumw2();
   h3->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y, 1.0);
      h3->Fill(x, y, h1->GetBinContent(h1->GetXaxis()->FindBin(x), h1->GetYaxis()->FindBin(y)));
   }

   // h3 has to be filled again so that the erros are properly calculated
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3->Fill(x, y, h2->GetBinContent(h2->GetXaxis()->FindBin(x), h2->GetYaxis()->FindBin(y)));
   }

   // No the bin contents has to be reduced, as it was filled twice!
   for (Int_t i = 0; i <= h3->GetNbinsX() + 1; ++i) {
      for (Int_t j = 0; j <= h3->GetNbinsY() + 1; ++j) {
         h3->SetBinContent(i, j, h3->GetBinContent(i, j) / 2);
      }
   }

   h1->Multiply(h2);

   EXPECT_TRUE(HistogramsEquals(h3, h1, cmpOptStats, 1E-12));
   delete h1;
   delete h2;
   delete h3;
}
