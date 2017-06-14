// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>
#include <cmath>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TRandom2.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestDivide2D1)
{
   // Tests the first Divide method for 2D Histograms

   Double_t c1 = r.Rndm() + 1;
   Double_t c2 = r.Rndm() + 1;

   TH2D *h1 = new TH2D("d2D1-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D *h2 = new TH2D("d2D1-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   h1->Sumw2();
   h2->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x, y;
      x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
      x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y, 1.0);
   }
   // avoid bins in h2 with zero content
   for (int i = 0; i < h2->GetSize(); ++i)
      if (h2->GetBinContent(i) == 0) h2->SetBinContent(i, 1);

   TH2D *h3 =
      new TH2D("d2D1-h3", "h3=(c1*h1)/(c2*h2)", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   h3->Divide(h1, h2, c1, c2);

   TH2D *h4 = new TH2D("d2D1-h4", "h4=h3*h2)", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   h4->Multiply(h2, h3, c2 / c1, 1);
   for (Int_t i = 0; i <= h4->GetNbinsX() + 1; ++i) {
      for (Int_t j = 0; j <= h4->GetNbinsY() + 1; ++j) {
         Double_t error = h4->GetBinError(i, j) * h4->GetBinError(i, j);
         error -= (2 * (c2 * c2) / (c1 * c1)) * h3->GetBinContent(i, j) * h3->GetBinContent(i, j) *
                  h2->GetBinError(i, j) * h2->GetBinError(i, j);
         h4->SetBinError(i, j, sqrt(error));
      }
   }

   h4->ResetStats();
   h1->ResetStats();

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats));
   delete h1;
   delete h2;
   delete h3;
   delete h4;
}

TEST(StressHistorgram, TestDivide2D2)
{
   // Tests the second Divide method for 2D Histograms

   TH2D *h1 = new TH2D("d2D2-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D *h2 = new TH2D("d2D2-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   h1->Sumw2();
   h2->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x, y;
      x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
      x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y, 1.0);
   }
   // avoid bins in h2 with zero content
   for (int i = 0; i < h2->GetSize(); ++i)
      if (h2->GetBinContent(i) == 0) h2->SetBinContent(i, 1);

   TH2D *h3 = static_cast<TH2D *>(h1->Clone());
   h3->Divide(h2);

   TH2D *h4 = new TH2D("d2D2-h4", "h4=h3*h2)", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   h4->Multiply(h2, h3, 1.0, 1.0);
   for (Int_t i = 0; i <= h4->GetNbinsX() + 1; ++i) {
      for (Int_t j = 0; j <= h4->GetNbinsY() + 1; ++j) {
         Double_t error = h4->GetBinError(i, j) * h4->GetBinError(i, j);
         error -= 2 * h3->GetBinContent(i, j) * h3->GetBinContent(i, j) * h2->GetBinError(i, j) * h2->GetBinError(i, j);
         h4->SetBinError(i, j, sqrt(error));
      }
   }

   h4->ResetStats();
   h1->ResetStats();

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats));
   delete h1;
   delete h2;
   delete h3;
   delete h4;
}
