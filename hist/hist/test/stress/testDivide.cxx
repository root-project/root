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

TEST(StressHistogram, TestDivide1)
{
   TRandom2 r;
   // Tests the first Divide method for 1D Histograms

   Double_t c1 = r.Rndm() + 1;
   Double_t c2 = r.Rndm() + 1;

   TH1D h1("d1D1-h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D h2("d1D1-h2", "h2-Title", numberOfBins, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value;
      value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
      value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(value, 1.0);
   }
   // avoid bins in h2 with zero content
   for (int i = 0; i < h2.GetSize(); ++i)
      if (h2.GetBinContent(i) == 0) h2.SetBinContent(i, 1);

   TH1D h3("d1D1-h3", "h3=(c1*h1)/(c2*h2)", numberOfBins, minRange, maxRange);
   h3.Divide(&h1, &h2, c1, c2);

   TH1D h4("d1D1-h4", "h4=h3*h2)", numberOfBins, minRange, maxRange);
   h4.Multiply(&h2, &h3, c2 / c1, 1);
   for (Int_t bin = 0; bin <= h4.GetNbinsX() + 1; ++bin) {
      Double_t error = h4.GetBinError(bin) * h4.GetBinError(bin);
      error -= (2 * (c2 * c2) / (c1 * c1)) * h3.GetBinContent(bin) * h3.GetBinContent(bin) * h2.GetBinError(bin) *
               h2.GetBinError(bin);
      h4.SetBinError(bin, sqrt(error));
   }

   h4.ResetStats();
   h1.ResetStats();

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats));
}

TEST(StressHistogram, TestDivide2)
{
   TRandom2 r;
   // Tests the second Divide method for 1D Histograms

   TH1D h1("d1D2-h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D h2("d1D2-h2", "h2-Title", numberOfBins, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value;
      value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
      value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(value, 1.0);
   }
   // avoid bins in h2 with zero content
   for (int i = 0; i < h2.GetSize(); ++i)
      if (h2.GetBinContent(i) == 0) h2.SetBinContent(i, 1);

   unique_ptr<TH1D> h3(static_cast<TH1D *>(h1.Clone()));
   h3->Divide(&h2);

   TH1D h4("d1D2-h4", "h4=h3*h2)", numberOfBins, minRange, maxRange);
   h4.Multiply(&h2, h3.get(), 1.0, 1.0);
   for (Int_t bin = 0; bin <= h4.GetNbinsX() + 1; ++bin) {
      Double_t error = h4.GetBinError(bin) * h4.GetBinError(bin);
      error -= 2 * h3->GetBinContent(bin) * h3->GetBinContent(bin) * h2.GetBinError(bin) * h2.GetBinError(bin);
      h4.SetBinError(bin, sqrt(error));
   }

   h4.ResetStats();
   h1.ResetStats();

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats));
}

/*
TEST(StressHistogram, TestDivideProf1)
{
   // Tests the first Divide method for 1D Profiles

   Double_t c1 = 1; // r.Rndm();
   Double_t c2 = 1; // r.Rndm();

   TProfile *p1 = new TProfile("d1D1-p1", "p1-Title", numberOfBins, minRange, maxRange);
   TProfile *p2 = new TProfile("d1D1-p2", "p2-Title", numberOfBins, minRange, maxRange);

   p1->Sumw2();
   p2->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x, y;
      x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, 1.0);
      x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p2->Fill(x, y, 1.0);
   }

   TProfile *p3 = new TProfile("d1D1-p3", "p3=(c1*p1)/(c2*p2)", numberOfBins, minRange, maxRange);
   p3->Divide(p1, p2, c1, c2);

   // There is no Multiply method to tests. And the errors are wrongly
   // calculated in the TProfile::Division method, so there is no
   // point to make the tests. Once the method is fixed, the tests
   // will be finished.

   SUCCEED();

   delete p1;
   delete p2;
   delete p3;
}*/