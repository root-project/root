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

TEST(StressHistogram, TestMulVar1)
{
   TRandom2 r;
   // Tests the first Multiply method for 1D Histograms with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TH1D h1("m1D1-h1", "h1-Title", numberOfBins, v);
   TH1D h2("m1D1-h2", "h2-Title", numberOfBins, v);
   TH1D h3("m1D1-h3", "h3=c1*h1*c2*h2", numberOfBins, v);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(value, 1.0);
      h3.Fill(value, c1 * c2 * h1.GetBinContent(h1.GetXaxis()->FindBin(value)));
   }

   // h3 has to be filled again so that the erros are properly calculated
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3.Fill(value, c1 * c2 * h2.GetBinContent(h2.GetXaxis()->FindBin(value)));
   }

   // No the bin contents has to be reduced, as it was filled twice!
   for (Int_t bin = 0; bin <= h3.GetNbinsX() + 1; ++bin) {
      h3.SetBinContent(bin, h3.GetBinContent(bin) / 2);
   }

   TH1D h4("m1D1-h4", "h4=h1*h2", numberOfBins, v);
   h4.Multiply(&h1, &h2, c1, c2);

   EXPECT_TRUE(HistogramsEquals(h3, h4, cmpOptStats, 1E-14));
}

TEST(StressHistogram, TestMulVar2)
{
   TRandom2 r(initialRandomSeed);
   // Tests the second Multiply method for 1D Histograms with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   TH1D h1("m1D2-h1", "h1-Title", numberOfBins, v);
   TH1D h2("m1D2-h2", "h2-Title", numberOfBins, v);
   TH1D h3("m1D2-h3", "h3=h1*h2", numberOfBins, v);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(value, 1.0);
      h3.Fill(value, h1.GetBinContent(h1.GetXaxis()->FindBin(value)));
   }

   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3.Fill(value, h2.GetBinContent(h2.GetXaxis()->FindBin(value)));
   }

   for (Int_t bin = 0; bin <= h3.GetNbinsX() + 1; ++bin) {
      h3.SetBinContent(bin, h3.GetBinContent(bin) / 2);
   }

   h1.Multiply(&h2);

   EXPECT_TRUE(HistogramsEquals(h3, h1, cmpOptStats, 1E-14));
}
