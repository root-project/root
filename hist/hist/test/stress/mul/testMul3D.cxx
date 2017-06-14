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

TEST(StressHistorgram, TestMul3D1)
{
   // Tests the first Multiply method for 3D Histograms

   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TH3D h1("m3D1-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);
   TH3D h2("m3D1-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);
   TH3D h3("m3D1-h3", "h3=c1*h1*c2*h2", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange,
                       maxRange, numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, z, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, z, 1.0);
      h3.Fill(
         x, y, z,
         c1 * c2 *
            h1.GetBinContent(h1.GetXaxis()->FindBin(x), h1.GetYaxis()->FindBin(y), h1.GetZaxis()->FindBin(z)));
   }

   // h3 has to be filled again so that the erros are properly calculated
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3.Fill(
         x, y, z,
         c1 * c2 *
            h2.GetBinContent(h2.GetXaxis()->FindBin(x), h2.GetYaxis()->FindBin(y), h2.GetZaxis()->FindBin(z)));
   }

   // No the bin contents has to be reduced, as it was filled twice!
   for (Int_t i = 0; i <= h3.GetNbinsX() + 1; ++i) {
      for (Int_t j = 0; j <= h3.GetNbinsY() + 1; ++j) {
         for (Int_t h = 0; h <= h3.GetNbinsZ() + 1; ++h) {
            h3.SetBinContent(i, j, h, h3.GetBinContent(i, j, h) / 2);
         }
      }
   }

   TH3D h4("m3D1-h4", "h4=h1*h2", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);
   h4.Multiply(&h1, &h2, c1, c2);

   EXPECT_TRUE(HistogramsEquals(h3, h4, cmpOptStats, 1E-13));
}

TEST(StressHistorgram, TestMul3D2)
{
   // Tests the second Multiply method for 3D Histograms

   TH3D h1("m3D2-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);
   TH3D h2("m3D2-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);
   TH3D h3("m3D2-h3", "h3=h1*h2", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, z, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, z, 1.0);
      h3.Fill(x, y, z,
               h1.GetBinContent(h1.GetXaxis()->FindBin(x), h1.GetYaxis()->FindBin(y), h1.GetZaxis()->FindBin(z)));
   }

   // h3 has to be filled again so that the errors are properly calculated
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3.Fill(x, y, z,
               h2.GetBinContent(h2.GetXaxis()->FindBin(x), h2.GetYaxis()->FindBin(y), h2.GetZaxis()->FindBin(z)));
   }

   // No the bin contents has to be reduced, as it was filled twice!
   for (Int_t i = 0; i <= h3.GetNbinsX() + 1; ++i) {
      for (Int_t j = 0; j <= h3.GetNbinsY() + 1; ++j) {
         for (Int_t h = 0; h <= h3.GetNbinsZ() + 1; ++h) {
            h3.SetBinContent(i, j, h, h3.GetBinContent(i, j, h) / 2);
         }
      }
   }

   h1.Multiply(&h2);

   EXPECT_TRUE(HistogramsEquals(h3, h1, cmpOptStats, 1E-13));
}
