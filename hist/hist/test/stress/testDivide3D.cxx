// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include "StressHistogramGlobal.h"

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TRandom2.h"

#include <cmath>
#include <sstream>
#include "gtest/gtest.h"

using namespace std;

TEST(StressHistogram, TestDivide3D1)
{
   TRandom2 r;
   // Tests the first Divide method for 3D Histograms

   Double_t c1 = r.Rndm() + 1;
   Double_t c2 = r.Rndm() + 1;

   TH3D h1("d3D1-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h2("d3D1-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x, y, z;
      x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, z, 1.0);
      x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, z, 1.0);
   }
   // avoid bins in h2 with zero content
   for (int i = 0; i < h2.GetSize(); ++i)
      if (h2.GetBinContent(i) == 0) h2.SetBinContent(i, 1);

   TH3D h3("d3D1-h3", "h3=(c1*h1)/(c2*h2)", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   h3.Divide(&h1, &h2, c1, c2);

   TH3D h4("d3D1-h4", "h4=h3*h2)", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   h4.Multiply(&h2, &h3, c2 / c1, 1.0);
   for (Int_t i = 0; i <= h4.GetNbinsX() + 1; ++i) {
      for (Int_t j = 0; j <= h4.GetNbinsY() + 1; ++j) {
         for (Int_t h = 0; h <= h4.GetNbinsZ() + 1; ++h) {
            Double_t error = h4.GetBinError(i, j, h) * h4.GetBinError(i, j, h);
            // error -= 2 *
            // h3->GetBinContent(i,j,h)*h3->GetBinContent(i,j,h)*h2->GetBinError(i,j,h)*h2->GetBinError(i,j,h);
            error -= (2 * (c2 * c2) / (c1 * c1)) * h3.GetBinContent(i, j, h) * h3.GetBinContent(i, j, h) *
                     h2.GetBinError(i, j, h) * h2.GetBinError(i, j, h);
            h4.SetBinError(i, j, h, sqrt(error));
         }
      }
   }

   h4.ResetStats();
   h1.ResetStats();

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats));
}

TEST(StressHistogram, TestDivide3D2)
{
   TRandom2 r;
   // Tests the second Divide method for 3D Histograms

   TH3D h1("d3D2-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h2("d3D2-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x, y, z;
      x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, z, 1.0);
      x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, z, 1.0);
   }
   // avoid bins in h2 with zero content
   for (int i = 0; i < h2.GetSize(); ++i)
      if (h2.GetBinContent(i) == 0) h2.SetBinContent(i, 1);

   unique_ptr<TH3D> h3(static_cast<TH3D *>(h1.Clone()));
   h3->Divide(&h2);

   TH3D h4("d3D2-h4", "h4=h3*h2)", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   h4.Multiply(&h2, h3.get(), 1.0, 1.0);
   for (Int_t i = 0; i <= h4.GetNbinsX() + 1; ++i) {
      for (Int_t j = 0; j <= h4.GetNbinsY() + 1; ++j) {
         for (Int_t h = 0; h <= h4.GetNbinsZ() + 1; ++h) {
            Double_t error = h4.GetBinError(i, j, h) * h4.GetBinError(i, j, h);
            error -= 2 * h3->GetBinContent(i, j, h) * h3->GetBinContent(i, j, h) * h2.GetBinError(i, j, h) *
                     h2.GetBinError(i, j, h);
            h4.SetBinError(i, j, h, sqrt(error));
         }
      }
   }

   h4.ResetStats();
   h1.ResetStats();

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats));
}
