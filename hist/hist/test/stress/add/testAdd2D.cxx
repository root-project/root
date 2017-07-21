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

TEST(StressHistogram, TestAdd2D1)
{
   TRandom2 r(initialRandomSeed);
   // Tests the first Add method for 2D Histograms

   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TH2D h1("t2D1-h1", "h1", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D h2("t2D1-h2", "h2", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D h3("t2D1-h3", "h3=c1*h1+c2*h2", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, 1.0);
      h3.Fill(x, y, c1);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, 1.0);
      h3.Fill(x, y, c2);
   }

   TH2D h4("t2D1-h4", "h4=c1*h1+c2*h2", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   h4.Add(&h1, &h2, c1, c2);
   EXPECT_TRUE(HistogramsEquals(h3, h4, cmpOptStats, 1E-10));
}

TEST(StressHistogram, TestAdd2D2)
{
   TRandom2 r(initialRandomSeed);
   // Tests the second Add method for 2D Histograms

   Double_t c2 = r.Rndm();

   TH2D h1("t2D2-h1", "h1", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D h2("t2D2-h2", "h2", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D h3("t2D2-h3", "h3=h1+c2*h2", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, 1.0);
      h3.Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, 1.0);
      h3.Fill(x, y, c2);
   }

   h1.Add(&h2, c2);
   EXPECT_TRUE(HistogramsEquals(h3, h1, cmpOptStats, 1E-10));
}

TEST(StressHistogram, TestAdd2D3)
{
   TRandom2 r(initialRandomSeed);
   // Tests the first add method to do scale of 2D Histograms

   Double_t c1 = r.Rndm();

   TH2D h1("t1D1-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D h2("t1D1-h2", "h2=c1*h1+c2*h2", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, 1.0);
      Int_t binx = h1.GetXaxis()->FindBin(x);
      Int_t biny = h1.GetYaxis()->FindBin(y);
      Double_t area = h1.GetXaxis()->GetBinWidth(binx) * h1.GetYaxis()->GetBinWidth(biny);
      h2.Fill(x, y, c1 / area);
   }

   TH2D h3("t1D1-h3", "h3=c1*h1", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   h3.Add(&h1, &h1, c1, -1);

   // TH1::Add will reset the stats in this case so we need to do for the reference histogram
   h2.ResetStats();

   EXPECT_TRUE(HistogramsEquals(h2, h3, cmpOptStats, 1E-10));
}
