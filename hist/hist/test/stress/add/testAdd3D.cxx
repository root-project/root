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

TEST(StressHistogram, TestAdd3D1)
{
   // Tests the first Add method for 3D Histograms

   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TH3D h1("t3D1-h1", "h1", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);
   TH3D h2("t3D1-h2", "h2", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);
   TH3D h3("t3D1-h3", "h3=c1*h1+c2*h2", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange,
                       maxRange, numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, z, 1.0);
      h3.Fill(x, y, z, c1);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, z, 1.0);
      h3.Fill(x, y, z, c2);
   }

   TH3D h4("t3D1-h4", "h4=c1*h1+c2*h2", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange,
                       maxRange, numberOfBins + 2, minRange, maxRange);
   h4.Add(&h1, &h2, c1, c2);
   EXPECT_TRUE(HistogramsEquals(h3, h4, cmpOptStats, 1E-10));
}

TEST(StressHistogram, TestAdd3D2)
{
   // Tests the second Add method for 3D Histograms

   Double_t c2 = r.Rndm();

   TH3D h1("t3D2-h1", "h1", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);
   TH3D h2("t3D2-h2", "h2", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);
   TH3D h3("t3D2-h3", "h3=h1+c2*h2", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, z, 1.0);
      h3.Fill(x, y, z, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, z, 1.0);
      h3.Fill(x, y, z, c2);
   }

   h1.Add(&h2, c2);
   EXPECT_TRUE(HistogramsEquals(h3, h1, cmpOptStats, 1E-10));
}

TEST(StressHistogram, TestAdd3D3)
{
   // Tests the first add method to do scalation of 3D Histograms

   Double_t c1 = r.Rndm();

   TH3D h1("t1D1-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);
   TH3D h2("t1D1-h2", "h2=c1*h1+c2*h2", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange,
                       maxRange, numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, z, 1.0);
      Int_t binx = h1.GetXaxis()->FindBin(x);
      Int_t biny = h1.GetYaxis()->FindBin(y);
      Int_t binz = h1.GetZaxis()->FindBin(z);
      Double_t area =
         h1.GetXaxis()->GetBinWidth(binx) * h1.GetYaxis()->GetBinWidth(biny) * h1.GetZaxis()->GetBinWidth(binz);
      h2.Fill(x, y, z, c1 / area);
   }

   TH3D h3("t1D1-h3", "h3=c1*h1", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);
   h3.Add(&h1, &h1, c1, -1);

   // TH1::Add will reset the stats in this case so we need to do for the reference histogram
   h2.ResetStats();

   EXPECT_TRUE(HistogramsEquals(h2, h3, cmpOptStats, 1E-10));
}
