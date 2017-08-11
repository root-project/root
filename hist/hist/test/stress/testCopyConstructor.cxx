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

#include <sstream>
#include "gtest/gtest.h"

using namespace std;

TEST(StressHistogram, TestCopyConstructor1D)
{
   TRandom2 r;
   // Tests the copy constructor for 1D Histograms

   TH1D h1("cc1D-h1", "h1-Title", numberOfBins, minRange, maxRange);

   h1.Sumw2();

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
   }

   TH1D h2(h1);

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats));
}

TEST(StressHistogram, TestCopyConstructor2D)
{
   TRandom2 r;
   // Tests the copy constructor for 2D Histograms

   TH2D h1("cc2D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, 1.0);
   }

   TH2D h2(h1);

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats));
}

TEST(StressHistogram, TestCopyConstructor3D)
{
   TRandom2 r;
   // Tests the copy constructor for 3D Histograms

   TH3D h1("cc3D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, z, 1.0);
   }

   TH3D h2(h1);

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats));
}

TEST(StressHistogram, TestCopyConstructorVar1D)
{
   TRandom2 r;
   // Tests the copy constructor for 1D Histograms with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   TH1D h1("cc1D-h1", "h1-Title", numberOfBins, v);

   h1.Sumw2();

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
   }

   TH1D h2(h1);

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats));
}
