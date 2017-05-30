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

#include "StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestAssign1D)
{
   // Tests the operator=() method for 1D Histograms

   TH1D *h1 = new TH1D("=1D-h1", "h1-Title", numberOfBins, minRange, maxRange);

   h1->Sumw2();

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(value, 1.0);
   }

   TH1D *h2 = new TH1D("=1D-h2", "h2-Title", numberOfBins, minRange, maxRange);
   *h2 = *h1;

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats));
   delete h1;
}

TEST(StressHistorgram, TestAssign2D)
{
   // Tests the operator=() method for 2D Histograms

   TH2D *h1 = new TH2D("=2D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   h1->Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
   }

   TH2D *h2 = new TH2D("=2D-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   *h2 = *h1;

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats));
   delete h1;
}

TEST(StressHistorgram, TestAssign3D)
{
   // Tests the operator=() method for 3D Histograms

   TH3D *h1 = new TH3D("=3D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);

   h1->Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, z, 1.0);
   }

   TH3D *h2 = new TH3D("=3D-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                       numberOfBins + 2, minRange, maxRange);
   *h2 = *h1;

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats));
   delete h1;
}

TEST(StressHistorgram, TestAssignVar1D)
{
   // Tests the operator=() method for 1D Histograms with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   TH1D *h1 = new TH1D("=1D-h1", "h1-Title", numberOfBins, v);

   h1->Sumw2();

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(value, 1.0);
   }

   TH1D *h2 = new TH1D("=1D-h2", "h2-Title", numberOfBins, v);
   *h2 = *h1;

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats));
   delete h1;
}
