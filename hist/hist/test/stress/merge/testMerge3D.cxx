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

#include "TApplication.h"
#include "TRandom2.h"
#include "TFile.h"

#include "gtest/gtest.h"

#include "../StressHistogramGlobal.h"

using namespace std;

TEST(StressHistogram, TestMerge3D)
{
   TRandom2 r;
   // Tests the merge method for 3D Histograms

   TH3D h1("merge3D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h2("merge3D-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h3("merge3D-h3", "h3-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h4("merge3D-h4", "h4-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, z, 1.0);
      h4.Fill(x, y, z, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, z, 1.0);
      h4.Fill(x, y, z, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3.Fill(x, y, z, 1.0);
      h4.Fill(x, y, z, 1.0);
   }

   TList list;
   list.Add(&h2);
   list.Add(&h3);

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}

TEST(StressHistogram, TestMerge3DExtend)
{
   TRandom2 r;
   UInt_t extendType = TH1::kAllAxes;
   // Tests the merge method for diferent 1D Histograms
   // when axis can be extended (e.g. for time histograms)

   TH3D h1("merge3D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h2("merge3D-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h4("merge3D-h4", "h4-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);

   h1.SetCanExtend(extendType);
   h2.SetCanExtend(extendType);
   h4.SetCanExtend(extendType);

   for (Int_t e = 0; e < 10 * nEvents; ++e) {
      Double_t x = r.Uniform(minRange, maxRange);
      Double_t y = r.Uniform(minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(minRange, 1.1 * maxRange);
      h1.Fill(x, y, z, 1.);
      h4.Fill(x, y, z, 1.);
   }
   for (Int_t e = 0; e < 10 * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * maxRange, 2.1 * maxRange);
      // Double_t x = r.Uniform(minRange,  maxRange);
      Double_t y = r.Uniform(minRange, 3 * maxRange);
      Double_t z = r.Uniform(0.8 * minRange, 4.1 * maxRange);
      h2.Fill(x, y, z, 1.);
      h4.Fill(x, y, z, 1.);
   }

   TList list;
   list.Add(&h2);

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}
