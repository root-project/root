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

TEST(StressHistogram, TestMerge2D)
{
   TRandom2 r;
   // Tests the merge method for 2D Histograms

   TH2D h1("merge2D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D h2("merge2D-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D h3("merge2D-h3", "h3-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D h4("merge2D-h4", "h4-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, y, 1.0);
      h4.Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, y, 1.0);
      h4.Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3.Fill(x, y, 1.0);
      h4.Fill(x, y, 1.0);
   }

   TList list;
   list.Add(&h2);
   list.Add(&h3);

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}
