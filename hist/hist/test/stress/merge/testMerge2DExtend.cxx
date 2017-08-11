// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <cmath>
#include <sstream>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "Riostream.h"
#include "TApplication.h"
#include "TClass.h"
#include "TFile.h"
#include "TRandom2.h"

#include "gtest/gtest.h"

#include "../StressHistogramGlobal.h"

using namespace std;

class Merge2DExtendTest : public ::testing::TestWithParam<UInt_t> {
};

TEST_P(Merge2DExtendTest, TestMerge2DExtend)
{
   TRandom2 r;
   UInt_t extendType = GetParam();
   // Tests the merge method for diferent 1D Histograms
   // when axis can be extended (e.g. for time histograms)

   TH2D h1("merge2D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D h2("merge2D-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);
   TH2D h4("merge2D-h4", "h4-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();
   h4.Sumw2();

   h1.SetCanExtend(extendType);
   h2.SetCanExtend(extendType);
   h4.SetCanExtend(extendType);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(minRange, maxRange);
      Double_t y = r.Uniform(minRange, maxRange);
      h1.Fill(x, y, 1.);
      h4.Fill(x, y, 1.);
   }
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * maxRange, 2.1 * maxRange);
      Double_t y = r.Uniform(0.8 * maxRange, 3. * maxRange);
      h2.Fill(x, y, 1.);
      h4.Fill(x, y, 1.);
   }

   TList list;
   list.Add(&h2);

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}

INSTANTIATE_TEST_CASE_P(StressHistogram, Merge2DExtendTest, ::testing::Values(TH1::kAllAxes, TH1::kXaxis, TH1::kYaxis));
