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

TEST(StressHistogram, TestMerge1DExtend)
{
   TRandom2 r;
   // Tests the merge method for diferent 1D Histograms
   // when axis can rebin (e.g. for time histograms)

   TH1D h1("h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D h2("h2", "h2-Title", numberOfBins, minRange, maxRange);
   TH1D h4("h4", "h4-Title", numberOfBins, minRange, maxRange);

   h1.Sumw2();
   h2.Sumw2();
   h4.Sumw2();
   h1.SetCanExtend(TH1::kAllAxes);
   h2.SetCanExtend(TH1::kAllAxes);
   h4.SetCanExtend(TH1::kAllAxes);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(minRange, maxRange);
      h1.Fill(value, 1.);
      h4.Fill(value, 1.);
   }
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * maxRange, 2.1 * maxRange);
      h2.Fill(value, 1.);
      h4.Fill(value, 1.);
   }

   TList list;
   list.Add(&h2);

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}

TEST(StressHistogram, TestMerge1DExtendProf)
{
   TRandom2 r;
   // Tests the merge method for diferent 1D Histograms
   // when axis can rebin (e.g. for time histograms)

   TProfile h1("p1", "h1-Title", numberOfBins, minRange, maxRange);
   TProfile h2("p2", "h2-Title", numberOfBins, minRange, maxRange);
   TProfile h4("p4", "h4-Title", numberOfBins, minRange, maxRange);

   h1.SetCanExtend(TH1::kAllAxes);
   h2.SetCanExtend(TH1::kAllAxes);
   h4.SetCanExtend(TH1::kAllAxes);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(minRange, maxRange);
      double t = r.Gaus(std::sin(value), 0.5);
      h1.Fill(value, t);
      h4.Fill(value, t);
   }
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * maxRange, 2.1 * maxRange);
      double t = r.Gaus(std::sin(value), 0.5);
      h2.Fill(value, t);
      h4.Fill(value, t);
   }

   TList list;
   list.Add(&h2);

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}
