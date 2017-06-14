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
#include "TFile.h"

#include "gtest/gtest.h"

#include "../StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestMerge1D)
{
   // Tests the merge method for 1D Histograms
   // simple merge with histogram with same limits

   TH1D *h1 = new TH1D("h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D *h2 = new TH1D("h2", "h2-Title", numberOfBins, minRange, maxRange);
   TH1D *h3 = new TH1D("h3", "h3-Title", numberOfBins, minRange, maxRange);
   TH1D *h4 = new TH1D("h4", "h4-Title", numberOfBins, minRange, maxRange);

   h1->Sumw2();
   h2->Sumw2();
   h3->Sumw2();

   FillHistograms(h1, h4);
   FillHistograms(h2, h4);
   FillHistograms(h3, h4);

   TList *list = new TList;
   list->Add(h2);
   list->Add(h3);

   h1->Merge(list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
   delete h1;
   delete h2;
   delete h3;
   delete h4;
}

TEST(StressHistorgram, TestMerge1DWithBuffer)
{
   bool allNoLimits = true;
   // Tests the merge method for different 1D Histograms
   // where different axis are used, BUT the largest bin width must be
   // a multiple of the smallest bin width

   double x1 = 1;
   double x2 = 0;
   if (!allNoLimits) {
      // case when one of the histogram has limits (mix mode)
      x1 = minRange;
      x2 = maxRange;
   }

   TH1D *h0 = new TH1D("h0", "h1-Title", numberOfBins, 1, 0);
   TH1D *h1 = new TH1D("h1", "h1-Title", numberOfBins, x1, x2);
   TH1D *h2 = new TH1D("h2", "h2-Title", 1, 1, 0);
   TH1D *h3 = new TH1D("h3", "h3-Title", 1, 1, 0);
   TH1D *h4 = new TH1D("h4", "h4-Title", numberOfBins, x1, x2);

   h0->Sumw2();
   h1->Sumw2();
   h2->Sumw2();
   h4->Sumw2();
   h1->SetBuffer(nEvents * 10);
   h2->SetBuffer(nEvents * 10);
   h3->SetBuffer(nEvents * 10);
   h4->SetBuffer(nEvents * 10);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(minRange, maxRange);
      Double_t weight = std::exp(r.Gaus(0, 1));
      h1->Fill(value, weight);
      h4->Fill(value, weight);
   }
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform((maxRange - minRange) / 2, maxRange);
      Double_t weight = std::exp(r.Gaus(0, 1));
      h2->Fill(value, weight);
      h4->Fill(value, weight);
   }
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(minRange, (maxRange - minRange) / 2);
      Double_t weight = std::exp(r.Gaus(0, 1));
      h3->Fill(value, weight);
      h4->Fill(value, weight);
   }

   TList *list = new TList;
   list->Add(h1);
   list->Add(h2);
   list->Add(h3);

   h0->Merge(list);

   // flush buffer before comparing
   h0->BufferEmpty();
   h4->BufferEmpty();

   EXPECT_TRUE(HistogramsEquals(h0, h4, cmpOptStats, 1E-10));
   delete h0;
   delete h1;
   delete h2;
   delete h3;
   delete h4;
}
