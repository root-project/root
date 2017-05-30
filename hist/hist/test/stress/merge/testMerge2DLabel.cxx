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
#include "TClass.h"

#include "gtest/gtest.h"

#include "../StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestMerge2DLabelAll)
{
   // Tests the merge method with fully equally labelled 2D Histograms

   TH2D *h1 = new TH2D("merge2DLabelAll-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange,
                       maxRange);
   TH2D *h2 = new TH2D("merge2DLabelAll-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange,
                       maxRange);
   TH2D *h3 = new TH2D("merge2DLabelAll-h3", "h3-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange,
                       maxRange);
   TH2D *h4 = new TH2D("merge2DLabelAll-h4", "h4-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange,
                       maxRange);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
      h4->Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y, 1.0);
      h4->Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3->Fill(x, y, 1.0);
      h4->Fill(x, y, 1.0);
   }

   for (Int_t i = 1; i <= numberOfBins; ++i) {
      ostringstream name;
      name << (char)((int)'a' + i - 1);
      h1->GetXaxis()->SetBinLabel(i, name.str().c_str());
      h2->GetXaxis()->SetBinLabel(i, name.str().c_str());
      h3->GetXaxis()->SetBinLabel(i, name.str().c_str());
      h4->GetXaxis()->SetBinLabel(i, name.str().c_str());
   }

   TList *list = new TList;
   list->Add(h2);
   list->Add(h3);

   h1->Merge(list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
   delete h1;
   delete h2;
   delete h3;
}

TEST(StressHistorgram, TestMerge2DLabelAllDiff)
{
   // Tests the merge method with fully differently labelled 2D Histograms

   // It does not work properly! Look, the bins with the same labels
   // are different ones and still the tests passes! This is not
   // consistent with TH1::Merge()

   TH2D *h1 = new TH2D("merge2DLabelAllDiff-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2,
                       minRange, maxRange);
   TH2D *h2 = new TH2D("merge2DLabelAllDiff-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2,
                       minRange, maxRange);
   TH2D *h3 = new TH2D("merge2DLabelAllDiff-h3", "h3-Title", numberOfBins, minRange, maxRange, numberOfBins + 2,
                       minRange, maxRange);
   TH2D *h4 = new TH2D("merge2DLabelAllDiff-h4", "h4-Title", numberOfBins, minRange, maxRange, numberOfBins + 2,
                       minRange, maxRange);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
      h4->Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y, 1.0);
      h4->Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3->Fill(x, y, 1.0);
      h4->Fill(x, y, 1.0);
   }

   for (Int_t i = 1; i <= numberOfBins; ++i) {
      ostringstream name;
      name << (char)((int)'a' + i - 1);
      h1->GetXaxis()->SetBinLabel(i, name.str().c_str());
      h1->GetYaxis()->SetBinLabel(i, name.str().c_str());
      name << 1;
      h2->GetXaxis()->SetBinLabel(i, name.str().c_str());
      h2->GetYaxis()->SetBinLabel(i, name.str().c_str());
      name << 2;
      h3->GetXaxis()->SetBinLabel(i, name.str().c_str());
      h3->GetYaxis()->SetBinLabel(i, name.str().c_str());
      name << 3;
      h4->GetXaxis()->SetBinLabel(i, name.str().c_str());
   }

   TList *list = new TList;
   list->Add(h2);
   list->Add(h3);

   h1->Merge(list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
   delete h1;
   delete h2;
   delete h3;
}

TEST(StressHistorgram, TestMerge2DLabelDiff)
{
   // Tests the merge with some different labels method for 2D Histograms

   // It does not work properly! Look, the bins with the same labels
   // are different ones and still the tests passes! This is not
   // consistent with TH1::Merge()

   TH2D *h1 = new TH2D("merge2DLabelDiff-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange,
                       maxRange);
   TH2D *h2 = new TH2D("merge2DLabelDiff-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange,
                       maxRange);
   TH2D *h3 = new TH2D("merge2DLabelDiff-h3", "h3-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange,
                       maxRange);
   TH2D *h4 = new TH2D("merge2DLabelDiff-h4", "h4-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange,
                       maxRange);

   h1->GetXaxis()->SetBinLabel(2, "gamma");
   h2->GetXaxis()->SetBinLabel(6, "beta");
   h3->GetXaxis()->SetBinLabel(4, "alpha");
   h4->GetXaxis()->SetBinLabel(4, "alpha");

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
      h4->Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y, 1.0);
      h4->Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3->Fill(x, y, 1.0);
      h4->Fill(x, y, 1.0);
   }

   TList *list = new TList;
   list->Add(h2);
   list->Add(h3);

   h1->Merge(list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
   delete h1;
   delete h2;
   delete h3;
}

TEST(StressHistorgram, TestMerge2DLabelSame)
{
   // Tests the merge with some equal labels method for 2D Histograms
   // Note by LM (Dec 2010)
   // In reality in 2D histograms the Merge does not support
   // histogram with labels - just merges according to the x-values
   // This test is basically useless

   TH2D *h1 = new TH2D("merge2DLabelSame-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange,
                       maxRange);
   TH2D *h2 = new TH2D("merge2DLabelSame-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange,
                       maxRange);
   TH2D *h3 = new TH2D("merge2DLabelSame-h3", "h3-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange,
                       maxRange);
   TH2D *h4 = new TH2D("merge2DLabelSame-h4", "h4-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange,
                       maxRange);

   h1->GetXaxis()->SetBinLabel(4, "alpha");
   h2->GetXaxis()->SetBinLabel(4, "alpha");
   h3->GetXaxis()->SetBinLabel(4, "alpha");
   h4->GetXaxis()->SetBinLabel(4, "alpha");

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
      h4->Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2->Fill(x, y, 1.0);
      h4->Fill(x, y, 1.0);
   }

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3->Fill(x, y, 1.0);
      h4->Fill(x, y, 1.0);
   }

   TList *list = new TList;
   list->Add(h2);
   list->Add(h3);

   h1->Merge(list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
   delete h1;
   delete h2;
   delete h3;
}
