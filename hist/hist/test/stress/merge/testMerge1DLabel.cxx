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

TEST(StressHistorgram, TestMerge1DLabelAll)
{
   // Tests the merge method with fully equally labelled 1D Histograms

   TH1D h1("h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D h2("h2", "h2-Title", numberOfBins, minRange, maxRange);
   TH1D h3("h3", "h3-Title", numberOfBins, minRange, maxRange);
   TH1D h4("h4", "h4-Title", numberOfBins, minRange, maxRange);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, 1.0);
      h4.Fill(x, 1.0);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, 1.0);
      h4.Fill(x, 1.0);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3.Fill(x, 1.0);
      h4.Fill(x, 1.0);
   }

   for (Int_t i = 1; i <= numberOfBins; ++i) {
      ostringstream name;
      name << (char)((int)'a' + i - 1);
      h1.GetXaxis()->SetBinLabel(i, name.str().c_str());
      h2.GetXaxis()->SetBinLabel(i, name.str().c_str());
      h3.GetXaxis()->SetBinLabel(i, name.str().c_str());
      h4.GetXaxis()->SetBinLabel(i, name.str().c_str());
   }

   TList list;
   list.Add(&h2);
   list.Add(&h3);

   // test to re-order some histos
   h1.LabelsOption("a");
   h2.LabelsOption("<");
   h3.LabelsOption(">");

   unique_ptr<TH1> h0((TH1 *)h1.Clone("h1clone"));

   h1.Merge(&list);

   h4.LabelsOption("a");

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptNone, 1E-10));
}

TEST(StressHistorgram, TestMerge1DLabelAllDiff)
{
   // LM: Dec 2010 : rmeake this test as
   // a test of histogram with some different labels not all filled

   TH1D h1("merge1DLabelAllDiff-h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D h2("merge1DLabelAllDiff-h2", "h2-Title", numberOfBins, minRange, maxRange);
   TH1D h3("merge1DLabelAllDiff-h3", "h3-Title", numberOfBins, minRange, maxRange);
   TH1D h4("merge1DLabelAllDiff-h4", "h4-Title", numberOfBins, minRange, maxRange);

   Int_t ibin = r.Integer(numberOfBins) + 1;
   h1.GetXaxis()->SetBinLabel(ibin, "aaa");
   ibin = r.Integer(numberOfBins) + 1;
   h2.GetXaxis()->SetBinLabel(ibin, "bbb");
   ibin = r.Integer(numberOfBins) + 1;
   h3.GetXaxis()->SetBinLabel(ibin, "ccc");

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(x, 1.0);
      h4.Fill(x, 1.0);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h2.Fill(x, 1.0);
      h4.Fill(x, 1.0);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h3.Fill(x, 1.0);
      h4.Fill(x, 1.0);
   }

   TList list;
   list.Add(&h2);
   list.Add(&h3);

   Int_t prevErrorLevel = gErrorIgnoreLevel;
   // // to suppress a Warning message
   //   Warning in <TH1D::Merge>: Histogram FirstClone contains non-empty bins without labels -
   //  falling back to bin numbering mode
   gErrorIgnoreLevel = kError;

   h1.Merge(&list);
   gErrorIgnoreLevel = prevErrorLevel;

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}

TEST(StressHistorgram, TestMerge1DLabelDiff)
{
   // Tests the merge with some different labels  for 1D Histograms

   TH1D h1("merge1DLabelDiff-h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D h2("merge1DLabelDiff-h2", "h2-Title", numberOfBins, minRange, maxRange);
   TH1D h3("merge1DLabelDiff-h3", "h3-Title", numberOfBins, minRange, maxRange);
   TH1D h4("merge1DLabelDiff-h4", "h4-Title", numberOfBins, minRange, maxRange);

   // This test fails, as expected! That is why it is not run in the tests suite.
   const char labels[10][5] = {"aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh", "iii", "lll"};

   // choose random same labels (nbins -2)
   std::vector<TString> labels2(8);
   for (int i = 0; i < 8; ++i) labels2[i] = labels[r.Integer(10)];

   for (Int_t e = 0; e < nEvents; ++e) {
      int i = r.Integer(8);
      if (i < 8) {
         h1.Fill(labels2[i], 1.0);
         h4.Fill(labels2[i], 1.0);
      } else {
         // add one empty label
         h1.Fill("", 1.0);
         h4.Fill("", 1.0);
      }
   }

   for (int i = 0; i < 8; ++i) labels2[i] = labels[r.Integer(10)];
   for (Int_t e = 0; e < nEvents; ++e) {
      Int_t i = r.Integer(8);
      h2.Fill(labels2[i], 1.0);
      h4.Fill(labels2[i], 1.0);
   }

   for (int i = 0; i < 8; ++i) labels2[i] = labels[r.Integer(10)];
   for (Int_t e = 0; e < nEvents; ++e) {
      Int_t i = r.Integer(8);
      h3.Fill(labels2[i], 1.0);
      h4.Fill(labels2[i], 1.0);
   }

   // test ordering label for one histo
   h2.LabelsOption("a");
   h3.LabelsOption(">");

   TList list;
   list.Add(&h2);
   list.Add(&h3);

   h1.Merge(&list);

   // need to order the histo to compare them
   h1.LabelsOption("a");
   h4.LabelsOption("a");

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}

TEST(StressHistorgram, TestMerge1DLabelSame)
{
   // Tests the merge with some equal labels method for 1D Histograms
   // number of labels used = number of bins

   TH1D h1("h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D h2("h2", "h2-Title", numberOfBins, minRange, maxRange);
   TH1D h3("h3", "h3-Title", numberOfBins, minRange, maxRange);
   TH1D h4("h4", "h4-Title", numberOfBins, minRange, maxRange);

   const char labels[10][5] = {"aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh", "iii", "lll"};

   for (Int_t i = 0; i < numberOfBins; ++i) {
      h1.GetXaxis()->SetBinLabel(i + 1, labels[i]);
      h2.GetXaxis()->SetBinLabel(i + 1, labels[i]);
      h3.GetXaxis()->SetBinLabel(i + 1, labels[i]);
      h4.GetXaxis()->SetBinLabel(i + 1, labels[i]);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Int_t i = r.Integer(11);
      if (i < 10) {
         h1.Fill(labels[i], 1.0);
         h4.Fill(labels[i], 1.0);
      } else {
         // add one empty label
         // should be added in underflow bin
         // to test merge of underflows
         h1.Fill("", 1.0);
         h4.Fill("", 1.0);
      }
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Int_t i = r.Integer(10);
      h2.Fill(labels[i], 1.0);
      h4.Fill(labels[i], 1.0);
   }

   for (Int_t e = 0; e < nEvents; ++e) {
      Int_t i = r.Integer(10);
      h3.Fill(labels[i], 1.0);
      h4.Fill(labels[i], 1.0);
   }

   TList list;
   list.Add(&h2);
   list.Add(&h3);

   h1.SetCanExtend(TH1::kAllAxes);

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}
