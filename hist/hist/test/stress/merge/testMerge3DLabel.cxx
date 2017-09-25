// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include "../StressHistogramGlobal.h"

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TApplication.h"
#include "TClass.h"
#include "TFile.h"
#include "TRandom2.h"

#include <sstream>
#include "gtest/gtest.h"

using namespace std;

TEST(StressHistogram, TestMerge3DLabelAll)
{
   TRandom2 r;
   // Tests the merge method with fully equally labelled 3D Histograms

   TH3D h1("merge3DLabelAll-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h2("merge3DLabelAll-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h3("merge3DLabelAll-h3", "h3-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h4("merge3DLabelAll-h4", "h4-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
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

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}

TEST(StressHistogram, TestMerge3DLabelAllDiff)
{
   TRandom2 r;
   // Tests the merge method with fully differently labelled 3D Histograms

   // It does not work properly! Look, the bins with the same labels
   // are different ones and still the tests passes! This is not
   // consistent with TH1::Merge()

   TH3D h1("merge3DLabelAllDiff-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h2("merge3DLabelAllDiff-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h3("merge3DLabelAllDiff-h3", "h3-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h4("merge3DLabelAllDiff-h4", "h4-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);

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

   for (Int_t i = 1; i <= numberOfBins; ++i) {
      ostringstream name;
      name << (char)((int)'a' + i - 1);
      h1.GetXaxis()->SetBinLabel(i, name.str().c_str());
      h1.GetYaxis()->SetBinLabel(i, name.str().c_str());
      h1.GetZaxis()->SetBinLabel(i, name.str().c_str());
      name << 1;
      h2.GetXaxis()->SetBinLabel(i, name.str().c_str());
      h2.GetYaxis()->SetBinLabel(i, name.str().c_str());
      h2.GetZaxis()->SetBinLabel(i, name.str().c_str());
      name << 2;
      h3.GetXaxis()->SetBinLabel(i, name.str().c_str());
      h3.GetYaxis()->SetBinLabel(i, name.str().c_str());
      h3.GetZaxis()->SetBinLabel(i, name.str().c_str());
      name << 3;
      h4.GetXaxis()->SetBinLabel(i, name.str().c_str());
      h4.GetYaxis()->SetBinLabel(i, name.str().c_str());
      h4.GetZaxis()->SetBinLabel(i, name.str().c_str());
   }

   TList list;
   list.Add(&h2);
   list.Add(&h3);

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}

TEST(StressHistogram, TestMerge3DLabelDiff)
{
   TRandom2 r;
   // Tests the merge with some different labels method for 3D Histograms

   // It does not work properly! Look, the bins with the same labels
   // are different ones and still the tests passes! This is not
   // consistent with TH1::Merge()

   TH3D h1("merge3DLabelDiff-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h2("merge3DLabelDiff-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h3("merge3DLabelDiff-h3", "h3-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h4("merge3DLabelDiff-h4", "h4-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);

   h1.GetXaxis()->SetBinLabel(2, "gamma");
   h2.GetXaxis()->SetBinLabel(6, "beta");
   h3.GetXaxis()->SetBinLabel(4, "alpha");
   h4.GetXaxis()->SetBinLabel(4, "alpha");

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

TEST(StressHistogram, TestMerge3DLabelSame)
{
   TRandom2 r;
   // Tests the merge with some equal labels method for 3D Histograms

   TH3D h1("merge3DLabelSame-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h2("merge3DLabelSame-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h3("merge3DLabelSame-h3", "h3-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);
   TH3D h4("merge3DLabelSame-h4", "h4-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
           numberOfBins + 2, minRange, maxRange);

   h1.GetXaxis()->SetBinLabel(4, "alpha");
   h2.GetXaxis()->SetBinLabel(4, "alpha");
   h3.GetXaxis()->SetBinLabel(4, "alpha");
   h4.GetXaxis()->SetBinLabel(4, "alpha");

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
