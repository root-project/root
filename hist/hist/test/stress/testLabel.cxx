// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>
#include <cmath>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "Riostream.h"
#include "TRandom2.h"
#include "TFile.h"
#include "TClass.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

TEST(StressHistogram, TestLabel)
{
   TRandom2 r(initialRandomSeed);
   // Tests labelling a 1D Histogram

   TH1D h1("lD1-h1", "h1-Title", 2 * numberOfBins, minRange, maxRange);
   // build histo with extra  labels to tets the deflate option
   int extraBins = 20;
   TH1D h2("lD1-h2", "h2-Title", 2 * numberOfBins + 20, minRange, maxRange + extraBins * h1.GetXaxis()->GetBinWidth(1));

   // set labels
   std::vector<std::string> vLabels;
   for (Int_t bin = 1; bin <= h1.GetNbinsX(); ++bin) {
      ostringstream label;
      label << bin;
      vLabels.push_back(label.str());
      h2.GetXaxis()->SetBinLabel(bin, label.str().c_str());
   }
   // sort labels in alphabetic order
   std::sort(vLabels.begin(), vLabels.end());

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(minRange, maxRange);
      Int_t bin = h1.GetXaxis()->FindBin(value);
      h1.Fill(h1.GetXaxis()->GetBinCenter(bin), 1.0);

      h2.Fill(vLabels[bin - 1].c_str(), 1.0);
   }

   h2.LabelsOption("a");
   h2.LabelsDeflate();

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats, 1E-13));
}

TEST(StressHistogram, TestLabel2DX)
{
   TRandom2 r(initialRandomSeed);
   // Tests labelling a 1D Histogram

   TH2D h1("lD2-h1", "h1-Title", 2 * numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);
   // build histo with extra  labels to tets the deflate option
   TH2D h2("lD2-h2", "h2-Title", 2 * numberOfBins + 20, minRange, maxRange + 20 * h1.GetXaxis()->GetBinWidth(1),
           numberOfBins, minRange, maxRange);

   // set labels
   std::vector<std::string> vLabels;
   for (Int_t bin = 1; bin <= h1.GetNbinsX(); ++bin) {
      ostringstream label;
      label << bin;
      vLabels.push_back(label.str());
      h2.GetXaxis()->SetBinLabel(bin, label.str().c_str());
   }
   // sort labels in alphabetic order
   std::sort(vLabels.begin(), vLabels.end());

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t xvalue = r.Uniform(minRange, maxRange);
      Double_t yvalue = r.Uniform(minRange, maxRange);
      Int_t binx = h1.GetXaxis()->FindBin(xvalue);
      Int_t biny = h1.GetYaxis()->FindBin(yvalue);
      h1.Fill(h1.GetXaxis()->GetBinCenter(binx), h1.GetYaxis()->GetBinCenter(biny), 1.0);

      h2.Fill(vLabels[binx - 1].c_str(), h1.GetYaxis()->GetBinCenter(biny), 1.0);
   }

   h2.LabelsOption("a");

   h2.LabelsDeflate();

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats, 1E-13));
}

TEST(StressHistogram, TestLabel2DY)
{
   TRandom2 r(initialRandomSeed);
   // Tests labelling a 1D Histogram

   TH2D h1("lD2-h1", "h1-Title", numberOfBins, minRange, maxRange, 2 * numberOfBins, minRange, maxRange);
   // build histo with extra  labels to tets the deflate option
   TH2D h2("lD2-h2", "h2-Title", numberOfBins, minRange, maxRange, 2 * numberOfBins + 20, minRange,
           maxRange + 20 * h1.GetYaxis()->GetBinWidth(1));

   // set labels
   std::vector<std::string> vLabels;
   for (Int_t bin = 1; bin <= h1.GetNbinsY(); ++bin) {
      ostringstream label;
      label << bin;
      vLabels.push_back(label.str());
      h2.GetYaxis()->SetBinLabel(bin, label.str().c_str());
   }
   // sort labels in alphabetic order
   std::sort(vLabels.begin(), vLabels.end());

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t xvalue = r.Uniform(minRange, maxRange);
      Double_t yvalue = r.Uniform(minRange, maxRange);
      Int_t binx = h1.GetXaxis()->FindBin(xvalue);
      Int_t biny = h1.GetYaxis()->FindBin(yvalue);
      h1.Fill(h1.GetXaxis()->GetBinCenter(binx), h1.GetYaxis()->GetBinCenter(biny), 1.0);

      h2.Fill(h1.GetXaxis()->GetBinCenter(binx), vLabels[biny - 1].c_str(), 1.0);
   }

   h2.GetYaxis()->LabelsOption("a");

   h2.LabelsDeflate("Y");

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats, 1E-13));
}

TEST(StressHistogram, TestLabelsInflateProf1D)
{
   TRandom2 r(initialRandomSeed);
   // Tests labelling a 1D Profile

   Int_t numberOfInflates = 4;
   Int_t numberOfFills = numberOfBins;
   Double_t maxRangeInflate = maxRange;
   for (Int_t i = 0; i < numberOfInflates; ++i) {
      numberOfFills *= 2;
      maxRangeInflate = 2 * maxRangeInflate - 1;
   }

   TProfile p1("tLI1D-p1", "p1-Title", numberOfBins, minRange, maxRange);
   TProfile p2("tLI1D-p2", "p2-Title", numberOfFills, minRange, maxRangeInflate);

   p1.GetXaxis()->SetTimeDisplay(1);

   for (Int_t e = 0; e < numberOfFills; ++e) {
      Double_t x = e;
      Double_t y = sin(x / 10);

      p1.SetBinContent(int(x + 0.5) + 1, y);
      p1.SetBinEntries(int(x + 0.5) + 1, 10.0);

      p2.SetBinContent(int(x + 0.5) + 1, y);
      p2.SetBinEntries(int(x + 0.5) + 1, 10.0);
   }

   EXPECT_TRUE(HistogramsEquals(p1, p2));
}
