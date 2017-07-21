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

#include "../StressHistogramGlobal.h"

using namespace std;

TEST(StressHistogram, TestAddVar1)
{
   TRandom2 r(initialRandomSeed);
   // Tests the second Add method for 1D Histograms with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TH1D h1("h1", "h1-Title", numberOfBins, v);
   TH1D h2("h2", "h2-Title", numberOfBins, v);
   TH1D h3("h3", "h3=c1*h1+c2*h2", numberOfBins, v);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();

   FillHistograms(h1, h3, 1.0, c1);
   FillHistograms(h2, h3, 1.0, c2);

   TH1D h4("t1D1-h4", "h4=c1*h1+h2*c2", numberOfBins, v);
   h4.Add(&h1, &h2, c1, c2);

   EXPECT_TRUE(HistogramsEquals(h3, h4, cmpOptStats, 1E-13));
}

TEST(StressHistogram, TestAddVar2)
{
   TRandom2 r(initialRandomSeed);
   // Tests the second Add method for 1D Histograms with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   Double_t c2 = r.Rndm();

   TH1D h5("t1D2-h5", "h5=   h6+c2*h7", numberOfBins, v);
   TH1D h6("t1D2-h6", "h6-Title", numberOfBins, v);
   TH1D h7("t1D2-h7", "h7-Title", numberOfBins, v);

   h5.Sumw2();
   h6.Sumw2();
   h7.Sumw2();

   FillHistograms(h6, h5, 1.0, 1.0);
   FillHistograms(h7, h5, 1.0, c2);

   h6.Add(&h7, c2);

   EXPECT_TRUE(HistogramsEquals(h5, h6, cmpOptStats, 1E-13));
}

TEST(StressHistogram, TestAddVar3)
{
   TRandom2 r(initialRandomSeed);
   // Tests the first add method to do scale of 1D Histograms with variable bin width

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   Double_t c1 = r.Rndm();

   TH1D h1("t1D1-h1", "h1-Title", numberOfBins, v);
   TH1D h2("t1D1-h2", "h2=c1*h1+c2*h2", numberOfBins, v);

   h1.Sumw2();
   h2.Sumw2();

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
      h2.Fill(value, c1 / h1.GetBinWidth(h1.FindBin(value)));
   }

   TH1D h3("t1D1-h3", "h3=c1*h1", numberOfBins, v);
   h3.Add(&h1, &h1, c1, -1);

   // TH1::Add will reset the stats in this case so we need to do for the reference histogram
   h2.ResetStats();

   EXPECT_TRUE(HistogramsEquals(h2, h3, cmpOptStats, 1E-13));
}
