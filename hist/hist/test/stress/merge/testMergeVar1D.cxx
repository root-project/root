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

TEST(StressHistogram, TestMergeVar1D)
{
   // Tests the merge method for 1D Histograms with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   TH1D h1("h1", "h1-Title", numberOfBins, v);
   TH1D h2("h2", "h2-Title", numberOfBins, v);
   TH1D h3("h3", "h3-Title", numberOfBins, v);
   TH1D h4("h4", "h4-Title", numberOfBins, v);

   h1.Sumw2();
   h2.Sumw2();
   h3.Sumw2();

   FillHistograms(h1, h4);
   FillHistograms(h2, h4);
   FillHistograms(h3, h4);

   TList list;
   list.Add(&h2);
   list.Add(&h3);

   h1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(h1, h4, cmpOptStats, 1E-10));
}
