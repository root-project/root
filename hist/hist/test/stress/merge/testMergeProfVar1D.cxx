// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TApplication.h"
#include "TFile.h"

#include "gtest/gtest.h"

#include "../StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestMergeProfVar1D)
{
   // Tests the merge method for 1D Profiles with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   TProfile *p1 = new TProfile("p1", "p1-Title", numberOfBins, v);
   TProfile *p2 = new TProfile("p2", "p2-Title", numberOfBins, v);
   TProfile *p3 = new TProfile("p3", "p3-Title", numberOfBins, v);
   TProfile *p4 = new TProfile("p4", "p4-Title", numberOfBins, v);

   FillProfiles(p1, p4);
   FillProfiles(p2, p4);
   FillProfiles(p3, p4);

   TList *list = new TList;
   list->Add(p2);
   list->Add(p3);

   p1->Merge(list);

   EXPECT_TRUE(HistogramsEquals(p1, p4, cmpOptStats, 1E-10));
   delete p1;
   delete p2;
   delete p3;
}
