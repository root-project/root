// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include "../StressHistogramGlobal.h"

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TApplication.h"
#include "TFile.h"

#include <sstream>
#include "gtest/gtest.h"

using namespace std;

TEST(StressHistogram, TestMergeProf1D)
{
   // Tests the merge method for 1D Profiles

   TProfile p1("p1", "p1-Title", numberOfBins, minRange, maxRange);
   TProfile p2("p2", "p2-Title", numberOfBins, minRange, maxRange);
   TProfile p3("p3", "p3-Title", numberOfBins, minRange, maxRange);
   TProfile p4("p4", "p4-Title", numberOfBins, minRange, maxRange);

   FillProfiles(p1, p4);
   FillProfiles(p2, p4);
   FillProfiles(p3, p4);

   TList list;
   list.Add(&p2);
   list.Add(&p3);

   p1.Merge(&list);

   EXPECT_TRUE(HistogramsEquals(p1, p4, cmpOptStats, 1E-10));
}
