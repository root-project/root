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

#include "TRandom2.h"

#include "gtest/gtest.h"

#include "../StressHistogramGlobal.h"

using namespace std;

TEST(StressHistogram, TestAddVarProf1)
{
   TH1::SetDefaultSumw2();
   // Tests the first Add method for 1D Profiles with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   Double_t c1 = r.Rndm();
   Double_t c2 = r.Rndm();

   TProfile p1("t1D1-p1", "p1-Title", numberOfBins, v);
   TProfile p2("t1D1-p2", "p2-Title", numberOfBins, v);
   TProfile p3("t1D1-p3", "p3=c1*p1+c2*p2", numberOfBins, v);

   FillProfiles(p1, p3, 1.0, c1);
   FillProfiles(p2, p3, 1.0, c2);

   TProfile p4("t1D1-p4", "p4=c1*p1+p2*c2", numberOfBins, v);
   p4.Add(&p1, &p2, c1, c2);

   EXPECT_TRUE(HistogramsEquals(p3, p4, cmpOptStats, 1E-13));
}

TEST(StressHistogram, TestAddVarProf2)
{
   // Tests the second Add method for 1D Profiles with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   Double_t c2 = r.Rndm();

   TProfile p5("t1D2-p5", "p5=   p6+c2*p7", numberOfBins, v);
   TProfile p6("t1D2-p6", "p6-Title", numberOfBins, v);
   TProfile p7("t1D2-p7", "p7-Title", numberOfBins, v);

   p5.Sumw2();
   p6.Sumw2();
   p7.Sumw2();

   FillProfiles(p6, p5, 1.0, 1.0);
   FillProfiles(p7, p5, 1.0, c2);

   p6.Add(&p7, c2);

   EXPECT_TRUE(HistogramsEquals(p5, p6, cmpOptStats, 1E-13));
}
