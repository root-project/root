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
#include "TFile.h"

#include "TROOT.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestWriteReadProfile1D)
{
   // Tests the write and read methods for 1D Profiles

   TProfile *p1 = new TProfile("wr1D-p1", "p1-Title", numberOfBins, minRange, maxRange);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   p1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TProfile *p2 = static_cast<TProfile *>(f2.Get("wr1D-p1"));

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
   delete p1;
}

TEST(StressHistorgram, TestWriteReadProfile2D)
{
   // Tests the write and read methods for 2D Profiles

   TProfile2D *p1 =
      new TProfile2D("wr2D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   p1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TProfile2D *p2 = static_cast<TProfile2D *>(f2.Get("wr2D-p1"));

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
   delete p1;
}

TEST(StressHistorgram, TestWriteReadProfile3D)
{
   // Tests the write and read methods for 3D Profile

   TProfile3D *p1 = new TProfile3D("wr3D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange,
                                   maxRange, numberOfBins + 2, minRange, maxRange);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, z, t, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   p1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TProfile3D *p2 = static_cast<TProfile3D *>(f2.Get("wr3D-p1"));

   // In this particular case the statistics are not checked. The
   // Chi2Test is not properly implemented for the TProfile3D
   // class. If the cmpOptStats flag is set, then there will be a
   // crash.
   EXPECT_TRUE(HistogramsEquals(p1, p2));
   delete p1;
}

TEST(StressHistorgram, TestWriteReadProfileVar1D)
{
   // Tests the write and read methods for 1D Profiles with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   TProfile *p1 = new TProfile("wr1D-p1", "p1-Title", numberOfBins, v);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1->Fill(x, y, 1.0);
   }

   TFile f("tmpHist.root", "RECREATE");
   p1->Write();
   f.Close();

   TFile f2("tmpHist.root");
   TProfile *p2 = static_cast<TProfile *>(f2.Get("wr1D-p1"));

   EXPECT_TRUE(HistogramsEquals(p1, p2, cmpOptStats));
   delete p1;
}
