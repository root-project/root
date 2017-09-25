// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include "StressHistogramGlobal.h"

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "TFile.h"
#include "TRandom2.h"

#include "TROOT.h"

#include <sstream>
#include "gtest/gtest.h"

using namespace std;

TEST(StressHistogram, TestWriteReadProfile1D)
{
   TRandom2 r;
   // Tests the write and read methods for 1D Profiles

   TProfile p1("wr1D-p1", "p1-Title", numberOfBins, minRange, maxRange);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, 1.0);
   }

   TFile f(("tmpHist" + std::to_string(getpid()) + ".root").c_str(), "RECREATE");
   p1.Write();
   f.Close();

   TFile f2(("tmpHist" + std::to_string(getpid()) + ".root").c_str());
   unique_ptr<TProfile> p2(static_cast<TProfile *>(f2.Get("wr1D-p1")));

   EXPECT_TRUE(HistogramsEquals(p1, *p2.get(), cmpOptStats));
}

TEST(StressHistogram, TestWriteReadProfile2D)
{
   TRandom2 r;
   // Tests the write and read methods for 2D Profiles

   TProfile2D p1("wr2D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins + 2, minRange, maxRange);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, z, 1.0);
   }

   TFile f(("tmpHist" + std::to_string(getpid()) + ".root").c_str(), "RECREATE");
   p1.Write();
   f.Close();

   TFile f2(("tmpHist" + std::to_string(getpid()) + ".root").c_str());
   unique_ptr<TProfile2D> p2(static_cast<TProfile2D *>(f2.Get("wr2D-p1")));

   EXPECT_TRUE(HistogramsEquals(p1, *p2.get(), cmpOptStats));
}

TEST(StressHistogram, TestWriteReadProfile3D)
{
   TRandom2 r;
   // Tests the write and read methods for 3D Profile

   TProfile3D p1("wr3D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins + 1, minRange, maxRange,
                 numberOfBins + 2, minRange, maxRange);

   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, z, t, 1.0);
   }

   TFile f(("tmpHist" + std::to_string(getpid()) + ".root").c_str(), "RECREATE");
   p1.Write();
   f.Close();

   TFile f2(("tmpHist" + std::to_string(getpid()) + ".root").c_str());
   unique_ptr<TProfile3D> p2(static_cast<TProfile3D *>(f2.Get("wr3D-p1")));

   // In this particular case the statistics are not checked. The
   // Chi2Test is not properly implemented for the TProfile3D
   // class. If the cmpOptStats flag is set, then there will be a
   // crash.
   EXPECT_TRUE(HistogramsEquals(p1, *p2.get()));
}

TEST(StressHistogram, TestWriteReadProfileVar1D)
{
   TRandom2 r;
   // Tests the write and read methods for 1D Profiles with variable bin size

   Double_t v[numberOfBins + 1];
   FillVariableRange(v);

   TProfile p1("wr1D-p1", "p1-Title", numberOfBins, v);

   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      p1.Fill(x, y, 1.0);
   }

   TFile f(("tmpHist" + std::to_string(getpid()) + ".root").c_str(), "RECREATE");
   p1.Write();
   f.Close();

   TFile f2(("tmpHist" + std::to_string(getpid()) + ".root").c_str());
   unique_ptr<TProfile> p2(static_cast<TProfile *>(f2.Get("wr1D-p1")));

   EXPECT_TRUE(HistogramsEquals(p1, *p2.get(), cmpOptStats));
}
