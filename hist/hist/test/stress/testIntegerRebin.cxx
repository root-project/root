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

#include "StressHistogramGlobal.h"

using namespace std;

TEST(StressHistorgram, TestIntegerRebin)
{
   // Tests rebin method with an integer as input for 1D Histogram

   const int rebin = TMath::Nint(r.Uniform(minRebin, maxRebin));
   UInt_t seed = r.GetSeed();
   TH1D h1("h1", "Original Histogram", TMath::Nint(r.Uniform(1, 5)) * rebin, minRange, maxRange);
   r.SetSeed(seed);
   h1.Sumw2();
   for (Int_t i = 0; i < nEvents; ++i) h1.Fill(r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(0, 10));

   unique_ptr<TH1D> h2(static_cast<TH1D *>(h1.Rebin(rebin, "testIntegerRebin")));

   TH1D h3("testIntegerRebin2", "testIntegerRebin2", h1.GetNbinsX() / rebin, minRange, maxRange);
   r.SetSeed(seed);
   h3.Sumw2();
   for (Int_t i = 0; i < nEvents; ++i) h3.Fill(r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(0, 10));

   EXPECT_TRUE(HistogramsEquals(*h2.get(), h3, cmpOptStats));
}

TEST(StressHistorgram, TestIntegerRebinNoName)
{
   // Tests rebin method with an integer as input and without name for 1D Histogram

   const int rebin = TMath::Nint(r.Uniform(minRebin, maxRebin));
   UInt_t seed = r.GetSeed();
   TH1D h1("h2", "Original Histogram", TMath::Nint(r.Uniform(1, 5)) * rebin, minRange, maxRange);
   r.SetSeed(seed);
   for (Int_t i = 0; i < nEvents; ++i) h1.Fill(r.Uniform(minRange * .9, maxRange * 1.1));

   unique_ptr<TH1D> h2(dynamic_cast<TH1D *>(h1.Clone()));
   h2->Rebin(rebin);

   TH1D h3("testIntegerRebinNoName", "testIntegerRebinNoName", int(h1.GetNbinsX() / rebin + 0.1), minRange,
                       maxRange);
   r.SetSeed(seed);
   for (Int_t i = 0; i < nEvents; ++i) h3.Fill(r.Uniform(minRange * .9, maxRange * 1.1));

   EXPECT_TRUE(HistogramsEquals(*h2.get(), h3, cmpOptStats));
}

TEST(StressHistorgram, TestIntegerRebinNoNameProfile)
{
   // Tests rebin method with an integer as input and without name for 1D Profile

   const int rebin = TMath::Nint(r.Uniform(minRebin, maxRebin));
   TProfile p1("p1", "p1-Title", TMath::Nint(r.Uniform(1, 5)) * rebin, minRange, maxRange);
   TProfile p3("testIntRebNNProf", "testIntRebNNProf", int(p1.GetNbinsX() / rebin + 0.1), minRange, maxRange);

   for (Int_t i = 0; i < nEvents; ++i) {
      Double_t x = r.Uniform(minRange * .9, maxRange * 1.1);
      Double_t y = r.Uniform(minRange * .9, maxRange * 1.1);
      p1.Fill(x, y);
      p3.Fill(x, y);
   }

   unique_ptr<TProfile> p2(dynamic_cast<TProfile *>(p1.Clone()));
   p2->Rebin(rebin);
   EXPECT_TRUE(HistogramsEquals(*p2.get(), p3, cmpOptStats));
}

TEST(StressHistorgram, TestIntegerRebinProfile)
{
   // Tests rebin method with an integer as input for 1D Profile

   const int rebin = TMath::Nint(r.Uniform(minRebin, maxRebin));
   TProfile p1("p1", "p1-Title", TMath::Nint(r.Uniform(1, 5)) * rebin, minRange, maxRange);
   TProfile p3("testIntRebProf", "testIntRebProf", p1.GetNbinsX() / rebin, minRange, maxRange);

   for (Int_t i = 0; i < nEvents; ++i) {
      Double_t x = r.Uniform(minRange * .9, maxRange * 1.1);
      Double_t y = r.Uniform(minRange * .9, maxRange * 1.1);
      p1.Fill(x, y);
      p3.Fill(x, y);
   }

   unique_ptr<TProfile> p2(static_cast<TProfile *>(p1.Rebin(rebin, "testIntegerRebin")));

   EXPECT_TRUE(HistogramsEquals(*p2.get(), p3, cmpOptStats));
}
