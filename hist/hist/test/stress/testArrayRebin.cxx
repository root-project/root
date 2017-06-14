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

TEST(StressHistorgram, TestArrayRebin)
{
   TH1::StatOverflows(kTRUE);

   // Tests rebin method with an array as input for 1D Histogram

   const int rebin = TMath::Nint(r.Uniform(minRebin, maxRebin)) + 1;
   UInt_t seed = r.GetSeed();
   TH1D h1("h3", "Original Histogram", TMath::Nint(r.Uniform(1, 5)) * rebin * 2, minRange, maxRange);
   r.SetSeed(seed);
   for (Int_t i = 0; i < nEvents; ++i) h1.Fill(r.Uniform(minRange * .9, maxRange * 1.1));

   // Create vector - generate bin edges ( nbins is always > 2)
   // ignore fact that array may contains bins with zero size
   Double_t rebinArray[rebin];
   r.RndmArray(rebin, rebinArray);
   std::sort(rebinArray, rebinArray + rebin);
   for (Int_t i = 0; i < rebin; ++i) {
      rebinArray[i] = TMath::Nint(rebinArray[i] * (h1.GetNbinsX() - 2) + 2);
      rebinArray[i] = h1.GetBinLowEdge((Int_t)rebinArray[i]);
   }

#ifdef __DEBUG__
   std::cout << "min range = " << minRange << " max range " << maxRange << std::endl;
   for (Int_t i = 0; i < rebin; ++i) std::cout << rebinArray[i] << std::endl;
   std::cout << "rebin: " << rebin << std::endl;
#endif

   unique_ptr<TH1D> h2(static_cast<TH1D *>(h1.Rebin(rebin - 1, "testArrayRebin", rebinArray)));

   TH1D h3("testArrayRebin2", "testArrayRebin2", rebin - 1, rebinArray);
   r.SetSeed(seed);
   for (Int_t i = 0; i < nEvents; ++i) h3.Fill(r.Uniform(minRange * .9, maxRange * 1.1));

   EXPECT_TRUE(HistogramsEquals(*h2.get(), h3, cmpOptStats));
}

TEST(StressHistorgram, TestArrayRebinProfile)
{
   TH1::StatOverflows(kTRUE);

   // Tests rebin method with an array as input for 1D Profile

   const int rebin = TMath::Nint(r.Uniform(minRebin, maxRebin)) + 1;
   UInt_t seed = r.GetSeed();
   TProfile p1("p3", "Original Histogram", TMath::Nint(r.Uniform(1, 5)) * rebin * 2, minRange, maxRange);
   r.SetSeed(seed);
   for (Int_t i = 0; i < nEvents; ++i) {
      Double_t x = r.Uniform(minRange * .9, maxRange * 1.1);
      Double_t y = r.Uniform(minRange * .9, maxRange * 1.1);
      p1.Fill(x, y);
   }

   // Create vector - generate bin edges ( nbins is always > 2)
   // ignore fact that array may contains bins with zero size
   Double_t rebinArray[rebin];
   r.RndmArray(rebin, rebinArray);
   std::sort(rebinArray, rebinArray + rebin);
   for (Int_t i = 0; i < rebin; ++i) {
      rebinArray[i] = TMath::Nint(rebinArray[i] * (p1.GetNbinsX() - 2) + 2);
      rebinArray[i] = p1.GetBinLowEdge((Int_t)rebinArray[i]);
   }

#ifdef __DEBUG__
   for (Int_t i = 0; i < rebin; ++i) std::cout << rebinArray[i] << std::endl;
   std::cout << "rebin: " << rebin << std::endl;
#endif

   unique_ptr<TProfile> p2(static_cast<TProfile *>(p1.Rebin(rebin - 1, "testArrayRebinProf", rebinArray)));

   TProfile p3("testArrayRebinProf2", "testArrayRebinProf2", rebin - 1, rebinArray);
   r.SetSeed(seed);
   for (Int_t i = 0; i < nEvents; ++i) {
      Double_t x = r.Uniform(minRange * .9, maxRange * 1.1);
      Double_t y = r.Uniform(minRange * .9, maxRange * 1.1);
      p3.Fill(x, y);
   }

   EXPECT_TRUE(HistogramsEquals(*p2.get(), p3, cmpOptStats));
}
