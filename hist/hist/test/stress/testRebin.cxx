// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include "StressHistogramGlobal.h"

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "HFitInterface.h"
#include "TF1.h"
#include "TRandom2.h"

#include <sstream>
#include "gtest/gtest.h"

using namespace std;

TEST(StressHistogram, Test2DRebin)
{
   TRandom2 r;
   r.SetSeed(10);

   // Tests rebin method for 2D Histogram

   Int_t xrebin = TMath::Nint(r.Uniform(minRebin, maxRebin));
   Int_t yrebin = TMath::Nint(r.Uniform(minRebin, maxRebin));
   // make the bins of the orginal histo not an exact divider to leave an extra bin
   TH2D h2d("h2d", "Original Histogram", xrebin * TMath::Nint(r.Uniform(1, 5)), minRange, maxRange,
            yrebin * TMath::Nint(r.Uniform(1, 5)), minRange, maxRange);

   h2d.Sumw2();
   UInt_t seed = r.GetSeed();
   r.SetSeed(seed);
   for (Int_t i = 0; i < nEvents; ++i)
      h2d.Fill(r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(0, 10.));

   unique_ptr<TH2D> h2d2((TH2D *)h2d.Rebin2D(xrebin, yrebin, "p2d2"));

   // range of rebinned histogram may be different than original one
   TH2D h3("test2DRebin", "test2DRebin", h2d.GetNbinsX() / xrebin, h2d2->GetXaxis()->GetXmin(),
           h2d2->GetXaxis()->GetXmax(), h2d.GetNbinsY() / yrebin, h2d2->GetYaxis()->GetXmin(),
           h2d2->GetYaxis()->GetXmax());

   h3.Sumw2();
   r.SetSeed(seed);
   for (Int_t i = 0; i < nEvents; ++i)
      h3.Fill(r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(0, 10.));

   EXPECT_TRUE(HistogramsEquals(*h2d2.get(), h3, cmpOptStats));
}

TEST(StressHistogram, Test2DRebinProfile)
{
   TRandom2 r;
   // Tests rebin method for 2D Profile Histogram
   TProfile2D::Approximate();

   Int_t xrebin = TMath::Nint(r.Uniform(minRebin, maxRebin));
   Int_t yrebin = TMath::Nint(r.Uniform(minRebin, maxRebin));
   TProfile2D h2d("p2d", "Original Profile Histogram", xrebin * TMath::Nint(r.Uniform(1, 5)), minRange, maxRange,
                  yrebin * TMath::Nint(r.Uniform(1, 5)), minRange, maxRange);

   UInt_t seed = r.GetSeed();
   r.SetSeed(seed);
   for (Int_t i = 0; i < nEvents; ++i)
      h2d.Fill(r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(0, 10));

   unique_ptr<TProfile2D> h2d2(h2d.Rebin2D(xrebin, yrebin, "p2d2"));

   TProfile2D h3("test2DRebinProfile", "test2DRebin", h2d.GetNbinsX() / xrebin, h2d2->GetXaxis()->GetXmin(),
                 h2d2->GetXaxis()->GetXmax(), h2d.GetNbinsY() / yrebin, h2d2->GetYaxis()->GetXmin(),
                 h2d2->GetYaxis()->GetXmax());
   r.SetSeed(seed);
   for (Int_t i = 0; i < nEvents; ++i)
      h3.Fill(r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(0, 10));

   EXPECT_TRUE(HistogramsEquals(*h2d2.get(), h3, cmpOptStats));
}

TEST(StressHistogram, Test3DRebin)
{
   TRandom2 r;
   // Tests rebin method for 2D Histogram

   Int_t xrebin = TMath::Nint(r.Uniform(minRebin, maxRebin));
   Int_t yrebin = TMath::Nint(r.Uniform(minRebin, maxRebin));
   Int_t zrebin = TMath::Nint(r.Uniform(minRebin, maxRebin));

   // make the bins of the orginal histo not an exact divider to leave an extra bin
   TH3D h3d("h3d", "Original Histogram", xrebin * TMath::Nint(r.Uniform(1, 5)), minRange, maxRange,
            yrebin * TMath::Nint(r.Uniform(1, 5)), minRange, maxRange, zrebin * TMath::Nint(r.Uniform(1, 5)), minRange,
            maxRange);
   h3d.Sumw2();

   UInt_t seed = r.GetSeed();
   r.SetSeed(seed);
   for (Int_t i = 0; i < 10 * nEvents; ++i)
      h3d.Fill(r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(minRange * .9, maxRange * 1.1),
               r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(0, 10.));

   unique_ptr<TH3D> h3d2((TH3D *)h3d.Rebin3D(xrebin, yrebin, zrebin, "h3-rebin"));

   // range of rebinned histogram may be different than original one
   TH3D h3("test3DRebin", "test3DRebin", h3d.GetNbinsX() / xrebin, h3d2->GetXaxis()->GetXmin(),
           h3d2->GetXaxis()->GetXmax(), h3d.GetNbinsY() / yrebin, h3d2->GetYaxis()->GetXmin(),
           h3d2->GetYaxis()->GetXmax(), h3d.GetNbinsZ() / zrebin, h3d2->GetZaxis()->GetXmin(),
           h3d2->GetZaxis()->GetXmax());
   h3.Sumw2();
   r.SetSeed(seed);
   for (Int_t i = 0; i < 10 * nEvents; ++i)
      h3.Fill(r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(minRange * .9, maxRange * 1.1),
              r.Uniform(minRange * .9, maxRange * 1.1), r.Uniform(0, 10.));

   EXPECT_TRUE(HistogramsEquals(*h3d2.get(), h3, cmpOptStats));
}
