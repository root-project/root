// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <sstream>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TProfile.h"
#include "TProfile2D.h"
#include "TProfile3D.h"

#include "TF1.h"

#include "Fit/SparseData.h"
#include "HFitInterface.h"

#include "Math/IntegratorOptions.h"

#include "TRandom2.h"
#include "TFile.h"
#include "TClass.h"

#include "TROOT.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

class RefReadTest : public ::testing::Test {
protected:
   unique_ptr<TFile> refFile;
   TRandom2 r = TRandom2();

   virtual void SetUp()
   {
      TH1::StatOverflows(kTRUE);
      r.SetSeed(8652);

      const char *refFileName = "http://root.cern.ch/files/stressHistogram.testRefRead.6.10.0.root";

      if (refFileOption == refFileWrite) {
         refFile.reset(new TFile(refFileName, "UPDATE"));
      } else {
         auto isBatch = gROOT->IsBatch();

         refFile.reset(TFile::Open(refFileName));
         gROOT->SetBatch();
         TFile::SetCacheFileDir(".");
         gROOT->SetBatch(isBatch);
      }
   }

   virtual void TearDown() { refFile->Close(); }
};

TEST_F(RefReadTest, TestRefRead1D)
{
   // Tests consistency with a reference file for 1D Histogram
   if (refFileOption == refFileWrite) {
      TH1D h1("rr1D-h1", "h1-Title", numberOfBins, minRange, maxRange);
      h1.Sumw2();

      for (Int_t e = 0; e < nEvents; ++e) {
         Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         h1.Fill(value, 1.0);
      }
      h1.Write();
   } else {
      unique_ptr<TH1D> h1(static_cast<TH1D *>(refFile->Get("rr1D-h1")));
      TH1D h2("rr1D-h2", "h2-Title", numberOfBins, minRange, maxRange);
      h2.Sumw2();

      for (Int_t e = 0; e < nEvents; ++e) {
         Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         h2.Fill(value, 1.0);
      }

      EXPECT_EQ(0, Equals("Ref Read Hist 1D", *h1.get(), h2, cmpOptStats));
   }
}

TEST_F(RefReadTest, TestRefRead2D)
{
   // Tests consistency with a reference file for 2D Histogram
   if (refFileOption == refFileWrite) {
      TH2D h1("rr2D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);
      h1.Sumw2();

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         h1.Fill(x, y, 1.0);
      }
      h1.Write();
   } else {
      unique_ptr<TH2D> h1(static_cast<TH2D *>(refFile->Get("rr2D-h1")));
      TH2D h2("rr2D-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);
      h2.Sumw2();

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         h2.Fill(x, y, 1.0);
      }

      EXPECT_EQ(0, Equals("Ref Read Hist 2D", *h1.get(), h2, cmpOptStats));
   }
}

TEST_F(RefReadTest, TestRefRead3D)
{
   // Tests consistency with a reference file for 3D Histogram

   if (refFileOption == refFileWrite) {
      TH3D h1("rr3D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange, numberOfBins,
              minRange, maxRange);
      h1.Sumw2();

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         h1.Fill(x, y, z, 1.0);
      }
      h1.Write();
   } else {
      unique_ptr<TH3D> h1(static_cast<TH3D *>(refFile->Get("rr3D-h1")));
      TH3D h2("rr3D-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange, numberOfBins,
              minRange, maxRange);
      h2.Sumw2();

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         h2.Fill(x, y, z, 1.0);
      }

      EXPECT_EQ(0, Equals("Ref Read Hist 3D", *h1.get(), h2, cmpOptStats));
   }
}

TEST_F(RefReadTest, TestRefReadProf1D)
{
   // Tests consistency with a reference file for 1D Profile

   if (refFileOption == refFileWrite) {
      TProfile p1("rr1D-p1", "p1-Title", numberOfBins, minRange, maxRange);
      //      p1.Sumw2();

      for (Int_t e = 0; e < nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p1.Fill(x, y, 1.0);
      }
      p1.Write();
   } else {
      TH1::SetDefaultSumw2(false);
      unique_ptr<TProfile> p1(static_cast<TProfile *>(refFile->Get("rr1D-p1")));
      TProfile p2("rr1D-p2", "p2-Title", numberOfBins, minRange, maxRange);
      //      p2.Sumw2();

      for (Int_t e = 0; e < nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p2.Fill(x, y, 1.0);
      }

      EXPECT_EQ(0, Equals("Ref Read Prof 1D", *p1.get(), p2, cmpOptStats));
      TH1::SetDefaultSumw2(true);
   }
}

TEST_F(RefReadTest, TestRefReadProf2D)
{
   // Tests consistency with a reference file for 2D Profile
   if (refFileOption == refFileWrite) {
      TProfile2D p1("rr2D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p1.Fill(x, y, z, 1.0);
      }
      p1.Write();
   } else {
      unique_ptr<TProfile2D> p1(static_cast<TProfile2D *>(refFile->Get("rr2D-p1")));
      TProfile2D p2("rr2D-p2", "p2-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p2.Fill(x, y, z, 1.0);
      }

      EXPECT_EQ(0, Equals("Ref Read Prof 2D", *p1.get(), p2, cmpOptStats));
   }
}

TEST_F(RefReadTest, TestRefReadProf3D)
{
   // Tests consistency with a reference file for 3D Profile
   if (refFileOption == refFileWrite) {
      TProfile3D p1("rr3D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange,
                    numberOfBins, minRange, maxRange);

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p1.Fill(x, y, z, t, 1.0);
      }
      p1.Write();
   } else {
      unique_ptr<TProfile3D> p1(static_cast<TProfile3D *>(refFile->Get("rr3D-p1")));
      TProfile3D p2("rr3D-p2", "p2-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange,
                    numberOfBins, minRange, maxRange);

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p2.Fill(x, y, z, t, 1.0);
      }

      EXPECT_EQ(0, Equals("Ref Read Prof 3D", *p1.get(), p2, cmpOptStats));
   }
}

TEST_F(RefReadTest, TestRefReadSparse)
{
   // Tests consistency with a reference file for Sparse Histogram
   Int_t bsize[] = {TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5))};
   Double_t xmin[] = {minRange, minRange, minRange};
   Double_t xmax[] = {maxRange, maxRange, maxRange};

   if (refFileOption == refFileWrite) {
      THnSparseD s1("rr-s1", "s1-Title", 3, bsize, xmin, xmax);
      s1.Sumw2();

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t points[3];
         points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
         points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
         points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
         s1.Fill(points);
      }
      s1.Write();
   } else {
      unique_ptr<THnSparseD> s1(static_cast<THnSparseD *>(refFile->Get("rr-s1")));
      THnSparseD s2("rr-s1", "s1-Title", 3, bsize, xmin, xmax);
      s2.Sumw2();

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t points[3];
         points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
         points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
         points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
         s2.Fill(points);
      }

      EXPECT_EQ(0, Equals("Ref Read Sparse", *s1.get(), s2, cmpOptStats));
   }
}
