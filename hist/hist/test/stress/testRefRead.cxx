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
   TFile *refFile;
   virtual void SetUp()
   {
      TH1::StatOverflows(kTRUE);
      r.SetSeed(8652);

      const char *refFileName = "http://root.cern.ch/files/stressHistogram.testRefRead.6.10.0.root";

      if (refFileOption == refFileWrite) {
         refFile = new TFile(refFileName, "UPDATE");
      } else {
         auto isBatch = gROOT->IsBatch();

         refFile = TFile::Open(refFileName);
         gROOT->SetBatch();
         TFile::SetCacheFileDir(".");
         gROOT->SetBatch(isBatch);
      }
   }

   virtual void TearDown()
   {
      refFile->Close();
      delete refFile;
   }
};

TEST_F(RefReadTest, TestRefRead1D)
{
   // Tests consistency with a reference file for 1D Histogram
   TH1D *h1 = 0;
   if (refFileOption == refFileWrite) {
      h1 = new TH1D("rr1D-h1", "h1-Title", numberOfBins, minRange, maxRange);
      h1->Sumw2();

      for (Int_t e = 0; e < nEvents; ++e) {
         Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         h1->Fill(value, 1.0);
      }
      h1->Write();
   } else {
      h1 = static_cast<TH1D *>(refFile->Get("rr1D-h1"));
      TH1D *h2 = new TH1D("rr1D-h2", "h2-Title", numberOfBins, minRange, maxRange);
      h2->Sumw2();

      for (Int_t e = 0; e < nEvents; ++e) {
         Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         h2->Fill(value, 1.0);
      }

      EXPECT_EQ(0, Equals("Ref Read Hist 1D", h1, h2, cmpOptStats));
   }

   if (h1) delete h1;
}

TEST_F(RefReadTest, TestRefRead2D)
{
   // Tests consistency with a reference file for 2D Histogram
   TH2D *h1 = 0;
   if (refFileOption == refFileWrite) {
      h1 = new TH2D("rr2D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);
      h1->Sumw2();

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         h1->Fill(x, y, 1.0);
      }
      h1->Write();
   } else {
      h1 = static_cast<TH2D *>(refFile->Get("rr2D-h1"));
      TH2D *h2 = new TH2D("rr2D-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);
      h2->Sumw2();

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         h2->Fill(x, y, 1.0);
      }

      EXPECT_EQ(0, Equals("Ref Read Hist 2D", h1, h2, cmpOptStats));
   }
   if (h1) delete h1;
}

TEST_F(RefReadTest, TestRefRead3D)
{
   // Tests consistency with a reference file for 3D Histogram

   TH3D *h1 = 0;
   if (refFileOption == refFileWrite) {
      h1 = new TH3D("rr3D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange,
                    numberOfBins, minRange, maxRange);
      h1->Sumw2();

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         h1->Fill(x, y, z, 1.0);
      }
      h1->Write();
   } else {
      h1 = static_cast<TH3D *>(refFile->Get("rr3D-h1"));
      TH3D *h2 = new TH3D("rr3D-h2", "h2-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange,
                          numberOfBins, minRange, maxRange);
      h2->Sumw2();

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         h2->Fill(x, y, z, 1.0);
      }

      EXPECT_EQ(0, Equals("Ref Read Hist 3D", h1, h2, cmpOptStats));
   }
   if (h1) delete h1;
}

TEST_F(RefReadTest, TestRefReadProf1D)
{
   // Tests consistency with a reference file for 1D Profile

   TProfile *p1 = 0;
   if (refFileOption == refFileWrite) {
      p1 = new TProfile("rr1D-p1", "p1-Title", numberOfBins, minRange, maxRange);
      //      p1->Sumw2();

      for (Int_t e = 0; e < nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p1->Fill(x, y, 1.0);
      }
      p1->Write();
   } else {
      TH1::SetDefaultSumw2(false);
      p1 = static_cast<TProfile *>(refFile->Get("rr1D-p1"));
      TProfile *p2 = new TProfile("rr1D-p2", "p2-Title", numberOfBins, minRange, maxRange);
      //      p2->Sumw2();

      for (Int_t e = 0; e < nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p2->Fill(x, y, 1.0);
      }

      EXPECT_EQ(0, Equals("Ref Read Prof 1D", p1, p2, cmpOptStats));
      TH1::SetDefaultSumw2(true);
   }

   if (p1) delete p1;
}

TEST_F(RefReadTest, TestRefReadProf2D)
{
   // Tests consistency with a reference file for 2D Profile
   TProfile2D *p1 = 0;
   if (refFileOption == refFileWrite) {
      p1 = new TProfile2D("rr2D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p1->Fill(x, y, z, 1.0);
      }
      p1->Write();
   } else {
      p1 = static_cast<TProfile2D *>(refFile->Get("rr2D-p1"));
      TProfile2D *p2 =
         new TProfile2D("rr2D-p2", "p2-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p2->Fill(x, y, z, 1.0);
      }

      EXPECT_EQ(0, Equals("Ref Read Prof 2D", p1, p2, cmpOptStats));
   }

   if (p1) delete p1;
}

TEST_F(RefReadTest, TestRefReadProf3D)
{
   // Tests consistency with a reference file for 3D Profile
   TProfile3D *p1 = 0;
   if (refFileOption == refFileWrite) {
      p1 = new TProfile3D("rr3D-p1", "p1-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange,
                          numberOfBins, minRange, maxRange);

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p1->Fill(x, y, z, t, 1.0);
      }
      p1->Write();
   } else {
      p1 = static_cast<TProfile3D *>(refFile->Get("rr3D-p1"));
      TProfile3D *p2 = new TProfile3D("rr3D-p2", "p2-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange,
                                      maxRange, numberOfBins, minRange, maxRange);

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t z = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         Double_t t = r.Uniform(0.9 * minRange, 1.1 * maxRange);
         p2->Fill(x, y, z, t, 1.0);
      }

      EXPECT_EQ(0, Equals("Ref Read Prof 3D", p1, p2, cmpOptStats));
   }

   if (p1) delete p1;
}

TEST_F(RefReadTest, TestRefReadSparse)
{
   // Tests consistency with a reference file for Sparse Histogram
   Int_t bsize[] = {TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5)), TMath::Nint(r.Uniform(1, 5))};
   Double_t xmin[] = {minRange, minRange, minRange};
   Double_t xmax[] = {maxRange, maxRange, maxRange};

   THnSparseD *s1 = 0;

   if (refFileOption == refFileWrite) {
      s1 = new THnSparseD("rr-s1", "s1-Title", 3, bsize, xmin, xmax);
      s1->Sumw2();

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t points[3];
         points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
         points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
         points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
         s1->Fill(points);
      }
      s1->Write();
   } else {
      s1 = static_cast<THnSparseD *>(refFile->Get("rr-s1"));
      THnSparseD *s2 = new THnSparseD("rr-s1", "s1-Title", 3, bsize, xmin, xmax);
      s2->Sumw2();

      for (Int_t e = 0; e < nEvents * nEvents; ++e) {
         Double_t points[3];
         points[0] = r.Uniform(minRange * .9, maxRange * 1.1);
         points[1] = r.Uniform(minRange * .9, maxRange * 1.1);
         points[2] = r.Uniform(minRange * .9, maxRange * 1.1);
         s2->Fill(points);
      }

      EXPECT_EQ(0, Equals("Ref Read Sparse", s1, s2, cmpOptStats));
   }

   if (s1) delete s1;
}
