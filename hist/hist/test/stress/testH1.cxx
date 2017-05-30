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

TEST(StressHistorgram, TestH1Buffer)
{
   TH1D *h1 = new TH1D("h1", "h1", 30, -3, 3);
   TH1D *h2 = new TH1D("h2", "h2", 30, -3, 3);

   // this activates the buffer for the histogram
   h1->SetBuffer(1000);

   // fill the histograms
   int nevt = 800;
   double x = 0;
   for (int i = 0; i < nevt; ++i) {
      x = gRandom->Gaus(0, 1);
      h1->Fill(x);
      h2->Fill(x);
   }
   double eps = TMath::Limits<double>::Epsilon();

   // now test that functions are consistent
   EXPECT_EQ(Equals(h1->GetMean(), h2->GetMean(), eps), 0)
      << "Histogram Mean = " << h1->GetMean() << "  " << h2->GetMean() << " -  " << std::endl;

   double s1[TH1::kNstat];
   double s2[TH1::kNstat];
   h1->GetStats(s1);
   h2->GetStats(s2);
   std::vector<std::string> snames = {"sumw", "sumw2", "sumwx", "sumwx2"};
   for (unsigned int i = 0; i < snames.size(); ++i) {
      EXPECT_EQ(Equals(s1[i], s2[i], eps), 0)
         << "Statistics " << snames[i] << "  = " << s1[i] << "  " << s2[i] << " -  " << std::endl;
   }

   // another fill will reset the histogram
   x = gRandom->Uniform(-3, 3);
   h1->Fill(x);
   h2->Fill(x);
   EXPECT_FALSE(h1->Integral() != h2->Integral() || h1->Integral() != h1->GetSumOfWeights())
      << "Histogram Integral = " << h1->Integral() << "  " << h2->Integral() << " s.o.w. = " << h1->GetSumOfWeights()
      << " -  " << std::endl;

   x = gRandom->Uniform(-3, 3);
   h1->Fill(x);
   h2->Fill(x);
   EXPECT_FALSE(h1->GetMaximum() != h2->GetMaximum())
      << "Histogram maximum = " << h1->GetMaximum() << "  " << h2->GetMaximum() << " -  " << std::endl;

   x = gRandom->Uniform(-3, 3);
   h1->Fill(x);
   h2->Fill(x);
   EXPECT_FALSE(h1->GetMinimum() != h2->GetMinimum())
      << "Histogram minimum = " << h1->GetMinimum() << "  " << h2->GetMinimum() << " - " << std::endl;

   x = gRandom->Uniform(-3, 3);
   h1->Fill(x);
   h2->Fill(x);
   int i1 = h1->FindFirstBinAbove(10);
   int i2 = h2->FindFirstBinAbove(10);
   EXPECT_EQ(i1, i2) << "Histogram first bin above  " << i1 << "  " << i2 << " - " << std::endl;

   x = gRandom->Uniform(-3, 3);
   h1->Fill(x);
   h2->Fill(x);
   h2->BufferEmpty();
   i1 = h1->FindLastBinAbove(10);
   i2 = h2->FindLastBinAbove(10);
   EXPECT_EQ(i1, i2) << "Histogram last bin above  " << i1 << "  " << i2 << " - " << std::endl;

   x = gRandom->Uniform(-3, 3);
   h1->Fill(x);
   h2->Fill(x);
   h2->BufferEmpty();
   double v1 = h1->Interpolate(0.1);
   double v2 = h2->Interpolate(0.1);
   EXPECT_EQ(Equals(v1, v2, eps), 0) << "Histogram interpolated value  " << v1 << "  " << v2 << " - " << std::endl;

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats, eps));

   delete h1;
}

TEST(StressHistorgram, TestH1BufferWeights)
{
   TH1D *h1 = new TH1D("h1", "h1", 30, -5, 5);
   TH1D *h2 = new TH1D("h2", "h2", 30, -5, 5);

   // set the buffer
   h1->SetBuffer(1000);

   // fill the histograms
   int nevt = 800;
   double x, w = 0;
   for (int i = 0; i < nevt; ++i) {
      x = gRandom->Gaus(0, 1);
      w = gRandom->Gaus(1, 0.1);
      h1->Fill(x, w);
      h2->Fill(x, w);
   }

   double eps = TMath::Limits<double>::Epsilon();

// Adjust the threshold on ARM64 bits. On this RISC architecture,
// there is a difference when incrementing the sumwx with variables
// saved in memory (in the histogram buffer) and passed as function
// arguments (Fill(x,w)).
#ifdef __aarch64__
   eps *= 28;
#endif

   double s1[TH1::kNstat];
   double s2[TH1::kNstat];
   h1->GetStats(s1);
   h2->GetStats(s2);
   std::vector<std::string> snames = {"sumw", "sumw2", "sumwx", "sumwx2"};
   for (unsigned int i = 0; i < snames.size(); ++i) {
      EXPECT_EQ(Equals(s1[i], s2[i], eps), 0)
         << "Statistics " << snames[i] << "  = " << s1[i] << "  " << s2[i] << " -  " << std::endl;
   }

   // another fill will reset the histogram
   x = gRandom->Uniform(-3, 3);
   w = 2;
   h1->Fill(x, w);
   h2->Fill(x, w);
   EXPECT_FALSE(h1->Integral() != h2->Integral() || h1->Integral() != h1->GetSumOfWeights())
      << "Histogram Integral = " << h1->Integral() << "  " << h2->Integral() << " s.o.w. = " << h1->GetSumOfWeights()
      << " -  " << std::endl;

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats, eps));

   delete h1;
}

TEST(StressHistorgram, TestH1Extend)
{

   TH1D *h1 = new TH1D("h1", "h1", 10, 0, 10);
   TH1D *h0 = new TH1D("h0", "h0", 10, 0, 20);
   h1->SetCanExtend(TH1::kXaxis);
   for (int i = 0; i < nEvents; ++i) {
      double x = gRandom->Gaus(10, 3);
      if (x <= 0 || x >= 20) continue; // do not want overflow in h0
      h1->Fill(x);
      h0->Fill(x);
   }
   EXPECT_TRUE(HistogramsEquals(h1, h0, cmpOptStats, 1E-10));
   delete h1;
}

TEST(StressHistorgram, TestH1Integral)
{
   int i1 = 1;
   int i2 = 100;

   int n = 10000;
   TH1D *h1 = new TH1D("h1", "h1", 100, -5, 5);
   TF1 *gaus = new TF1("gaus1d", gaus1d, -5, 5, 3);
   gaus->SetParameters(1, 0, 1);

   h1->FillRandom("gaus1d", n);

   TString fitOpt = "LQ0";
   if (defaultEqualOptions & cmpOptDebug) fitOpt = "L0";
   h1->Fit(gaus, fitOpt);

   // test first nentries
   double err = 0;
   double nent = h1->IntegralAndError(0, -1, err);

   EXPECT_EQ(nent, h1->GetEntries());

   double err1 = 0;
   double igh = h1->IntegralAndError(i1, i2, err1, "width");

   double x1 = h1->GetXaxis()->GetBinLowEdge(i1);
   double x2 = h1->GetXaxis()->GetBinUpEdge(i2);

   double igf = gaus->Integral(x1, x2);
   double err2 = gaus->IntegralError(x1, x2);

   double delta = fabs(igh - igf) / err2;

   if (defaultEqualOptions & cmpOptDebug) {
      std::cout << "Estimated entries = " << nent << " +/- " << err << std::endl;
      std::cout << "Histogram integral =  " << igh << " +/- " << err1 << std::endl;
      std::cout << "Function  integral =  " << igf << " +/- " << err2 << std::endl;
      std::cout << " Difference (histogram - function) in nsigma  = " << delta << std::endl;
   }

   EXPECT_FALSE(delta > 3);

   delete h1;
}
