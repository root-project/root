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

TEST(StressHistogram, TestH2Buffer)
{
   TRandom2 r;
   TH2D h1("h1", "h1", 10, -5, 5, 10, -5, 5);
   TH2D h2("h2", "h2", 10, -5, 5, 10, -5, 5);

   // set the buffer
   h1.SetBuffer(1000);

   // fill the histograms
   int nevt = 800;
   double x, y = 0;
   for (int i = 0; i < nevt; ++i) {
      x = gRandom->Gaus(0, 2);
      y = gRandom->Gaus(1, 3);
      h1.Fill(x, y);
      h2.Fill(x, y);
   }

   EXPECT_FALSE(h1.Integral() != h2.Integral() || h1.Integral() != h1.GetSumOfWeights())
      << "Histogram Integral = " << h1.Integral() << "  " << h2.Integral() << " s.o.w. = " << h1.GetSumOfWeights()
      << " -  " << std::endl;

   // test adding an extra fill
   x = gRandom->Uniform(-3, 3);
   y = gRandom->Uniform(-3, 3);
   double w = 2;
   h1.Fill(x, y, w);
   h2.Fill(x, y, w);

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats, 1.E-15));
}

TEST(StressHistogram, TestH2Extend)
{
   TRandom2 r;
   TH2D h1("h1", "h1", 10, 0, 10, 10, 0, 10);
   TH2D h2("h2", "h0", 10, 0, 10, 10, 0, 20);
   h1.SetCanExtend(TH1::kYaxis);
   for (int i = 0; i < nEvents; ++i) {
      double x = r.Uniform(-1, 11);
      double y = r.Gaus(10, 3);
      if (y <= 0 || y >= 20) continue; // do not want overflow in h0
      h1.Fill(x, y);
      h2.Fill(x, y);
   }

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats, 1E-10));
}

TEST(StressHistogram, TestH2Integral)
{
   TRandom2 r;
   int ix1 = 1;
   int ix2 = 50;
   int iy1 = 1;
   int iy2 = 50;

   int n = 10000;
   TH2D h2("h2", "h2", 50, -5, 5, 50, -5, 5);

   TF2 gaus("gaus2d", gaus2d, -5, 5, -5, 5, 5);
   gaus.SetParameters(100, 0, 1.2, 1., 1);
   h2.FillRandom("gaus2d", n);
   TString fitOpt = "LQ0";
   if (defaultEqualOptions & cmpOptDebug) fitOpt = "L0";
   h2.Fit(&gaus, fitOpt);

   // test first nentries
   double err = 0;
   double nent = h2.IntegralAndError(0, -1, 0, -1, err);

   EXPECT_EQ(nent, h2.GetEntries());

   double err1 = 0;
   double igh = h2.IntegralAndError(ix1, ix2, iy1, iy2, err1, "width");

   double x1 = h2.GetXaxis()->GetBinLowEdge(ix1);
   double x2 = h2.GetXaxis()->GetBinUpEdge(ix2);
   double y1 = h2.GetYaxis()->GetBinLowEdge(iy1);
   double y2 = h2.GetYaxis()->GetBinUpEdge(iy2);

   double a[2];
   double b[2];
   a[0] = x1;
   a[1] = y1;
   b[0] = x2;
   b[1] = y2;

   // double igf = gaus->Integral(x1,x2,y1,y2,1.E-4);
   double relerr = 0;
   double igf = gaus.IntegralMultiple(2, a, b, 1.E-4, relerr); // don't need high tolerance (use 10-4)
   double err2 = gaus.IntegralError(2, a, b);

   double delta = fabs(igh - igf) / err1;

   if (defaultEqualOptions & cmpOptDebug) {
      std::cout << "Estimated entries = " << nent << " +/- " << err << std::endl;
      std::cout << "Histogram integral =  " << igh << " +/- " << err1 << std::endl;
      std::cout << "Function  integral =  " << igf << " +/- " << err2 << " +/- " << igf * relerr << std::endl;
      std::cout << " Difference (histogram - function) in nsigma  = " << delta << std::endl;
   }

   EXPECT_FALSE(delta > 3);
}
