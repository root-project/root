// @(#)root/test:$Id$
// Authors: David Gonzalez Maline November 2008
//          Martin Storø Nyfløtt  June 2017

#include <cmath>
#include <sstream>

#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include "TF1.h"

#include "HFitInterface.h"

#include "Riostream.h"
#include "TClass.h"
#include "TFile.h"
#include "TRandom2.h"

#include "gtest/gtest.h"

#include "StressHistogramGlobal.h"

using namespace std;

TEST(StressHistogram, TestH3Buffer)
{
   TH3D h1("h1", "h1", 4, -5, 5, 4, -5, 5, 4, -5, 5);
   TH3D h2("h2", "h2", 4, -5, 5, 4, -5, 5, 4, -5, 5);

   // set the buffer
   h1.SetBuffer(10000);

   // fill the histograms
   int nevt = 8000;
   double x, y, z = 0;
   for (int i = 0; i < nevt; ++i) {
      x = gRandom->Gaus(0, 2);
      y = gRandom->Gaus(1, 3);
      z = gRandom->Uniform(-5, 5);
      h1.Fill(x, y, z);
      h2.Fill(x, y, z);
   }

   EXPECT_FALSE(h1.Integral() != h2.Integral() || h1.Integral() != h1.GetSumOfWeights())
      << "Histogram Integral = " << h1.Integral() << "  " << h2.Integral() << " s.o.w. = " << h1.GetSumOfWeights()
      << " -  " << std::endl;

   // test adding extra fills with weights
   for (int i = 0; i < nevt; ++i) {
      x = gRandom->Uniform(-3, 3);
      y = gRandom->Uniform(-3, 3);
      z = gRandom->Uniform(-5, 5);
      double w = 2;
      h1.Fill(x, y, z, w);
      h2.Fill(x, y, z, w);
   }

   EXPECT_TRUE(HistogramsEquals(h1, h2, cmpOptStats, 1.E-15));
}

TEST(StressHistogram, TestH3Integral)
{
   int ix1 = 1;
   int ix2 = 50;
   int iy1 = 1;
   int iy2 = 50;
   int iz1 = 1;
   int iz2 = 50;

   TStopwatch w;
   int n = 1000000;
   TH3D h3("h3", "h3", 50, -5, 5, 50, -5, 5, 50, -5, 5);

   // TF3 * gaus = new TF3("gaus3d",gaus3d,-5,5,-5,5,-5,5,7);
   TF3 gaus("gaus3d", gaus3d, -5, 5, -5, 5, -5, 5, 7);
   gaus.SetParameters(100, 0, 1.3, 1., 1., -1, 0.9);
   w.Start();
   h3.FillRandom("gaus3d", n);

   // gaus.SetParameter(0, h3.GetMaximum() );

   TString fitOpt = "LQ0";
   w.Stop();
   if (defaultEqualOptions & cmpOptDebug) {
      std::cout << "Time to fill random " << w.RealTime() << std::endl;
      fitOpt = "L0";
   }
   w.Start();
   h3.Fit(&gaus, fitOpt);
   if (defaultEqualOptions & cmpOptDebug) std::cout << "Time to fit         " << w.RealTime() << std::endl;

   // test first nentries
   double err = 0;
   w.Start();
   double nent = h3.IntegralAndError(0, -1, 0, -1, 0, -1, err);
   w.Stop();
   if (defaultEqualOptions & cmpOptDebug) {
      std::cout << "Estimated entries = " << nent << " +/- " << err << std::endl;
      std::cout << "Time to integral of all  " << w.RealTime() << std::endl;
   }

   EXPECT_EQ(nent, h3.GetEntries());

   double err1 = 0;
   w.Start();
   double igh = h3.IntegralAndError(ix1, ix2, iy1, iy2, iz1, iz2, err1, "width");
   w.Stop();
   if (defaultEqualOptions & cmpOptDebug) std::cout << "Time to integral of selected  " << w.RealTime() << std::endl;

   double x1 = h3.GetXaxis()->GetBinLowEdge(ix1);
   double x2 = h3.GetXaxis()->GetBinUpEdge(ix2);
   double y1 = h3.GetYaxis()->GetBinLowEdge(iy1);
   double y2 = h3.GetYaxis()->GetBinUpEdge(iy2);
   double z1 = h3.GetZaxis()->GetBinLowEdge(iz1);
   double z2 = h3.GetZaxis()->GetBinUpEdge(iz2);

   double a[3];
   double b[3];
   a[0] = x1;
   a[1] = y1;
   a[2] = z1;
   b[0] = x2;
   b[1] = y2;
   b[2] = z2;

   w.Start();
   double relerr = 0;
   double igf = gaus.IntegralMultiple(3, a, b, 1.E-4, relerr); // don't need high tolerance (use 10-4)
   // double igf = gaus.Integral(x1,x2,y1,y2,z1,z2,1.E-4);  // don't need high tolerance

   double err2 = gaus.IntegralError(3, a, b);
   w.Stop();

   double delta = fabs(igh - igf) / err1;

   if (defaultEqualOptions & cmpOptDebug) {
      std::cout << "Time to function integral   " << w.RealTime() << std::endl;
      std::cout << "Histogram integral =  " << igh << " +/- " << err1 << std::endl;
      std::cout << "Function  integral =  " << igf << " +/- " << err2 << " +/- " << igf * relerr << std::endl;
      std::cout << " Difference (histogram - function) in nsigma  = " << delta << std::endl;
   }

   EXPECT_FALSE(delta > 3);
}
