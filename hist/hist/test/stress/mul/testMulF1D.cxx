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

#include "gtest/gtest.h"

#include "../StressHistogramGlobal.h"

using namespace std;

TEST(StressHistogram, TestMulF1D)
{
   Double_t c1 = r.Rndm();

   TH1D h1("mf1D-h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D h2("mf1D-h2", "h2=h1*c1*f1", numberOfBins, minRange, maxRange);

   TF1 f("sin", "sin(x)", minRange - 2, maxRange + 2);

   h1.Sumw2();
   h2.Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
      h2.Fill(value, f.Eval(h2.GetBinCenter(h2.FindBin(value))) * c1);
   }

   h1.Multiply(&f, c1);

   // stats fails because of the error precision
   EXPECT_TRUE(HistogramsEquals(h1, h2));
}

TEST(StressHistogram, TestMulF1D2)
{
   Double_t c1 = r.Rndm();

   TH1D h1("mf1D2-h1", "h1-Title", numberOfBins, minRange, maxRange);
   TH1D h2("mf1D2-h2", "h2=h1*c1*f1", numberOfBins, minRange, maxRange);

   TF2 f("sin2", "sin(x)*cos(y)", minRange - 2, maxRange + 2, minRange - 2, maxRange + 2);
   h1.Sumw2();
   h2.Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents; ++e) {
      Double_t value = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1.Fill(value, 1.0);
      h2.Fill(value,
               f.Eval(h2.GetXaxis()->GetBinCenter(h2.GetXaxis()->FindBin(value)),
                       h2.GetYaxis()->GetBinCenter(h2.GetYaxis()->FindBin(double(0)))) *
                  c1);
   }

   h1.Multiply(&f, c1);

   // stats fails because of the error precision
   EXPECT_TRUE(HistogramsEquals(h1, h2));
}
