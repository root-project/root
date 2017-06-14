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

TEST(StressHistorgram, TestMulF2D)
{
   Double_t c1 = r.Rndm();

   TH2D *h1 = new TH2D("mf2D-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);
   TH2D *h2 = new TH2D("mf2D-h2", "h2=h1*c1*f1", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);

   TF1 *f = new TF1("sin", "sin(x)", minRange - 2, maxRange + 2);

   h1->Sumw2();
   h2->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
      h2->Fill(x, y, f->Eval(h2->GetXaxis()->GetBinCenter(h2->GetXaxis()->FindBin(x))) * c1);
   }

   h1->Multiply(f, c1);

   // stats fails because of the error precision
   EXPECT_TRUE(HistogramsEquals(h1, h2)); //, cmpOptStats | cmpOptDebug);
   delete h1;
   delete h2;
   delete f;
}

TEST(StressHistorgram, TestMulF2D2)
{
   Double_t c1 = r.Rndm();

   TH2D *h1 = new TH2D("mf2D2-h1", "h1-Title", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);
   TH2D *h2 = new TH2D("mf2D2-h2", "h2=h1*c1*f1", numberOfBins, minRange, maxRange, numberOfBins, minRange, maxRange);

   TF2 *f = new TF2("sin2", "sin(x)*cos(y)", minRange - 2, maxRange + 2, minRange - 2, maxRange + 2);

   h1->Sumw2();
   h2->Sumw2();

   UInt_t seed = r.GetSeed();
   // For possible problems
   r.SetSeed(seed);
   for (Int_t e = 0; e < nEvents * nEvents; ++e) {
      Double_t x = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      Double_t y = r.Uniform(0.9 * minRange, 1.1 * maxRange);
      h1->Fill(x, y, 1.0);
      h2->Fill(x, y,
               f->Eval(h2->GetXaxis()->GetBinCenter(h2->GetXaxis()->FindBin(x)),
                       h2->GetYaxis()->GetBinCenter(h2->GetYaxis()->FindBin(y))) *
                  c1);
   }

   h1->Multiply(f, c1);

   // stats fails because of the error precision
   EXPECT_TRUE(HistogramsEquals(h1, h2));
   delete h1;
   delete h2;
   delete f;
}
