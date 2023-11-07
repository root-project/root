#include "gtest/gtest.h"

#include "TH1.h"
#include "TH1F.h"
#include "THLimitsFinder.h"

#include <vector>

// StatOverflows TH1
TEST(TH1, StatOverflows)
{
   TH1F h0("h0", "h0", 1, 0, 1);
   TH1F h1("h1", "h1", 1, 0, 1);
   TH1F h2("h2", "h2", 1, 0, 1);
   EXPECT_EQ(TH1::EStatOverflows::kNeutral, h0.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kNeutral, h1.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kNeutral, h2.GetStatOverflows());

   h0.SetStatOverflows(TH1::EStatOverflows::kIgnore);
   h1.SetStatOverflows(TH1::EStatOverflows::kConsider);
   h2.SetStatOverflows(TH1::EStatOverflows::kNeutral);
   EXPECT_EQ(TH1::EStatOverflows::kIgnore,   h0.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kConsider, h1.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kNeutral,  h2.GetStatOverflows());

   TH1::StatOverflows(true);
   EXPECT_EQ(TH1::EStatOverflows::kIgnore,   h0.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kConsider, h1.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kNeutral,  h2.GetStatOverflows());

   TH1::StatOverflows(false);
   EXPECT_EQ(TH1::EStatOverflows::kIgnore,   h0.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kConsider, h1.GetStatOverflows());
   EXPECT_EQ(TH1::EStatOverflows::kNeutral,  h2.GetStatOverflows());
}

// THLimitsFinder, borderline cases
TEST(THLimitsFinder, Degenerate)
{
   // https://root-forum.cern.ch/t/problem-using-long64-t-and-l-type-with-tbranch/43021

   Int_t newBins = -1;
   static const Long64_t centralValue = 3711308690032;
   Double_t xmin = centralValue - 5.;
   Double_t xmax = centralValue + 5.;
   THLimitsFinder::OptimizeLimits(10, newBins, xmin, xmax, /*isInteger*/ true);
   EXPECT_LT(xmin, xmax);
   EXPECT_LE(xmin, centralValue - 5.);
   EXPECT_GE(xmax, centralValue + 5.);
}

// Simple cross-check that TH1::SmoothArray() is not doing anything if input
// array is already smooth.
TEST(TH1, SmoothArrayCrossCheck)
{
   std::vector<double> arr1{0., 1., 2., 3., 4.};
   std::vector<double> arr2{1., 1., 1., 1., 1.};
   TH1::SmoothArray(arr1.size(), arr1.data());
   TH1::SmoothArray(arr2.size(), arr2.data());

   for (std::size_t i = 0; i < arr1.size(); ++i) {
      EXPECT_FLOAT_EQ(arr1[i], i);
      EXPECT_FLOAT_EQ(arr2[i], 1.0);
   }
}
