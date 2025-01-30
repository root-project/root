#include "gtest/gtest.h"

#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
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

// ROOT-5439
TEST(TH1, DumpOutput)
{
   TH1F h;
   const auto line_fArray = "*fArray                       ->0";
   testing::internal::CaptureStdout();
   h.Dump();
   const std::string output = testing::internal::GetCapturedStdout();
   EXPECT_TRUE(output.find(line_fArray) != std::string::npos) << "Could not find '" << line_fArray << "' in the multiline output '" << output;
}


// https://github.com/root-project/root/issues/17552
TEST(TH1, AddBinContent)
{
   TH1F h1("h1", "h1", 10, 0, 1);
   h1.AddBinContent(1,1.);
   EXPECT_FLOAT_EQ(h1.GetBinContent(1),1.);
   TH2F h2("h2", "h2", 10, 0, 1, 2, 0, 3);
   h2.AddBinContent(1,1,1.);
   EXPECT_FLOAT_EQ(h2.GetBinContent(1,1),1.);
   TH3F h3("h3", "h3", 5, 0, 1, 2, 0, 2, 2, 0, 3);;
   h3.AddBinContent(1,1,1,1.);
   EXPECT_FLOAT_EQ(h3.GetBinContent(1,1,1),1.);
}
