#include "gtest/gtest.h"

#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TH1F.h"
#include "THLimitsFinder.h"

#include <cmath>
#include <cstddef>
#include <random>
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

// https://github.com/root-project/root/pull/17751
// https://root-forum.cern.ch/t/different-ways-of-normalizing-histograms/15582
TEST(TH1, Normalize)
{
   TH1F h1("h1", "h1", 10, 0, 1);
   h1.SetBinContent(1, 1.);
   h1.SetBinContent(3, 5.);
   h1.SetBinContent(5, 10.);
   h1.SetBinContent(7, -1.);
   h1.SetBinContent(10, 3.);
   EXPECT_FLOAT_EQ(h1.GetEntries(), 5.);
   EXPECT_FLOAT_EQ(h1.GetSumOfWeights(), 18.);
   EXPECT_FLOAT_EQ(h1.Integral(), 18.);
   EXPECT_FLOAT_EQ(h1.Integral("width"), 1.8);
   EXPECT_FLOAT_EQ(h1.GetMaximum(), 10.);
   TH1F h2(h1);
   h2.Normalize();
   EXPECT_FLOAT_EQ(h2.GetEntries(), 5);
   EXPECT_FLOAT_EQ(h2.GetSumOfWeights(), 10.);
   EXPECT_FLOAT_EQ(h2.Integral(), 10.);
   EXPECT_FLOAT_EQ(h2.Integral("width"), 1.);
   EXPECT_FLOAT_EQ(h2.GetMaximum(), 50. / 9);
   TH1F h3(h1);
   h3.Normalize("max");
   EXPECT_FLOAT_EQ(h3.GetEntries(), 5.);
   EXPECT_FLOAT_EQ(h3.GetSumOfWeights(), 1.8);
   EXPECT_FLOAT_EQ(h3.Integral(), 1.8);
   EXPECT_FLOAT_EQ(h3.Integral("width"), 0.18);
   EXPECT_FLOAT_EQ(h3.GetMaximum(), 1.);
   TH1F h4(h1);
   h4.Normalize("sum");
   EXPECT_FLOAT_EQ(h4.GetEntries(), 5.);
   EXPECT_FLOAT_EQ(h4.GetSumOfWeights(), 1);
   EXPECT_FLOAT_EQ(h4.Integral(), 1);
   EXPECT_FLOAT_EQ(h4.Integral("width"), 0.1);
   EXPECT_FLOAT_EQ(h4.GetMaximum(), 5. / 9);
   const Float_t xbins[10 + 1]{0., 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 0.95, 1.};
   TH1F v1("v1", "vbw", 10, xbins);
   v1.SetBinContent(1, 1.);
   v1.SetBinContent(3, 5.);
   v1.SetBinContent(5, 10.);
   v1.SetBinContent(7, -1.);
   v1.SetBinContent(10, 3.);
   EXPECT_FLOAT_EQ(v1.GetEntries(), 5.);
   EXPECT_FLOAT_EQ(v1.GetSumOfWeights(), 18.);
   EXPECT_FLOAT_EQ(v1.Integral(), 18.);
   EXPECT_FLOAT_EQ(v1.Integral("width"), 1.25);
   EXPECT_FLOAT_EQ(v1.GetMaximum(), 10.);
   TH1F v2(v1);
   v2.Normalize();
   EXPECT_FLOAT_EQ(v2.GetEntries(), 5);
   EXPECT_FLOAT_EQ(v2.GetSumOfWeights(), 14.399998);
   EXPECT_FLOAT_EQ(v2.Integral(), 14.399998);
   EXPECT_FLOAT_EQ(v2.Integral("width"), 1.);
   EXPECT_FLOAT_EQ(v2.GetMaximum(), 7.9999990);
}

TEST(TAxis, BinComputation_FPAccuracy)
{
   // Example from 1703c54
   EXPECT_EQ(TAxis(1, -1., 0.).FindBin(-1e-17), 1);

   {
      // https://root-forum.cern.ch/t/floating-point-rounding-error-when-filling-the-histogram/35189
      TAxis axis(128, -0.352, 0.352);
      constexpr double x = -0.0220;
      EXPECT_EQ(axis.FindBin(x), 61);
      EXPECT_EQ(axis.FindFixBin(x), 61);
      EXPECT_LE(axis.GetBinLowEdge(61), x);
      EXPECT_GT(axis.GetBinUpEdge(61), x);
   }

   {
      // https://github.com/root-project/root/issues/14091
      constexpr int nBins = 30;
      constexpr double xMin = 3.0, xMax = 6.0;
      TAxis ax(nBins, xMin, xMax);

      for (Int_t i = 1; i <= ax.GetNbins(); i++) {
         EXPECT_EQ(i, ax.FindBin(ax.GetBinLowEdge(i)));
      }
   }

   {
      TAxis axis(2000, -1000., 1000.);
      EXPECT_EQ(axis.FindFixBin(std::nextafter(-1000., -2000.)), 0);
      EXPECT_EQ(axis.FindFixBin(-1000.), 1);
      EXPECT_EQ(axis.FindFixBin(std::nextafter(-1000., 0)), 1);

      EXPECT_EQ(axis.FindFixBin(-500.00000000001), 500);
      EXPECT_EQ(axis.FindFixBin(-500.), 501);
      EXPECT_EQ(axis.FindFixBin(-499.9), 501);

      EXPECT_EQ(axis.FindFixBin(-1.E-20), 1000);
      EXPECT_EQ(axis.FindFixBin(std::nextafter(-0., -1)), 1000);
      EXPECT_EQ(axis.FindFixBin(-0.), 1001);
      EXPECT_EQ(axis.FindFixBin(0.), 1001);

      EXPECT_EQ(axis.FindFixBin(499.9), 1500);
      EXPECT_EQ(axis.FindFixBin(500.), 1501);

      EXPECT_EQ(axis.FindFixBin(1000. - 1.E-13), 2000);
      EXPECT_EQ(axis.FindFixBin(std::nextafter(1000., 0.)), 2000);
      EXPECT_EQ(axis.FindFixBin(1000.), 2001);

      EXPECT_EQ(axis.FindBin(std::nextafter(-1000., -2000.)), 0);
      EXPECT_EQ(axis.FindBin(-1000.), 1);
      EXPECT_EQ(axis.FindBin(std::nextafter(-1000., 0)), 1);

      EXPECT_EQ(axis.FindBin(-500.00000000001), 500);
      EXPECT_EQ(axis.FindBin(-500.), 501);
      EXPECT_EQ(axis.FindBin(-499.9), 501);

      EXPECT_EQ(axis.FindBin(-1.E-20), 1000);
      EXPECT_EQ(axis.FindBin(std::nextafter(-0., -1)), 1000);
      EXPECT_EQ(axis.FindBin(-0.), 1001);
      EXPECT_EQ(axis.FindBin(0.), 1001);

      EXPECT_EQ(axis.FindBin(499.9), 1500);
      EXPECT_EQ(axis.FindBin(500.), 1501);

      EXPECT_EQ(axis.FindBin(1000. - 1.E-13), 2000);
      EXPECT_EQ(axis.FindBin(std::nextafter(1000., 0.)), 2000);
      EXPECT_EQ(axis.FindBin(1000.), 2001);
   }

   for (const auto &[low, high] : std::initializer_list<std::pair<long double, long double>>{{-10654.1l, 32165.l},
                                                                                             {-1.0656E23l, -20654.l},
                                                                                             {1.1234E4l, 4.5678E20l},
                                                                                             {1.E-60l, 1.E-20l},
                                                                                             {-1.E-20l, -1.E-60l}}) {
      constexpr int N = 100;
      const double width = (high - low) / N;
      std::mt19937 gen;
      std::uniform_real_distribution dist{low - width, high + width};
      TAxis axis(N, low, high);
      for (unsigned int i = 0; i < 100000; ++i) {
         long double x = dist(gen);
         if (i == 0) {
            x = std::nextafter(double(high), double(low));
         }
         const auto bin = axis.FindFixBin(x);
         EXPECT_EQ(bin, axis.FindBin(x));

         if (x < double(low)) {
            EXPECT_EQ(bin, 0);
         } else if (x >= double(high)) {
            EXPECT_EQ(bin, N + 1);
         }

         EXPECT_LE(axis.GetBinLowEdge(bin), x);
         EXPECT_LT(x, axis.GetBinUpEdge(bin));
      }
   }
}

// The tests below are taken from https://github.com/root-project/root/pull/14105
TEST(TAxis, FindBinExact)
{
   // Test the case where bin edges are exactly represented as floating points
   TAxis ax(88, 1010, 1098);
   for (int i = 1; i <= ax.GetNbins(); i++) {
      double x = ax.GetBinLowEdge(i);
      EXPECT_EQ(i, ax.FindBin(x));
      EXPECT_EQ(i, ax.FindFixBin(x));
      x = ax.GetBinUpEdge(i);
      EXPECT_EQ(i + 1, ax.FindBin(x));
      EXPECT_EQ(i + 1, ax.FindFixBin(x));
      x -= x * std::numeric_limits<double>::epsilon();
      EXPECT_EQ(i, ax.FindBin(x));
   }
}
TEST(TAxis, FindBinApprox)
{
   TAxis ax(90, 0., 10.);
   for (int i = 1; i <= ax.GetNbins(); i++) {
      double x = ax.GetBinLowEdge(i);
      EXPECT_EQ(i, ax.FindBin(x));
      EXPECT_EQ(i, ax.FindFixBin(x));
      x = ax.GetBinUpEdge(i);
      EXPECT_EQ(i + 1, ax.FindBin(x));
      EXPECT_EQ(i + 1, ax.FindFixBin(x));
      x -= x * std::numeric_limits<double>::epsilon();
      EXPECT_EQ(i, ax.FindBin(x));
   }
}

// https://github.com/root-project/root/issues/19359
TEST(TH1, SetBufferedSumw2)
{
   // TH1::SetBuffer auto-adjusts small buffer sizes to at least 100 entries...
   static constexpr std::size_t Entries = 200;
   static constexpr double Weight = 2.0;

   TH1D h1("name", "title", 1, 0, 1);
   h1.SetBuffer(Entries);
   for (std::size_t i = 0; i < Entries; i++) {
      h1.Fill(0.5, Weight);
   }
   h1.Sumw2();

   EXPECT_FLOAT_EQ(h1.GetBinContent(1), Entries * Weight);
   EXPECT_FLOAT_EQ(h1.GetBinError(1), std::sqrt(Entries * Weight * Weight));
}
