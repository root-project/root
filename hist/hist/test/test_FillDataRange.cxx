#include "gtest/gtest.h"

#include "Fit/BinData.h"
#include "Fit/DataRange.h"
#include "Fit/Fitter.h"
#include "HFitInterface.h"
#include "Math/WrappedMultiTF1.h"
#include "TF1.h"
#include "TH1.h"
#include "TH2.h"

#include <algorithm>
#include <vector>

namespace {

// linearly increasing bin contents, so that every bin is distinguishable
std::unique_ptr<TH1F> MakeH1()
{
   auto h = std::make_unique<TH1F>("h1_filldata", "h1", 100, -10., 10.);
   for (int i = 1; i <= 100; ++i)
      h->SetBinContent(i, 1. + 0.01 * i);
   return h;
}

std::vector<double> CollectedCoords(const ROOT::Fit::BinData &data)
{
   std::vector<double> x;
   x.reserve(data.Size());
   for (unsigned int i = 0; i < data.Size(); ++i)
      x.push_back(*data.Coords(i));
   return x;
}

} // namespace

// Without any range the whole axis must be used. This is the default path of
// TH1::Fit and must not be affected by the multiple-range support.
TEST(FillDataRange, NoRange)
{
   auto h = MakeH1();

   ROOT::Fit::DataOptions opt;
   ROOT::Fit::DataRange range;
   ROOT::Fit::BinData data(opt, range);
   ROOT::Fit::FillData(data, h.get());

   EXPECT_EQ(data.Size(), 100u);
}

// A single range behaves as before.
TEST(FillDataRange, SingleRange)
{
   auto h = MakeH1();

   ROOT::Fit::DataOptions opt;
   ROOT::Fit::DataRange range;
   range.AddRange(0, -10., -5.); // bins 1..25
   ROOT::Fit::BinData data(opt, range);
   ROOT::Fit::FillData(data, h.get());

   EXPECT_EQ(data.Size(), 25u);
   for (double x : CollectedCoords(data)) {
      EXPECT_GE(x, -10.);
      EXPECT_LE(x, -5.);
   }
}

// Two disjoint ranges must both contribute, and nothing outside them.
TEST(FillDataRange, TwoDisjointRanges)
{
   auto h = MakeH1();

   ROOT::Fit::DataOptions opt;
   ROOT::Fit::DataRange range;
   range.AddRange(0, -10., -5.); // bins 1..25
   range.AddRange(0, 5., 10.);   // bins 76..100
   ROOT::Fit::BinData data(opt, range);
   ROOT::Fit::FillData(data, h.get());

   EXPECT_EQ(data.Size(), 50u);
   for (double x : CollectedCoords(data))
      EXPECT_TRUE((x >= -10. && x <= -5.) || (x >= 5. && x <= 10.));
}

// Overlapping ranges are merged by DataRange, so no bin is counted twice.
TEST(FillDataRange, OverlappingRangesAreNotDoubleCounted)
{
   auto h = MakeH1();

   ROOT::Fit::DataOptions opt;
   ROOT::Fit::DataRange range;
   range.AddRange(0, -10., 0.);
   range.AddRange(0, -5., 5.); // overlaps the previous one
   ROOT::Fit::BinData data(opt, range);
   ROOT::Fit::FillData(data, h.get());

   ASSERT_EQ(range.Size(0), 1u); // merged into [-10,5]
   EXPECT_EQ(data.Size(), 75u);

   auto x = CollectedCoords(data);
   EXPECT_EQ(std::adjacent_find(x.begin(), x.end()), x.end()); // no duplicates
}

// Two ranges along both axes of a 2D histogram give the product of the bins.
TEST(FillDataRange, TwoDimensions)
{
   TH2F h("h2_filldata", "h2", 10, 0., 10., 10, 0., 10.);
   for (int i = 1; i <= 10; ++i)
      for (int j = 1; j <= 10; ++j)
         h.SetBinContent(i, j, 1. + i + 10. * j);

   ROOT::Fit::DataOptions opt;
   ROOT::Fit::DataRange range;
   range.AddRange(0, 0., 2.); // x bins 1..2
   range.AddRange(0, 8., 10.);// x bins 9..10
   range.AddRange(1, 0., 3.); // y bins 1..3
   ROOT::Fit::BinData data(opt, range);
   ROOT::Fit::FillData(data, &h);

   EXPECT_EQ(data.Size(), 4u * 3u);
}

// End-to-end: fit a linear background over two ranges chosen to exclude a peak.
TEST(FillDataRange, BackgroundFitExcludingPeak)
{
   TF1 comb("comb_filldata", "[0] + x*[1] + [2]*exp(-((x-[3])**2)/[4])", -10., 10.);
   comb.SetParameters(10., -0.5, 15., 1., 1.5);

   TH1F h("h1_peak", "h1", 100, -10., 10.);
   for (int i = 1; i <= 100; ++i)
      h.SetBinContent(i, comb.Eval(h.GetBinCenter(i)));

   ROOT::Fit::DataOptions opt;
   ROOT::Fit::DataRange range;
   range.AddRange(0, -10., -2.5);
   range.AddRange(0, 4.5, 10.);
   ROOT::Fit::BinData data(opt, range);
   ROOT::Fit::FillData(data, &h);

   TF1 back("back_filldata", "[0] + x*[1]", -10., 10.);
   back.SetParameters(1., 0.);
   ROOT::Math::WrappedMultiTF1 fitFunc(back, back.GetNdim());
   ROOT::Fit::Fitter fitter;
   fitter.SetFunction(fitFunc, false);

   ASSERT_TRUE(fitter.Fit(data));
   EXPECT_NEAR(fitter.Result().Parameter(0), 10., 1e-3);
   EXPECT_NEAR(fitter.Result().Parameter(1), -0.5, 1e-3);
}
