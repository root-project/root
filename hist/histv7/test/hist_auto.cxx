#include "hist_test.hxx"

#include <limits>
#include <stdexcept>
#include <utility>

TEST(RHistAutoAxisFiller, Constructor)
{
   static constexpr std::size_t Bins = 20;
   RHistAutoAxisFiller<int> filler(Bins);
   EXPECT_EQ(filler.GetNNormalBins(), Bins);
   EXPECT_EQ(filler.GetMaxBufferSize(), 1024);
   EXPECT_EQ(filler.GetMarginFraction(), 0.05);

   EXPECT_THROW(RHistAutoAxisFiller<int>(0), std::invalid_argument);
   EXPECT_THROW(RHistAutoAxisFiller<int>(1, 0), std::invalid_argument);
   EXPECT_THROW(RHistAutoAxisFiller<int>(1, 1024, -1), std::invalid_argument);
}

TEST(RHistAutoAxisFiller, Fill)
{
   static constexpr std::size_t Bins = 20;
   RHistAutoAxisFiller<int> filler(Bins);

   // Fill some entries
   for (std::size_t i = 0; i < Bins; i++) {
      filler.Fill(i);
   }

   // NaN should be ignored for the axis interval
   filler.Fill(std::numeric_limits<double>::quiet_NaN());

   // Get the histogram, which first flushes the buffer
   auto &hist = filler.GetHist();
   auto &axis = std::get<RRegularAxis>(hist.GetAxes()[0]);
   EXPECT_EQ(axis.GetNNormalBins(), Bins);
   EXPECT_TRUE(axis.HasFlowBins());
   EXPECT_LT(axis.GetLow(), 0);
   EXPECT_GT(axis.GetHigh(), Bins - 1);

   EXPECT_EQ(hist.GetNEntries(), Bins + 1);
   EXPECT_EQ(hist.GetBinContent(RBinIndex::Underflow()), 0);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist.GetBinContent(index), 1);
   }
   // The NaN entry
   EXPECT_EQ(hist.GetBinContent(RBinIndex::Overflow()), 1);

   // Fill some more entries that are now directly forwarded to the histogram
   for (std::size_t i = 0; i < Bins; i++) {
      filler.Fill(i);
   }
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(hist.GetBinContent(index), 2);
   }
}

TEST(RHistAutoAxisFiller, FillAutoFlush)
{
   static constexpr std::size_t Bins = 20;
   RHistAutoAxisFiller<int> filler(Bins);

   // Fill entries so that it triggers auto-flushing
   for (std::size_t i = 0; i < 1024; i++) {
      filler.Fill(i);
   }

   // Further entries may land in the flow bins
   filler.Fill(-1000);
   filler.Fill(2000);

   auto &hist = filler.GetHist();
   EXPECT_EQ(hist.GetBinContent(RBinIndex::Underflow()), 1);
   EXPECT_EQ(hist.GetBinContent(RBinIndex::Overflow()), 1);
}

TEST(RHistAutoAxisFiller, FillMax0)
{
   static constexpr std::size_t Bins = 20;
   RHistAutoAxisFiller<int> filler(Bins);

   filler.Fill(-1);
   filler.Fill(0);

   auto &hist = filler.GetHist();
   EXPECT_EQ(hist.GetBinContent(RBinIndex::Underflow()), 0);
   EXPECT_EQ(hist.GetBinContent(RBinIndex::Overflow()), 0);
}

TEST(RHistAutoAxisFiller, FillFloat)
{
   RHistAutoAxisFiller<float> filler(3);

   filler.Fill(1);
   filler.Fill(2);

   auto &hist = filler.GetHist();
   EXPECT_EQ(hist.GetBinContent(0), 1);
   EXPECT_EQ(hist.GetBinContent(2), 1);

   filler.Fill(1.5);
   EXPECT_EQ(hist.GetBinContent(1), 1);
}

TEST(RHistAutoAxisFiller, FillWeight)
{
   RHistAutoAxisFiller<float> filler(3);

   filler.Fill(1, RWeight(0.8));
   filler.Fill(2, RWeight(0.9));

   auto &hist = filler.GetHist();
   EXPECT_FLOAT_EQ(hist.GetBinContent(0), 0.8);
   EXPECT_FLOAT_EQ(hist.GetBinContent(2), 0.9);

   filler.Fill(1.5, RWeight(0.85));
   EXPECT_FLOAT_EQ(hist.GetBinContent(1), 0.85);
}

TEST(RHistAutoAxisFiller, FlushError)
{
   static constexpr std::size_t Bins = 20;

   {
      RHistAutoAxisFiller<int> filler(Bins);
      // Flush without entries
      EXPECT_THROW(filler.Flush(), std::runtime_error);
   }

   {
      RHistAutoAxisFiller<int> filler(Bins);
      // NaN should be ignored for the axis interval
      filler.Fill(std::numeric_limits<double>::quiet_NaN());
      EXPECT_THROW(filler.Flush(), std::runtime_error);
   }

   {
      RHistAutoAxisFiller<int> filler(Bins);
      // Fill with infinities
      filler.Fill(std::numeric_limits<double>::infinity());
      filler.Fill(-std::numeric_limits<double>::infinity());
      EXPECT_THROW(filler.Flush(), std::runtime_error);
   }

   {
      RHistAutoAxisFiller<int> filler(Bins);
      // Fill with identical values
      filler.Fill(1);
      filler.Fill(1);
      EXPECT_THROW(filler.Flush(), std::runtime_error);
   }
}

TEST(RHistAutoAxisFiller, GetHist)
{
   static constexpr std::size_t Bins = 20;
   RHistAutoAxisFiller<int> filler(Bins);

   filler.Fill(0);
   filler.Fill(1);

   // The histogram can be moved out of the filler that constructed it.
   RHist<int> hist(std::move(filler.GetHist()));
}
