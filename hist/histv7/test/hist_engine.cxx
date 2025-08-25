#include "hist_test.hxx"

#include <stdexcept>
#include <variant>
#include <vector>

TEST(RHistEngine, Constructor)
{
   static constexpr std::size_t BinsX = 20;
   const RRegularAxis regularAxis(BinsX, 0, BinsX);
   static constexpr std::size_t BinsY = 30;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   RHistEngine<int> engine({regularAxis, variableBinAxis});
   EXPECT_EQ(engine.GetNDimensions(), 2);
   const auto &axes = engine.GetAxes();
   ASSERT_EQ(axes.size(), 2);
   EXPECT_EQ(axes[0].index(), 0);
   EXPECT_EQ(axes[1].index(), 1);
   EXPECT_TRUE(std::get_if<RRegularAxis>(&axes[0]) != nullptr);
   EXPECT_TRUE(std::get_if<RVariableBinAxis>(&axes[1]) != nullptr);

   // Both axes include underflow and overflow bins.
   EXPECT_EQ(engine.GetTotalNBins(), (BinsX + 2) * (BinsY + 2));

   engine = RHistEngine<int>(BinsX, 0, BinsX);
   ASSERT_EQ(engine.GetNDimensions(), 1);
   auto *regular = std::get_if<RRegularAxis>(&engine.GetAxes()[0]);
   ASSERT_TRUE(regular != nullptr);
   EXPECT_EQ(regular->GetNNormalBins(), BinsX);
   EXPECT_EQ(regular->GetLow(), 0);
   EXPECT_EQ(regular->GetHigh(), BinsX);
}

TEST(RHistEngine, GetBinContentInvalidNumberOfArguments)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins);
   const RHistEngine<int> engine1({axis});
   ASSERT_EQ(engine1.GetNDimensions(), 1);
   const RHistEngine<int> engine2({axis, axis});
   ASSERT_EQ(engine2.GetNDimensions(), 2);

   EXPECT_NO_THROW(engine1.GetBinContent(1));
   EXPECT_THROW(engine1.GetBinContent(1, 2), std::invalid_argument);

   EXPECT_THROW(engine2.GetBinContent(1), std::invalid_argument);
   EXPECT_NO_THROW(engine2.GetBinContent(1, 2));
   EXPECT_THROW(engine2.GetBinContent(1, 2, 3), std::invalid_argument);
}

TEST(RHistEngine, GetBinContentNotFound)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins);
   const RHistEngine<int> engine({axis});

   EXPECT_THROW(engine.GetBinContent(Bins), std::invalid_argument);
}

TEST(RHistEngine, Fill)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins);
   RHistEngine<int> engine({axis});

   engine.Fill(-100);
   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(i);
   }
   engine.Fill(100);

   EXPECT_EQ(engine.GetBinContent(RBinIndex::Underflow()), 1);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(engine.GetBinContent(index), 1);
   }
   EXPECT_EQ(engine.GetBinContent(RBinIndex::Overflow()), 1);
}

TEST(RHistEngine, FillDiscard)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins, /*enableFlowBins=*/false);
   RHistEngine<int> engine({axis});

   engine.Fill(-100);
   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(i);
   }
   engine.Fill(100);

   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(engine.GetBinContent(index), 1);
   }
}

TEST(RHistEngine, FillOnlyInner)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins);
   RHistEngine<int> engine({axis});
   const RRegularAxis axisNoFlowBins(Bins, 0, Bins, /*enableFlowBins=*/false);
   RHistEngine<int> engineNoFlowBins({axisNoFlowBins});

   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(i);
      engineNoFlowBins.Fill(i);
   }

   EXPECT_EQ(engine.GetBinContent(RBinIndex::Underflow()), 0);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(engine.GetBinContent(index), 1);
      EXPECT_EQ(engineNoFlowBins.GetBinContent(index), 1);
   }
   EXPECT_EQ(engine.GetBinContent(RBinIndex::Overflow()), 0);
}

TEST(RHistEngine, FillTuple)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins);
   RHistEngine<int> engine({axis});

   engine.Fill(std::make_tuple(-100));
   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(std::make_tuple(i));
   }
   engine.Fill(std::make_tuple(100));

   std::array<RBinIndex, 1> indices = {RBinIndex::Underflow()};
   EXPECT_EQ(engine.GetBinContent(indices), 1);
   for (auto index : axis.GetNormalRange()) {
      indices[0] = index;
      EXPECT_EQ(engine.GetBinContent(indices), 1);
   }
   indices[0] = RBinIndex::Overflow();
   EXPECT_EQ(engine.GetBinContent(indices), 1);
}

TEST(RHistEngine, FillInvalidNumberOfArguments)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins);
   RHistEngine<int> engine1({axis});
   ASSERT_EQ(engine1.GetNDimensions(), 1);
   RHistEngine<int> engine2({axis, axis});
   ASSERT_EQ(engine2.GetNDimensions(), 2);

   EXPECT_NO_THROW(engine1.Fill(1));
   EXPECT_THROW(engine1.Fill(1, 2), std::invalid_argument);

   EXPECT_THROW(engine2.Fill(1), std::invalid_argument);
   EXPECT_NO_THROW(engine2.Fill(1, 2));
   EXPECT_THROW(engine2.Fill(1, 2, 3), std::invalid_argument);
}
