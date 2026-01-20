#include "hist_test.hxx"

#include <iterator>
#include <string>
#include <vector>

TEST(RAxisVariant, RegularAxis)
{
   static constexpr std::size_t Bins = 20;
   {
      const RAxisVariant axis{RRegularAxis(Bins, {0, Bins})};
      EXPECT_EQ(axis.GetVariant().index(), 0);
      EXPECT_TRUE(axis.GetRegularAxis() != nullptr);
      EXPECT_TRUE(axis.GetVariableBinAxis() == nullptr);
      EXPECT_TRUE(axis.GetCategoricalAxis() == nullptr);

      EXPECT_EQ(axis.GetNNormalBins(), Bins);
      EXPECT_EQ(axis.GetTotalNBins(), Bins + 2);

      const auto normal = axis.GetNormalRange();
      EXPECT_EQ(std::distance(normal.begin(), normal.end()), Bins);
      const auto normal12 = axis.GetNormalRange(1, 2);
      EXPECT_EQ(std::distance(normal12.begin(), normal12.end()), 1);
      const auto full = axis.GetFullRange();
      EXPECT_EQ(std::distance(full.begin(), full.end()), Bins + 2);
   }

   {
      const RAxisVariant axis{RRegularAxis(Bins, {0, Bins}, /*enableFlowBins=*/false)};
      EXPECT_EQ(axis.GetNNormalBins(), Bins);
      EXPECT_EQ(axis.GetTotalNBins(), Bins);

      const auto normal = axis.GetNormalRange();
      EXPECT_EQ(std::distance(normal.begin(), normal.end()), Bins);
      const auto normal12 = axis.GetNormalRange(1, 2);
      EXPECT_EQ(std::distance(normal12.begin(), normal12.end()), 1);
      const auto full = axis.GetFullRange();
      EXPECT_EQ(std::distance(full.begin(), full.end()), Bins);
   }
}

TEST(RAxisVariant, VariableBinAxis)
{
   static constexpr std::size_t Bins = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < Bins; i++) {
      bins.push_back(i);
   }
   bins.push_back(Bins);

   {
      const RAxisVariant axis{RVariableBinAxis(bins)};
      EXPECT_EQ(axis.GetVariant().index(), 1);
      EXPECT_TRUE(axis.GetRegularAxis() == nullptr);
      EXPECT_TRUE(axis.GetVariableBinAxis() != nullptr);
      EXPECT_TRUE(axis.GetCategoricalAxis() == nullptr);

      EXPECT_EQ(axis.GetNNormalBins(), Bins);
      EXPECT_EQ(axis.GetTotalNBins(), Bins + 2);

      const auto normal = axis.GetNormalRange();
      EXPECT_EQ(std::distance(normal.begin(), normal.end()), Bins);
      const auto normal12 = axis.GetNormalRange(1, 2);
      EXPECT_EQ(std::distance(normal12.begin(), normal12.end()), 1);
      const auto full = axis.GetFullRange();
      EXPECT_EQ(std::distance(full.begin(), full.end()), Bins + 2);
   }

   {
      const RAxisVariant axis{RVariableBinAxis(bins, /*enableFlowBins=*/false)};
      EXPECT_EQ(axis.GetNNormalBins(), Bins);
      EXPECT_EQ(axis.GetTotalNBins(), Bins);

      const auto normal = axis.GetNormalRange();
      EXPECT_EQ(std::distance(normal.begin(), normal.end()), Bins);
      const auto normal12 = axis.GetNormalRange(1, 2);
      EXPECT_EQ(std::distance(normal12.begin(), normal12.end()), 1);
      const auto full = axis.GetFullRange();
      EXPECT_EQ(std::distance(full.begin(), full.end()), Bins);
   }
}

TEST(RAxisVariant, CategoricalAxis)
{
   const std::vector<std::string> categories = {"a", "b", "c"};

   {
      const RAxisVariant axis{RCategoricalAxis(categories)};
      EXPECT_EQ(axis.GetVariant().index(), 2);
      EXPECT_TRUE(axis.GetRegularAxis() == nullptr);
      EXPECT_TRUE(axis.GetVariableBinAxis() == nullptr);
      EXPECT_TRUE(axis.GetCategoricalAxis() != nullptr);

      EXPECT_EQ(axis.GetNNormalBins(), 3);
      EXPECT_EQ(axis.GetTotalNBins(), 4);

      const auto normal = axis.GetNormalRange();
      EXPECT_EQ(std::distance(normal.begin(), normal.end()), 3);
      const auto normal12 = axis.GetNormalRange(1, 2);
      EXPECT_EQ(std::distance(normal12.begin(), normal12.end()), 1);
      const auto full = axis.GetFullRange();
      EXPECT_EQ(std::distance(full.begin(), full.end()), 4);
   }

   {
      const RAxisVariant axis{RCategoricalAxis(categories, /*enableOverflowBin=*/false)};
      EXPECT_EQ(axis.GetNNormalBins(), 3);
      EXPECT_EQ(axis.GetTotalNBins(), 3);

      const auto normal = axis.GetNormalRange();
      EXPECT_EQ(std::distance(normal.begin(), normal.end()), 3);
      const auto normal12 = axis.GetNormalRange(1, 2);
      EXPECT_EQ(std::distance(normal12.begin(), normal12.end()), 1);
      const auto full = axis.GetFullRange();
      EXPECT_EQ(std::distance(full.begin(), full.end()), 3);
   }
}
