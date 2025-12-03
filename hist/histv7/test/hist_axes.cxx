#include "hist_test.hxx"

#include <array>
#include <stdexcept>
#include <tuple>
#include <variant>
#include <vector>

TEST(RAxes, Constructor)
{
   static constexpr std::size_t BinsX = 20;
   const RRegularAxis regularAxis(BinsX, {0, BinsX});
   static constexpr std::size_t BinsY = 30;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis categoricalAxis(categories);

   RAxes axes({regularAxis, variableBinAxis, categoricalAxis});
   EXPECT_EQ(axes.GetNDimensions(), 3);
   const auto &v = axes.Get();
   ASSERT_EQ(v.size(), 3);
   EXPECT_EQ(v[0].index(), 0);
   EXPECT_EQ(v[1].index(), 1);
   EXPECT_EQ(v[2].index(), 2);
   EXPECT_TRUE(std::get_if<RRegularAxis>(&v[0]) != nullptr);
   EXPECT_TRUE(std::get_if<RVariableBinAxis>(&v[1]) != nullptr);
   EXPECT_TRUE(std::get_if<RCategoricalAxis>(&v[2]) != nullptr);

   std::vector<RAxisVariant> newAxes{categoricalAxis, variableBinAxis, regularAxis};
   axes = RAxes(newAxes);
   EXPECT_EQ(axes.GetNDimensions(), 3);

   EXPECT_THROW(RAxes({}), std::invalid_argument);
}

TEST(RAxes, Equality)
{
   static constexpr std::size_t BinsX = 20;
   const RRegularAxis regularAxis(BinsX, {0, BinsX});
   static constexpr std::size_t BinsY = 30;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis categoricalAxis(categories);

   const RAxes axesA({regularAxis, variableBinAxis, categoricalAxis});
   const RAxes axesA2({regularAxis, variableBinAxis, categoricalAxis});
   const RAxes axesB({categoricalAxis, variableBinAxis, regularAxis});
   const RAxes axesC({regularAxis});
   const RAxes axesD({variableBinAxis});
   const RAxes axesE({categoricalAxis});

   EXPECT_TRUE(axesA == axesA);
   EXPECT_TRUE(axesA == axesA2);
   EXPECT_TRUE(axesA2 == axesA);

   EXPECT_FALSE(axesA == axesB);
   EXPECT_FALSE(axesA == axesC);
   EXPECT_FALSE(axesA == axesD);
   EXPECT_FALSE(axesA == axesE);

   EXPECT_FALSE(axesB == axesC);
   EXPECT_FALSE(axesB == axesD);
   EXPECT_FALSE(axesB == axesE);

   EXPECT_FALSE(axesC == axesD);
   EXPECT_FALSE(axesC == axesE);

   EXPECT_FALSE(axesD == axesE);
}

TEST(RAxes, ComputeTotalNBins)
{
   static constexpr std::size_t BinsX = 20;
   const RRegularAxis regularAxis(BinsX, {0, BinsX});
   static constexpr std::size_t BinsY = 30;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis categoricalAxis(categories);
   const RAxes axes({regularAxis, variableBinAxis, categoricalAxis});

   // All axes include underflow and overflow bins.
   EXPECT_EQ(axes.ComputeTotalNBins(), (BinsX + 2) * (BinsY + 2) * (categories.size() + 1));
}

TEST(RAxes, ComputeGlobalIndex)
{
   static constexpr std::size_t BinsX = 20;
   const RRegularAxis regularAxis(BinsX, {0, BinsX});
   static constexpr std::size_t BinsY = 30;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis categoricalAxis(categories);
   const RAxes axes({regularAxis, variableBinAxis, categoricalAxis});

   {
      auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(1.5, 2.5, "c"));
      EXPECT_EQ(globalIndex.fIndex, (1 * (BinsY + 2) + 2) * (categories.size() + 1) + 2);
      EXPECT_TRUE(globalIndex.fValid);
      const std::array<RBinIndex, 3> indices = {1, 2, 2};
      globalIndex = axes.ComputeGlobalIndex(indices);
      EXPECT_EQ(globalIndex.fIndex, (1 * (BinsY + 2) + 2) * (categories.size() + 1) + 2);
      EXPECT_TRUE(globalIndex.fValid);
   }

   {
      // Underflow bin of the first axis.
      auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(-1, 2.5, "c"));
      EXPECT_EQ(globalIndex.fIndex, (BinsX * (BinsY + 2) + 2) * (categories.size() + 1) + 2);
      EXPECT_TRUE(globalIndex.fValid);
      const std::array<RBinIndex, 3> indices = {RBinIndex::Underflow(), 2, 2};
      globalIndex = axes.ComputeGlobalIndex(indices);
      EXPECT_EQ(globalIndex.fIndex, (BinsX * (BinsY + 2) + 2) * (categories.size() + 1) + 2);
      EXPECT_TRUE(globalIndex.fValid);
   }

   {
      // Overflow bin of the second axis.
      auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(1.5, 42, "c"));
      EXPECT_EQ(globalIndex.fIndex, (1 * (BinsY + 2) + BinsY + 1) * (categories.size() + 1) + 2);
      EXPECT_TRUE(globalIndex.fValid);
      const std::array<RBinIndex, 3> indices = {1, RBinIndex::Overflow(), 2};
      globalIndex = axes.ComputeGlobalIndex(indices);
      EXPECT_EQ(globalIndex.fIndex, (1 * (BinsY + 2) + BinsY + 1) * (categories.size() + 1) + 2);
      EXPECT_TRUE(globalIndex.fValid);
   }

   {
      // Overflow bin of the third axis.
      auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(1.5, 2.5, "d"));
      EXPECT_EQ(globalIndex.fIndex, (1 * (BinsY + 2) + 2) * (categories.size() + 1) + categories.size());
      EXPECT_TRUE(globalIndex.fValid);
      const std::array<RBinIndex, 3> indices = {1, 2, RBinIndex::Overflow()};
      globalIndex = axes.ComputeGlobalIndex(indices);
      EXPECT_EQ(globalIndex.fIndex, (1 * (BinsY + 2) + 2) * (categories.size() + 1) + categories.size());
      EXPECT_TRUE(globalIndex.fValid);
   }
}

TEST(RAxes, ComputeGlobalIndexNoFlowBins)
{
   static constexpr std::size_t BinsX = 20;
   const RRegularAxis regularAxis(BinsX, {0, BinsX}, /*enableFlowBins=*/false);
   static constexpr std::size_t BinsY = 30;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins, /*enableFlowBins=*/false);
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis categoricalAxis(categories, /*enableOverflowBin=*/false);
   const RAxes axes({regularAxis, variableBinAxis, categoricalAxis});
   ASSERT_EQ(axes.ComputeTotalNBins(), BinsX * BinsY * categories.size());

   {
      auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(1.5, 2.5, "c"));
      EXPECT_EQ(globalIndex.fIndex, (1 * BinsY + 2) * categories.size() + 2);
      EXPECT_TRUE(globalIndex.fValid);
      const std::array<RBinIndex, 3> indices = {1, 2, 2};
      globalIndex = axes.ComputeGlobalIndex(indices);
      EXPECT_EQ(globalIndex.fIndex, (1 * BinsY + 2) * categories.size() + 2);
      EXPECT_TRUE(globalIndex.fValid);
   }

   {
      // Underflow bin of the first axis.
      auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(-1, 2.5, "c"));
      EXPECT_EQ(globalIndex.fIndex, 0);
      EXPECT_FALSE(globalIndex.fValid);
      const std::array<RBinIndex, 3> indices = {RBinIndex::Underflow(), 2, 2};
      globalIndex = axes.ComputeGlobalIndex(indices);
      EXPECT_EQ(globalIndex.fIndex, 0);
      EXPECT_FALSE(globalIndex.fValid);
   }

   {
      // Overflow bin of the second axis.
      auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(1.5, 42, "c"));
      EXPECT_EQ(globalIndex.fIndex, 0);
      EXPECT_FALSE(globalIndex.fValid);
      const std::array<RBinIndex, 3> indices = {1, RBinIndex::Overflow(), 2};
      globalIndex = axes.ComputeGlobalIndex(indices);
      EXPECT_EQ(globalIndex.fIndex, 0);
      EXPECT_FALSE(globalIndex.fValid);
   }

   {
      // Overflow bin of the third axis.
      auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(1.5, 2.5, "d"));
      EXPECT_EQ(globalIndex.fIndex, 0);
      EXPECT_FALSE(globalIndex.fValid);
      const std::array<RBinIndex, 3> indices = {1, 2, RBinIndex::Overflow()};
      globalIndex = axes.ComputeGlobalIndex(indices);
      EXPECT_EQ(globalIndex.fIndex, 0);
      EXPECT_FALSE(globalIndex.fValid);
   }
}

TEST(RAxes, ComputeGlobalIndexInvalidNumberOfArguments)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RAxes axes1({axis});
   ASSERT_EQ(axes1.GetNDimensions(), 1);
   const RAxes axes2({axis, axis});
   ASSERT_EQ(axes2.GetNDimensions(), 2);

   EXPECT_NO_THROW(axes1.ComputeGlobalIndex(std::make_tuple(1)));
   EXPECT_THROW(axes1.ComputeGlobalIndex(std::make_tuple(1, 2)), std::invalid_argument);

   EXPECT_THROW(axes2.ComputeGlobalIndex(std::make_tuple(1)), std::invalid_argument);
   EXPECT_NO_THROW(axes2.ComputeGlobalIndex(std::make_tuple(1, 2)));
   EXPECT_THROW(axes2.ComputeGlobalIndex(std::make_tuple(1, 2, 3)), std::invalid_argument);

   const std::array<RBinIndex, 1> indices1 = {1};
   const std::array<RBinIndex, 2> indices2 = {1, 2};
   const std::array<RBinIndex, 3> indices3 = {1, 2, 3};

   EXPECT_NO_THROW(axes1.ComputeGlobalIndex(indices1));
   EXPECT_THROW(axes1.ComputeGlobalIndex(indices2), std::invalid_argument);

   EXPECT_THROW(axes2.ComputeGlobalIndex(indices1), std::invalid_argument);
   EXPECT_NO_THROW(axes2.ComputeGlobalIndex(indices2));
   EXPECT_THROW(axes2.ComputeGlobalIndex(indices3), std::invalid_argument);
}

TEST(RAxes, ComputeGlobalIndexInvalidArgumentType)
{
   static constexpr std::size_t BinsX = 20;
   const RRegularAxis regularAxis(BinsX, {0, BinsX});
   static constexpr std::size_t BinsY = 30;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis categoricalAxis(categories);
   const RAxes axes({regularAxis, variableBinAxis, categoricalAxis});

   EXPECT_NO_THROW(axes.ComputeGlobalIndex(std::make_tuple(1, 2, "a")));
   EXPECT_THROW(axes.ComputeGlobalIndex(std::make_tuple("1", 2, "a")), std::invalid_argument);
   EXPECT_THROW(axes.ComputeGlobalIndex(std::make_tuple(1, "2", "a")), std::invalid_argument);
   EXPECT_THROW(axes.ComputeGlobalIndex(std::make_tuple(1, 2, 3)), std::invalid_argument);
}
