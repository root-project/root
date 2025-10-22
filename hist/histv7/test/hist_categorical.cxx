#include "hist_test.hxx"

#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

TEST(RCategoricalAxis, Constructor)
{
   const std::vector<std::string> categories = {"a", "b", "c"};

   RCategoricalAxis axis(categories);
   EXPECT_EQ(axis.GetNNormalBins(), 3);
   EXPECT_EQ(axis.GetTotalNBins(), 4);
   EXPECT_TRUE(axis.HasOverflowBin());

   axis = RCategoricalAxis(categories, /*enableOverflowBins=*/false);
   EXPECT_EQ(axis.GetNNormalBins(), 3);
   EXPECT_EQ(axis.GetTotalNBins(), 3);
   EXPECT_FALSE(axis.HasOverflowBin());

   EXPECT_THROW(RCategoricalAxis({}), std::invalid_argument);
   EXPECT_THROW(RCategoricalAxis({"a", "a"}), std::invalid_argument);
}

TEST(RCategoricalAxis, Equality)
{
   const std::vector<std::string> categoriesA = {"a", "b", "c"};
   const std::vector<std::string> categoriesB = {"c", "b", "a"};
   const std::vector<std::string> categoriesC = {"a", "ab", "abc"};

   const RCategoricalAxis axisA(categoriesA);
   const RCategoricalAxis axisANoOverflow(categoriesA, /*enableOverflowBin=*/false);
   const RCategoricalAxis axisA2(categoriesA);
   const RCategoricalAxis axisB(categoriesB);
   const RCategoricalAxis axisC(categoriesC);

   EXPECT_TRUE(axisA == axisA);
   EXPECT_TRUE(axisA == axisA2);

   EXPECT_FALSE(axisA == axisANoOverflow);

   EXPECT_FALSE(axisA == axisB);
   EXPECT_FALSE(axisA == axisC);
   EXPECT_FALSE(axisB == axisC);
}

TEST(RCategoricalAxis, ComputeLinearizedIndex)
{
   const std::vector<std::string> categories = {"a", "b", "c"};

   const RCategoricalAxis axis(categories);
   const RCategoricalAxis axisNoOverflow(categories, /*enableOverflowBin=*/false);

   for (std::size_t i = 0; i < categories.size(); i++) {
      auto linIndex = axis.ComputeLinearizedIndex(categories[i]);
      EXPECT_EQ(linIndex.fIndex, i);
      EXPECT_TRUE(linIndex.fValid);
      linIndex = axisNoOverflow.ComputeLinearizedIndex(categories[i]);
      EXPECT_EQ(linIndex.fIndex, i);
      EXPECT_TRUE(linIndex.fValid);
   }

   // Overflow
   for (std::string overflow : {"", "d"}) {
      auto axisBin = axis.ComputeLinearizedIndex(overflow);
      EXPECT_EQ(axisBin.fIndex, 3);
      EXPECT_TRUE(axisBin.fValid);
      axisBin = axisNoOverflow.ComputeLinearizedIndex(overflow);
      EXPECT_EQ(axisBin.fIndex, 3);
      EXPECT_FALSE(axisBin.fValid);
   }
}

TEST(RCategoricalAxis, GetLinearizedIndex)
{
   const std::vector<std::string> categories = {"a", "b", "c"};

   const RCategoricalAxis axis(categories);
   const RCategoricalAxis axisNoOverflow(categories, /*enableOverflowBin=*/false);

   {
      const auto underflow = RBinIndex::Underflow();
      auto linIndex = axis.GetLinearizedIndex(underflow);
      EXPECT_FALSE(linIndex.fValid);
      linIndex = axisNoOverflow.GetLinearizedIndex(underflow);
      EXPECT_FALSE(linIndex.fValid);
   }

   for (std::size_t i = 0; i < categories.size(); i++) {
      auto linIndex = axis.GetLinearizedIndex(i);
      EXPECT_EQ(linIndex.fIndex, i);
      EXPECT_TRUE(linIndex.fValid);
      linIndex = axisNoOverflow.GetLinearizedIndex(i);
      EXPECT_EQ(linIndex.fIndex, i);
      EXPECT_TRUE(linIndex.fValid);
   }

   // Out of bounds
   {
      auto linIndex = axis.GetLinearizedIndex(categories.size());
      EXPECT_EQ(linIndex.fIndex, categories.size());
      EXPECT_FALSE(linIndex.fValid);
      linIndex = axisNoOverflow.GetLinearizedIndex(categories.size());
      EXPECT_EQ(linIndex.fIndex, categories.size());
      EXPECT_FALSE(linIndex.fValid);
   }

   {
      const auto overflow = RBinIndex::Overflow();
      auto linIndex = axis.GetLinearizedIndex(overflow);
      EXPECT_TRUE(linIndex.fValid);
      EXPECT_EQ(linIndex.fIndex, categories.size());
      linIndex = axisNoOverflow.GetLinearizedIndex(overflow);
      EXPECT_FALSE(linIndex.fValid);
   }

   {
      const RBinIndex invalid;
      auto linIndex = axis.GetLinearizedIndex(invalid);
      EXPECT_FALSE(linIndex.fValid);
      linIndex = axisNoOverflow.GetLinearizedIndex(invalid);
      EXPECT_FALSE(linIndex.fValid);
   }
}

TEST(RCategoricalAxis, GetNormalRange)
{
   const std::vector<std::string> categories = {"a", "b", "c"};

   const RCategoricalAxis axis(categories);
   const auto index0 = RBinIndex(0);
   const auto index1 = RBinIndex(1);
   const auto indexBins = RBinIndex(categories.size());

   {
      const auto normal = axis.GetNormalRange();
      EXPECT_EQ(normal.GetBegin(), index0);
      EXPECT_EQ(normal.GetEnd(), indexBins);
      EXPECT_EQ(std::distance(normal.begin(), normal.end()), categories.size());
   }

   {
      const auto normal = axis.GetNormalRange(index0, indexBins);
      EXPECT_EQ(normal.GetBegin(), index0);
      EXPECT_EQ(normal.GetEnd(), indexBins);
      EXPECT_EQ(std::distance(normal.begin(), normal.end()), categories.size());
   }

   {
      const auto index2 = RBinIndex(2);
      const auto normal = axis.GetNormalRange(index1, index2);
      EXPECT_EQ(normal.GetBegin(), index1);
      EXPECT_EQ(normal.GetEnd(), index2);
      EXPECT_EQ(std::distance(normal.begin(), normal.end()), 1);
   }

   {
      const auto empty = axis.GetNormalRange(index1, index1);
      EXPECT_EQ(empty.GetBegin(), index1);
      EXPECT_EQ(empty.GetEnd(), index1);
      EXPECT_EQ(empty.begin(), empty.end());
      EXPECT_EQ(std::distance(empty.begin(), empty.end()), 0);
   }

   const auto underflow = RBinIndex::Underflow();
   const auto overflow = RBinIndex::Overflow();
   EXPECT_THROW(axis.GetNormalRange(underflow, index0), std::invalid_argument);
   EXPECT_THROW(axis.GetNormalRange(indexBins, indexBins), std::invalid_argument);
   EXPECT_THROW(axis.GetNormalRange(index0, overflow), std::invalid_argument);
   EXPECT_THROW(axis.GetNormalRange(index0, indexBins + 1), std::invalid_argument);
   EXPECT_THROW(axis.GetNormalRange(index1, index0), std::invalid_argument);
}

TEST(RCategoricalAxis, GetFullRange)
{
   const std::vector<std::string> categories = {"a", "b", "c"};

   {
      const RCategoricalAxis axis(categories);
      const auto full = axis.GetFullRange();
      EXPECT_EQ(full.GetBegin(), RBinIndex(0));
      EXPECT_EQ(full.GetEnd(), RBinIndex());
      EXPECT_EQ(std::distance(full.begin(), full.end()), categories.size() + 1);
   }

   {
      const RCategoricalAxis axisNoOverflow(categories, /*enableOverflowBin=*/false);
      const auto full = axisNoOverflow.GetFullRange();
      EXPECT_EQ(full.GetBegin(), RBinIndex(0));
      EXPECT_EQ(full.GetEnd(), RBinIndex(categories.size()));
      EXPECT_EQ(std::distance(full.begin(), full.end()), categories.size());
   }
}
