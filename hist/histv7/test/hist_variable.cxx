#include "hist_test.hxx"

#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

TEST(RVariableBinAxis, Constructor)
{
   static constexpr std::size_t Bins = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < Bins; i++) {
      bins.push_back(i);
   }
   bins.push_back(Bins);

   RVariableBinAxis axis(bins);
   EXPECT_EQ(axis.GetNNormalBins(), Bins);
   EXPECT_EQ(axis.GetTotalNBins(), Bins + 2);
   EXPECT_TRUE(axis.HasFlowBins());

   axis = RVariableBinAxis(bins, /*enableFlowBins=*/false);
   EXPECT_EQ(axis.GetNNormalBins(), Bins);
   EXPECT_EQ(axis.GetTotalNBins(), Bins);
   EXPECT_FALSE(axis.HasFlowBins());

   EXPECT_THROW(RVariableBinAxis({}), std::invalid_argument);
   EXPECT_THROW(RVariableBinAxis({0}), std::invalid_argument);
   EXPECT_THROW(RVariableBinAxis({0, 0}), std::invalid_argument);
   EXPECT_THROW(RVariableBinAxis({0, 1, 0}), std::invalid_argument);
   EXPECT_THROW(RVariableBinAxis({0, 1, 1}), std::invalid_argument);
}

TEST(RVariableBinAxis, Equality)
{
   static constexpr std::size_t Bins = 20;
   std::vector<double> binsA;
   for (std::size_t i = 0; i < Bins; i++) {
      binsA.push_back(i);
   }
   binsA.push_back(Bins);

   std::vector<double> binsB;
   for (std::size_t i = 0; i < Bins / 2; i++) {
      binsB.push_back(i);
   }
   binsB.push_back(Bins / 2);

   std::vector<double> binsC;
   for (std::size_t i = Bins / 2; i < Bins; i++) {
      binsC.push_back(i);
   }
   binsC.push_back(Bins);

   const RVariableBinAxis axisA(binsA);
   const RVariableBinAxis axisANoFlowBins(binsA, /*enableFlowBins=*/false);
   const RVariableBinAxis axisA2(binsA);
   const RVariableBinAxis axisB(binsB);
   const RVariableBinAxis axisC(binsC);

   EXPECT_TRUE(axisA == axisA);
   EXPECT_TRUE(axisA == axisA2);
   EXPECT_TRUE(axisA2 == axisA);

   EXPECT_FALSE(axisA == axisANoFlowBins);

   EXPECT_FALSE(axisA == axisB);
   EXPECT_FALSE(axisA == axisC);
   EXPECT_FALSE(axisB == axisC);
}

TEST(RVariableBinAxis, ComputeLinearizedIndex)
{
   static constexpr std::size_t Bins = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < Bins; i++) {
      bins.push_back(i);
   }
   bins.push_back(Bins);

   const RVariableBinAxis axis(bins);
   const RVariableBinAxis axisNoFlowBins(bins, /*enableFlowBins=*/false);

   // Underflow
   static constexpr double NegativeInfinity = -std::numeric_limits<double>::infinity();
   static constexpr double UnderflowLarge = -static_cast<double>(Bins);
   static constexpr double UnderflowSmall = -0.1;
   for (double underflow : {NegativeInfinity, UnderflowLarge, UnderflowSmall}) {
      auto linIndex = axis.ComputeLinearizedIndex(underflow);
      EXPECT_EQ(linIndex.fIndex, Bins);
      EXPECT_TRUE(linIndex.fValid);
      linIndex = axisNoFlowBins.ComputeLinearizedIndex(underflow);
      EXPECT_EQ(linIndex.fIndex, Bins);
      EXPECT_FALSE(linIndex.fValid);
   }

   for (std::size_t i = 0; i < Bins; i++) {
      auto linIndex = axis.ComputeLinearizedIndex(i + 0.5);
      EXPECT_EQ(linIndex.fIndex, i);
      EXPECT_TRUE(linIndex.fValid);
      linIndex = axisNoFlowBins.ComputeLinearizedIndex(i + 0.5);
      EXPECT_EQ(linIndex.fIndex, i);
      EXPECT_TRUE(linIndex.fValid);
   }

   // Exactly on the bin edges
   for (std::size_t i = 0; i < Bins; i++) {
      auto linIndex = axis.ComputeLinearizedIndex(i);
      EXPECT_EQ(linIndex.fIndex, i);
      EXPECT_TRUE(linIndex.fValid);
      linIndex = axisNoFlowBins.ComputeLinearizedIndex(i);
      EXPECT_EQ(linIndex.fIndex, i);
      EXPECT_TRUE(linIndex.fValid);
   }

   // Exactly the upper end of the axis interval
   {
      auto linIndex = axis.ComputeLinearizedIndex(Bins);
      EXPECT_EQ(linIndex.fIndex, Bins + 1);
      EXPECT_TRUE(linIndex.fValid);
      linIndex = axisNoFlowBins.ComputeLinearizedIndex(Bins);
      EXPECT_EQ(linIndex.fIndex, Bins + 1);
      EXPECT_FALSE(linIndex.fValid);
   }

   // Overflow
   static constexpr double PositiveInfinity = std::numeric_limits<double>::infinity();
   static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
   static constexpr double OverflowLarge = static_cast<double>(Bins * 2);
   static constexpr double OverflowSmall = Bins + 0.1;
   for (double overflow : {PositiveInfinity, NaN, OverflowLarge, OverflowSmall}) {
      auto linIndex = axis.ComputeLinearizedIndex(overflow);
      EXPECT_EQ(linIndex.fIndex, Bins + 1);
      EXPECT_TRUE(linIndex.fValid);
      linIndex = axisNoFlowBins.ComputeLinearizedIndex(overflow);
      EXPECT_EQ(linIndex.fIndex, Bins + 1);
      EXPECT_FALSE(linIndex.fValid);
   }
}

TEST(RVariableBinAxis, GetLinearizedIndex)
{
   static constexpr std::size_t Bins = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < Bins; i++) {
      bins.push_back(i);
   }
   bins.push_back(Bins);

   const RVariableBinAxis axis(bins);
   const RVariableBinAxis axisNoFlowBins(bins, /*enableFlowBins=*/false);

   {
      const auto underflow = RBinIndex::Underflow();
      auto linIndex = axis.GetLinearizedIndex(underflow);
      EXPECT_EQ(linIndex.fIndex, Bins);
      EXPECT_TRUE(linIndex.fValid);
      linIndex = axisNoFlowBins.GetLinearizedIndex(underflow);
      EXPECT_EQ(linIndex.fIndex, Bins);
      EXPECT_FALSE(linIndex.fValid);
   }

   for (std::size_t i = 0; i < Bins; i++) {
      auto linIndex = axis.GetLinearizedIndex(i);
      EXPECT_EQ(linIndex.fIndex, i);
      EXPECT_TRUE(linIndex.fValid);
      linIndex = axisNoFlowBins.GetLinearizedIndex(i);
      EXPECT_EQ(linIndex.fIndex, i);
      EXPECT_TRUE(linIndex.fValid);
   }

   // Out of bounds
   {
      auto linIndex = axis.GetLinearizedIndex(Bins);
      EXPECT_EQ(linIndex.fIndex, Bins);
      EXPECT_FALSE(linIndex.fValid);
      linIndex = axisNoFlowBins.GetLinearizedIndex(Bins);
      EXPECT_EQ(linIndex.fIndex, Bins);
      EXPECT_FALSE(linIndex.fValid);
   }

   {
      const auto overflow = RBinIndex::Overflow();
      auto linIndex = axis.GetLinearizedIndex(overflow);
      EXPECT_TRUE(linIndex.fValid);
      EXPECT_EQ(linIndex.fIndex, Bins + 1);
      linIndex = axisNoFlowBins.GetLinearizedIndex(overflow);
      EXPECT_FALSE(linIndex.fValid);
   }

   {
      const RBinIndex invalid;
      auto linIndex = axis.GetLinearizedIndex(invalid);
      EXPECT_FALSE(linIndex.fValid);
      linIndex = axisNoFlowBins.GetLinearizedIndex(invalid);
      EXPECT_FALSE(linIndex.fValid);
   }
}

TEST(RVariableBinAxis, GetNormalRange)
{
   static constexpr std::size_t Bins = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < Bins; i++) {
      bins.push_back(i);
   }
   bins.push_back(Bins);

   const RVariableBinAxis axis(bins);
   const auto index0 = RBinIndex(0);
   const auto index1 = RBinIndex(1);
   const auto indexBins = RBinIndex(Bins);

   {
      const auto normal = axis.GetNormalRange();
      EXPECT_EQ(normal.GetBegin(), index0);
      EXPECT_EQ(normal.GetEnd(), indexBins);
      EXPECT_EQ(std::distance(normal.begin(), normal.end()), Bins);
   }

   {
      const auto normal = axis.GetNormalRange(index0, indexBins);
      EXPECT_EQ(normal.GetBegin(), index0);
      EXPECT_EQ(normal.GetEnd(), indexBins);
      EXPECT_EQ(std::distance(normal.begin(), normal.end()), Bins);
   }

   {
      const auto index5 = RBinIndex(5);
      const auto normal = axis.GetNormalRange(index1, index5);
      EXPECT_EQ(normal.GetBegin(), index1);
      EXPECT_EQ(normal.GetEnd(), index5);
      EXPECT_EQ(std::distance(normal.begin(), normal.end()), 4);
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

TEST(RVariableBinAxis, GetFullRange)
{
   static constexpr std::size_t Bins = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < Bins; i++) {
      bins.push_back(i);
   }
   bins.push_back(Bins);

   {
      const RVariableBinAxis axis(bins);
      const auto full = axis.GetFullRange();
      EXPECT_EQ(full.GetBegin(), RBinIndex::Underflow());
      EXPECT_EQ(full.GetEnd(), RBinIndex());
      EXPECT_EQ(std::distance(full.begin(), full.end()), Bins + 2);
   }

   {
      const RVariableBinAxis axisNoFlowBins(bins, /*enableFlowBins=*/false);
      const auto full = axisNoFlowBins.GetFullRange();
      EXPECT_EQ(full.GetBegin(), RBinIndex(0));
      EXPECT_EQ(full.GetEnd(), RBinIndex(Bins));
      EXPECT_EQ(std::distance(full.begin(), full.end()), Bins);
   }
}
