#include "hist_test.hxx"

#include <iterator>
#include <limits>
#include <stdexcept>

TEST(RRegularAxis, Constructor)
{
   static constexpr std::size_t Bins = 20;
   RRegularAxis axis(Bins, 0, Bins);
   EXPECT_EQ(axis.GetNNormalBins(), Bins);
   EXPECT_EQ(axis.GetTotalNBins(), Bins + 2);
   EXPECT_EQ(axis.GetLow(), 0);
   EXPECT_EQ(axis.GetHigh(), Bins);
   EXPECT_TRUE(axis.HasFlowBins());

   axis = RRegularAxis(Bins, 0, Bins, /*enableFlowBins=*/false);
   EXPECT_EQ(axis.GetNNormalBins(), Bins);
   EXPECT_EQ(axis.GetTotalNBins(), Bins);
   EXPECT_FALSE(axis.HasFlowBins());

   EXPECT_THROW(RRegularAxis(0, 0, Bins), std::invalid_argument);
   EXPECT_THROW(RRegularAxis(Bins, 1, 1), std::invalid_argument);
}

TEST(RRegularAxis, Equality)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axisA(Bins, 0, Bins);
   const RRegularAxis axisANoFlowBins(Bins, 0, Bins, /*enableFlowBins=*/false);
   const RRegularAxis axisA2(Bins, 0, Bins);
   const RRegularAxis axisB(Bins / 2, 0, Bins);
   const RRegularAxis axisC(Bins, 0, Bins / 2);
   const RRegularAxis axisD(Bins, Bins / 2, Bins);

   EXPECT_TRUE(axisA == axisA);
   EXPECT_TRUE(axisA == axisA2);
   EXPECT_TRUE(axisA2 == axisA);

   EXPECT_FALSE(axisA == axisANoFlowBins);

   EXPECT_FALSE(axisA == axisB);
   EXPECT_FALSE(axisA == axisC);
   EXPECT_FALSE(axisA == axisD);

   EXPECT_FALSE(axisB == axisC);
   EXPECT_FALSE(axisB == axisD);

   EXPECT_FALSE(axisC == axisD);
   EXPECT_FALSE(axisD == axisC);
}

TEST(RRegularAxis, ComputeLinearizedIndex)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins);
   const RRegularAxis axisNoFlowBins(Bins, 0, Bins, /*enableFlowBins=*/false);

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

   // Exactly the lower end of the axis interval
   {
      auto linIndex = axis.ComputeLinearizedIndex(0);
      EXPECT_EQ(linIndex.fIndex, 0);
      EXPECT_TRUE(linIndex.fValid);
      linIndex = axisNoFlowBins.ComputeLinearizedIndex(0);
      EXPECT_EQ(linIndex.fIndex, 0);
      EXPECT_TRUE(linIndex.fValid);
   }

   for (std::size_t i = 0; i < Bins; i++) {
      auto linIndex = axis.ComputeLinearizedIndex(i + 0.5);
      EXPECT_EQ(linIndex.fIndex, i);
      EXPECT_TRUE(linIndex.fValid);
      linIndex = axisNoFlowBins.ComputeLinearizedIndex(i + 0.5);
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

TEST(RRegularAxis, GetLinearizedIndex)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins);
   const RRegularAxis axisNoFlowBins(Bins, 0, Bins, /*enableFlowBins=*/false);

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

TEST(RRegularAxis, GetNormalRange)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins);
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

TEST(RRegularAxis, GetFullRange)
{
   static constexpr std::size_t Bins = 20;

   {
      const RRegularAxis axis(Bins, 0, Bins);
      const auto full = axis.GetFullRange();
      EXPECT_EQ(full.GetBegin(), RBinIndex::Underflow());
      EXPECT_EQ(full.GetEnd(), RBinIndex());
      EXPECT_EQ(std::distance(full.begin(), full.end()), Bins + 2);
   }

   {
      const RRegularAxis axisNoFlowBins(Bins, 0, Bins, /*enableFlowBins=*/false);
      const auto full = axisNoFlowBins.GetFullRange();
      EXPECT_EQ(full.GetBegin(), RBinIndex(0));
      EXPECT_EQ(full.GetEnd(), RBinIndex(Bins));
      EXPECT_EQ(std::distance(full.begin(), full.end()), Bins);
   }
}
