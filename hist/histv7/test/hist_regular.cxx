#include "hist_test.hxx"

#include <limits>

TEST(RRegularAxis, Constructor)
{
   static constexpr std::size_t Bins = 20;
   RRegularAxis axis(Bins, 0, Bins);
   EXPECT_EQ(axis.GetNumNormalBins(), Bins);
   EXPECT_EQ(axis.GetTotalNumBins(), Bins + 2);
   EXPECT_EQ(axis.GetLow(), 0);
   EXPECT_EQ(axis.GetHigh(), Bins);
   EXPECT_TRUE(axis.HasFlowBins());

   axis = RRegularAxis(Bins, 0, Bins, /*enableFlowBins=*/false);
   EXPECT_EQ(axis.GetNumNormalBins(), Bins);
   EXPECT_EQ(axis.GetTotalNumBins(), Bins);
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
