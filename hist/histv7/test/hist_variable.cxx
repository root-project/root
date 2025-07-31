#include "hist_test.hxx"

#include <limits>
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
   EXPECT_EQ(axis.GetNumNormalBins(), Bins);
   EXPECT_EQ(axis.GetTotalNumBins(), Bins + 2);
   EXPECT_TRUE(axis.HasFlowBins());

   axis = RVariableBinAxis(bins, /*enableFlowBins=*/false);
   EXPECT_EQ(axis.GetNumNormalBins(), Bins);
   EXPECT_EQ(axis.GetTotalNumBins(), Bins);
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
