#include "hist_test.hxx"

#include <stdexcept>
#include <tuple>
#include <variant>
#include <vector>

TEST(RAxes, Constructor)
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

   RAxes axes({regularAxis, variableBinAxis});
   EXPECT_EQ(axes.GetNumDimensions(), 2);
   const auto &v = axes.Get();
   ASSERT_EQ(v.size(), 2);
   EXPECT_EQ(v[0].index(), 0);
   EXPECT_EQ(v[1].index(), 1);
   EXPECT_TRUE(std::get_if<RRegularAxis>(&v[0]) != nullptr);
   EXPECT_TRUE(std::get_if<RVariableBinAxis>(&v[1]) != nullptr);

   std::vector<RAxes::AxisVariant> newAxes{variableBinAxis, regularAxis};
   axes = RAxes(newAxes);
   EXPECT_EQ(axes.GetNumDimensions(), 2);

   EXPECT_THROW(RAxes({}), std::invalid_argument);
}

TEST(RAxes, Equality)
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

   const RAxes axesA({regularAxis, variableBinAxis});
   const RAxes axesA2({regularAxis, variableBinAxis});
   const RAxes axesB({variableBinAxis, regularAxis});
   const RAxes axesC({regularAxis});
   const RAxes axesD({variableBinAxis});

   EXPECT_TRUE(axesA == axesA);
   EXPECT_TRUE(axesA == axesA2);
   EXPECT_TRUE(axesA2 == axesA);

   EXPECT_FALSE(axesA == axesB);
   EXPECT_FALSE(axesA == axesC);
   EXPECT_FALSE(axesA == axesD);

   EXPECT_FALSE(axesB == axesC);
   EXPECT_FALSE(axesB == axesD);

   EXPECT_FALSE(axesC == axesD);
}

TEST(RAxes, ComputeTotalNumBins)
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
   const RAxes axes({regularAxis, variableBinAxis});

   // Both axes include underflow and overflow bins.
   EXPECT_EQ(axes.ComputeTotalNumBins(), (BinsX + 2) * (BinsY + 2));
}

TEST(RAxes, ComputeGlobalIndex)
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
   const RAxes axes({regularAxis, variableBinAxis});

   {
      const auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(1.5, 2.5));
      EXPECT_EQ(globalIndex.fIndex, 1 * (BinsY + 2) + 2);
      EXPECT_TRUE(globalIndex.fValid);
   }

   {
      // Underflow bin of the first axis.
      const auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(-1, 2.5));
      EXPECT_EQ(globalIndex.fIndex, BinsX * (BinsY + 2) + 2);
      EXPECT_TRUE(globalIndex.fValid);
   }

   {
      // Overflow bin of the second axis.
      const auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(1.5, 42));
      EXPECT_EQ(globalIndex.fIndex, 1 * (BinsY + 2) + BinsY + 1);
      EXPECT_TRUE(globalIndex.fValid);
   }
}

TEST(RAxes, ComputeGlobalIndexNoFlowBins)
{
   static constexpr std::size_t BinsX = 20;
   const RRegularAxis regularAxis(BinsX, 0, BinsX, /*enableFlowBins=*/false);
   static constexpr std::size_t BinsY = 30;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins, /*enableFlowBins=*/false);
   const RAxes axes({regularAxis, variableBinAxis});
   ASSERT_EQ(axes.ComputeTotalNumBins(), BinsX * BinsY);

   {
      const auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(1.5, 2.5));
      EXPECT_EQ(globalIndex.fIndex, 1 * BinsY + 2);
      EXPECT_TRUE(globalIndex.fValid);
   }

   {
      // Underflow bin of the first axis.
      const auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(-1, 2.5));
      EXPECT_EQ(globalIndex.fIndex, 0);
      EXPECT_FALSE(globalIndex.fValid);
   }

   {
      // Overflow bin of the second axis.
      const auto globalIndex = axes.ComputeGlobalIndex(std::make_tuple(1.5, 42));
      EXPECT_EQ(globalIndex.fIndex, 0);
      EXPECT_FALSE(globalIndex.fValid);
   }
}

TEST(RAxes, ComputeGlobalIndexInvalidNumberOfArguments)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins);
   const RAxes axes1({axis});
   ASSERT_EQ(axes1.GetNumDimensions(), 1);
   const RAxes axes2({axis, axis});
   ASSERT_EQ(axes2.GetNumDimensions(), 2);

   EXPECT_NO_THROW(axes1.ComputeGlobalIndex(std::make_tuple(1)));
   EXPECT_THROW(axes1.ComputeGlobalIndex(std::make_tuple(1, 2)), std::invalid_argument);

   EXPECT_THROW(axes2.ComputeGlobalIndex(std::make_tuple(1)), std::invalid_argument);
   EXPECT_NO_THROW(axes2.ComputeGlobalIndex(std::make_tuple(1, 2)));
   EXPECT_THROW(axes2.ComputeGlobalIndex(std::make_tuple(1, 2, 3)), std::invalid_argument);
}
