#include "hist_test.hxx"

#include <array>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

static_assert(std::is_nothrow_move_constructible_v<RHistEngine<int>>);
static_assert(std::is_nothrow_move_assignable_v<RHistEngine<int>>);

TEST(RHistEngine, Constructor)
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

   RHistEngine<int> engine({regularAxis, variableBinAxis, categoricalAxis});
   EXPECT_EQ(engine.GetNDimensions(), 3);
   const auto &axes = engine.GetAxes();
   ASSERT_EQ(axes.size(), 3);
   EXPECT_EQ(axes[0].index(), 0);
   EXPECT_EQ(axes[1].index(), 1);
   EXPECT_EQ(axes[2].index(), 2);
   EXPECT_TRUE(std::get_if<RRegularAxis>(&axes[0]) != nullptr);
   EXPECT_TRUE(std::get_if<RVariableBinAxis>(&axes[1]) != nullptr);
   EXPECT_TRUE(std::get_if<RCategoricalAxis>(&axes[2]) != nullptr);

   // Both axes include underflow and overflow bins.
   EXPECT_EQ(engine.GetTotalNBins(), (BinsX + 2) * (BinsY + 2) * (categories.size() + 1));

   engine = RHistEngine<int>(BinsX, {0, BinsX});
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
   const RRegularAxis axis(Bins, {0, Bins});
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
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<int> engine({axis});

   EXPECT_THROW(engine.GetBinContent(Bins), std::invalid_argument);
}

TEST(RHistEngine, SetBinContent)
{
   using ROOT::Experimental::Internal::SetBinContent;

   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine({axis});

   std::array<RBinIndex, 1> indices = {7};
   SetBinContent(engine, indices, 42);
   EXPECT_EQ(engine.GetBinContent(indices), 42);

   // "bin not found"
   indices = {Bins};
   EXPECT_THROW(SetBinContent(engine, indices, 43), std::invalid_argument);
}

TEST(RHistEngine, Add)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engineA({axis});
   RHistEngine<int> engineB({axis});
   RHistEngine<int> engineC({axis});

   engineA.Fill(-100);
   for (std::size_t i = 0; i < Bins; i++) {
      engineA.Fill(i);
      engineA.Fill(i);
      engineB.Fill(i);
   }
   engineB.Fill(100);

   engineC.Add(engineA);
   engineC.Add(engineB);

   engineA.Add(engineB);

   EXPECT_EQ(engineA.GetBinContent(RBinIndex::Underflow()), 1);
   EXPECT_EQ(engineB.GetBinContent(RBinIndex::Underflow()), 0);
   EXPECT_EQ(engineC.GetBinContent(RBinIndex::Underflow()), 1);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(engineA.GetBinContent(index), 3);
      EXPECT_EQ(engineB.GetBinContent(index), 1);
      EXPECT_EQ(engineC.GetBinContent(index), 3);
   }
   EXPECT_EQ(engineA.GetBinContent(RBinIndex::Overflow()), 1);
   EXPECT_EQ(engineB.GetBinContent(RBinIndex::Overflow()), 1);
   EXPECT_EQ(engineC.GetBinContent(RBinIndex::Overflow()), 1);
}

TEST(RHistEngine, AddDifferent)
{
   // The equality operators of RAxes and the axis objects are already unit-tested separately, so here we only check one
   // case with different the number of bins.
   RHistEngine<int> engineA(10, {0, 1});
   RHistEngine<int> engineB(20, {0, 1});

   EXPECT_THROW(engineA.Add(engineB), std::invalid_argument);
}

TEST(RHistEngine, Clear)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine({axis});

   engine.Fill(-100);
   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(i);
   }
   engine.Fill(100);

   engine.Clear();

   EXPECT_EQ(engine.GetBinContent(RBinIndex::Underflow()), 0);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(engine.GetBinContent(index), 0);
   }
   EXPECT_EQ(engine.GetBinContent(RBinIndex::Overflow()), 0);
}

TEST(RHistEngine, Clone)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engineA({axis});

   engineA.Fill(-100);
   for (std::size_t i = 0; i < Bins; i++) {
      engineA.Fill(i);
   }
   engineA.Fill(100);

   RHistEngine<int> engineB = engineA.Clone();
   ASSERT_EQ(engineB.GetNDimensions(), 1);
   ASSERT_EQ(engineB.GetTotalNBins(), Bins + 2);

   EXPECT_EQ(engineB.GetBinContent(RBinIndex::Underflow()), 1);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(engineB.GetBinContent(index), 1);
   }
   EXPECT_EQ(engineB.GetBinContent(RBinIndex::Overflow()), 1);

   // Check that we can continue filling the clone.
   for (std::size_t i = 0; i < Bins; i++) {
      engineB.Fill(i);
   }

   for (auto index : axis.GetNormalRange()) {
      EXPECT_EQ(engineA.GetBinContent(index), 1);
      EXPECT_EQ(engineB.GetBinContent(index), 2);
   }
}

TEST(RHistEngine, Fill)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
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
   const RRegularAxis axis(Bins, {0, Bins}, /*enableFlowBins=*/false);
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
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine({axis});
   const RRegularAxis axisNoFlowBins(Bins, {0, Bins}, /*enableFlowBins=*/false);
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
   const RRegularAxis axis(Bins, {0, Bins});
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
   const RRegularAxis axis(Bins, {0, Bins});
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

TEST(RHistEngine, FillWeight)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<float> engine({axis});

   engine.Fill(-100, RWeight(0.25));
   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(i, RWeight(0.1 + i * 0.03));
   }
   engine.Fill(100, RWeight(0.75));

   EXPECT_FLOAT_EQ(engine.GetBinContent(RBinIndex::Underflow()), 0.25);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_FLOAT_EQ(engine.GetBinContent(index), 0.1 + index.GetIndex() * 0.03);
   }
   EXPECT_EQ(engine.GetBinContent(RBinIndex::Overflow()), 0.75);
}

TEST(RHistEngine, FillTupleWeight)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<float> engine({axis});

   engine.Fill(std::make_tuple(-100), RWeight(0.25));
   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(std::make_tuple(i), RWeight(0.1 + i * 0.03));
   }
   engine.Fill(std::make_tuple(100), RWeight(0.75));

   std::array<RBinIndex, 1> indices = {RBinIndex::Underflow()};
   EXPECT_FLOAT_EQ(engine.GetBinContent(indices), 0.25);
   for (auto index : axis.GetNormalRange()) {
      indices[0] = index;
      EXPECT_FLOAT_EQ(engine.GetBinContent(indices), 0.1 + index.GetIndex() * 0.03);
   }
   indices[0] = RBinIndex::Overflow();
   EXPECT_EQ(engine.GetBinContent(indices), 0.75);
}

TEST(RHistEngine, FillWeightInvalidNumberOfArguments)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<float> engine1({axis});
   ASSERT_EQ(engine1.GetNDimensions(), 1);
   RHistEngine<float> engine2({axis, axis});
   ASSERT_EQ(engine2.GetNDimensions(), 2);

   EXPECT_NO_THROW(engine1.Fill(1, RWeight(1)));
   EXPECT_THROW(engine1.Fill(1, 2, RWeight(1)), std::invalid_argument);

   EXPECT_THROW(engine2.Fill(1, RWeight(1)), std::invalid_argument);
   EXPECT_NO_THROW(engine2.Fill(1, 2, RWeight(1)));
   EXPECT_THROW(engine2.Fill(1, 2, 3, RWeight(1)), std::invalid_argument);
}

TEST(RHistEngine, FillTupleWeightInvalidNumberOfArguments)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<float> engine1({axis});
   ASSERT_EQ(engine1.GetNDimensions(), 1);
   RHistEngine<float> engine2({axis, axis});
   ASSERT_EQ(engine2.GetNDimensions(), 2);

   EXPECT_NO_THROW(engine1.Fill(std::make_tuple(1), RWeight(1)));
   EXPECT_THROW(engine1.Fill(std::make_tuple(1, 2), RWeight(1)), std::invalid_argument);

   EXPECT_THROW(engine2.Fill(std::make_tuple(1), RWeight(1)), std::invalid_argument);
   EXPECT_NO_THROW(engine2.Fill(std::make_tuple(1, 2), RWeight(1)));
   EXPECT_THROW(engine2.Fill(std::make_tuple(1, 2, 3), RWeight(1)), std::invalid_argument);
}

TEST(RHistEngine, Scale)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<float> engine({axis});

   engine.Fill(-100, RWeight(0.25));
   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(i, RWeight(0.1 + i * 0.03));
   }
   engine.Fill(100, RWeight(0.75));

   static constexpr double Factor = 0.8;
   engine.Scale(Factor);

   EXPECT_FLOAT_EQ(engine.GetBinContent(RBinIndex::Underflow()), Factor * 0.25);
   for (auto index : axis.GetNormalRange()) {
      EXPECT_FLOAT_EQ(engine.GetBinContent(index), Factor * (0.1 + index.GetIndex() * 0.03));
   }
   EXPECT_FLOAT_EQ(engine.GetBinContent(RBinIndex::Overflow()), Factor * 0.75);
}

TEST(RHistEngine_RBinWithError, Add)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<RBinWithError> engineA({axis});
   RHistEngine<RBinWithError> engineB({axis});

   for (std::size_t i = 0; i < Bins; i++) {
      engineA.Fill(i, RWeight(0.2 + i * 0.03));
      engineB.Fill(i, RWeight(0.1 + i * 0.05));
   }

   engineA.Add(engineB);

   for (auto index : axis.GetNormalRange()) {
      auto &bin = engineA.GetBinContent(index);
      double weightA = 0.2 + index.GetIndex() * 0.03;
      double weightB = 0.1 + index.GetIndex() * 0.05;
      EXPECT_FLOAT_EQ(bin.fSum, weightA + weightB);
      EXPECT_FLOAT_EQ(bin.fSum2, weightA * weightA + weightB * weightB);
   }
}

TEST(RHistEngine_RBinWithError, Fill)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<RBinWithError> engine({axis});

   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(i);
   }

   for (auto index : axis.GetNormalRange()) {
      auto &bin = engine.GetBinContent(index);
      EXPECT_EQ(bin.fSum, 1);
      EXPECT_EQ(bin.fSum2, 1);
   }
}

TEST(RHistEngine_RBinWithError, FillWeight)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<RBinWithError> engine({axis});

   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(i, RWeight(0.1 + i * 0.03));
   }

   for (auto index : axis.GetNormalRange()) {
      auto &bin = engine.GetBinContent(index);
      double weight = 0.1 + index.GetIndex() * 0.03;
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
   }
}

TEST(RHistEngine_RBinWithError, Scale)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<RBinWithError> engine({axis});

   for (std::size_t i = 0; i < Bins; i++) {
      engine.Fill(i, RWeight(0.1 + i * 0.03));
   }

   static constexpr double Factor = 0.8;
   engine.Scale(Factor);

   for (auto index : axis.GetNormalRange()) {
      auto &bin = engine.GetBinContent(index);
      double weight = Factor * (0.1 + index.GetIndex() * 0.03);
      EXPECT_FLOAT_EQ(bin.fSum, weight);
      EXPECT_FLOAT_EQ(bin.fSum2, weight * weight);
   }
}
