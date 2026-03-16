#include "hist_test.hxx"

#include <stdexcept>
#include <vector>

using ROOT::Experimental::Internal::CreateBinIndexRange;

TEST(RSliceSpec, Constructor)
{
   const RSliceSpec full;
   EXPECT_TRUE(full.GetRange().IsInvalid());
   EXPECT_FALSE(full.HasOperation());

   // Slice specification without operation
   const auto normalRange = CreateBinIndexRange(RBinIndex(0), RBinIndex(1), 0);
   const RSliceSpec slice(normalRange);
   EXPECT_EQ(slice.GetRange(), normalRange);
   EXPECT_FALSE(slice.HasOperation());

   // Operations without range
   const RSliceSpec rebin(RSliceSpec::ROperationRebin(2));
   EXPECT_TRUE(rebin.GetRange().IsInvalid());
   EXPECT_TRUE(rebin.HasOperation());
   auto *opRebin = rebin.GetOperationRebin();
   ASSERT_TRUE(opRebin != nullptr);
   EXPECT_EQ(opRebin->GetNGroup(), 2);
   EXPECT_TRUE(rebin.GetOperationSum() == nullptr);

   const RSliceSpec sum(RSliceSpec::ROperationSum{});
   EXPECT_TRUE(sum.GetRange().IsInvalid());
   EXPECT_TRUE(sum.GetOperationRebin() == nullptr);
   EXPECT_TRUE(sum.GetOperationSum() != nullptr);

   // Slice specification with both a range and an operation
   const RSliceSpec sliceRebin(normalRange, RSliceSpec::ROperationRebin(2));
   EXPECT_EQ(sliceRebin.GetRange(), normalRange);
   EXPECT_TRUE(sliceRebin.GetOperationRebin() != nullptr);

   const RSliceSpec sliceSum(normalRange, RSliceSpec::ROperationSum{});
   EXPECT_EQ(sliceSum.GetRange(), normalRange);
   EXPECT_TRUE(sliceSum.GetOperationSum() != nullptr);

   EXPECT_THROW(RSliceSpec::ROperationRebin(0), std::invalid_argument);
}

TEST(RSliceBinIndexMapper, Constructor)
{
   const auto normalRange = CreateBinIndexRange(RBinIndex(0), RBinIndex(1), 0);
   std::vector<RSliceSpec> specs;
   specs.emplace_back();
   specs.emplace_back(normalRange);
   specs.emplace_back(RSliceSpec::ROperationRebin(2));
   specs.emplace_back(RSliceSpec::ROperationSum{});
   specs.emplace_back(normalRange, RSliceSpec::ROperationRebin(2));
   specs.emplace_back(normalRange, RSliceSpec::ROperationSum{});

   const RSliceBinIndexMapper mapper(specs);
   EXPECT_EQ(mapper.GetSliceSpecs().size(), 6);
   // There are two sum operations
   EXPECT_EQ(mapper.GetMappedDimensionality(), 4);

   EXPECT_THROW(RSliceBinIndexMapper({}), std::invalid_argument);
}

TEST(RSliceBinIndexMapper, MapInvalidNumberOfArguments)
{
   const std::vector<RSliceSpec> specs = {RSliceSpec(), RSliceSpec::ROperationSum{}};
   const RSliceBinIndexMapper mapper(specs);

   std::vector<RBinIndex> indices0;
   std::vector<RBinIndex> indices1(1);
   std::vector<RBinIndex> indices2(2);
   std::vector<RBinIndex> indices3(3);

   EXPECT_THROW(mapper.Map(indices1, indices0), std::invalid_argument);
   EXPECT_THROW(mapper.Map(indices1, indices1), std::invalid_argument);
   EXPECT_THROW(mapper.Map(indices1, indices2), std::invalid_argument);
   EXPECT_THROW(mapper.Map(indices2, indices0), std::invalid_argument);
   // The mapper would expect two original indices and one mapped index...
   EXPECT_THROW(mapper.Map(indices2, indices2), std::invalid_argument);
   EXPECT_THROW(mapper.Map(indices3, indices0), std::invalid_argument);
   EXPECT_THROW(mapper.Map(indices3, indices1), std::invalid_argument);
   EXPECT_THROW(mapper.Map(indices3, indices2), std::invalid_argument);
}

TEST(RSliceBinIndexMapper, MapInvalid)
{
   const RSliceBinIndexMapper mapper({RSliceSpec()});
   ASSERT_EQ(mapper.GetMappedDimensionality(), 1);
   std::vector<RBinIndex> original(1);
   ASSERT_TRUE(original[0].IsInvalid());
   std::vector<RBinIndex> mapped(1);

   EXPECT_THROW(mapper.Map(original, mapped), std::invalid_argument);
}

TEST(RSliceBinIndexMapper, MapFull)
{
   const RSliceBinIndexMapper mapper({RSliceSpec()});
   ASSERT_EQ(mapper.GetMappedDimensionality(), 1);
   std::vector<RBinIndex> original(1);
   std::vector<RBinIndex> mapped(1);

   // Each index should be mapped to itself...
   for (auto index : {RBinIndex::Underflow(), RBinIndex(0), RBinIndex::Overflow()}) {
      original[0] = index;
      bool success = mapper.Map(original, mapped);
      EXPECT_TRUE(success);
      EXPECT_EQ(mapped[0], index);
   }
}

TEST(RSliceBinIndexMapper, MapSliceNormal)
{
   const auto range = CreateBinIndexRange(RBinIndex(1), RBinIndex(2), 0);
   const RSliceBinIndexMapper mapper({range});
   ASSERT_EQ(mapper.GetMappedDimensionality(), 1);
   std::vector<RBinIndex> original(1);
   std::vector<RBinIndex> mapped(1);

   // Underflow and overflow indices should be mapped to themselves...
   for (auto index : {RBinIndex::Underflow(), RBinIndex::Overflow()}) {
      original[0] = index;
      bool success = mapper.Map(original, mapped);
      EXPECT_TRUE(success);
      EXPECT_EQ(mapped[0], index);
   }

   {
      // This should be mapped to the underflow bin...
      original[0] = RBinIndex(0);
      bool success = mapper.Map(original, mapped);
      EXPECT_TRUE(success);
      EXPECT_TRUE(mapped[0].IsUnderflow());
   }

   {
      // Contained normal bins are shifted...
      original[0] = RBinIndex(1);
      bool success = mapper.Map(original, mapped);
      EXPECT_TRUE(success);
      EXPECT_EQ(mapped[0], RBinIndex(0));
   }

   {
      // This should be mapped to the overflow bin...
      original[0] = RBinIndex(2);
      bool success = mapper.Map(original, mapped);
      EXPECT_TRUE(success);
      EXPECT_TRUE(mapped[0].IsOverflow());
   }
}

TEST(RSliceBinIndexMapper, MapSliceFull)
{
   const auto range = CreateBinIndexRange(RBinIndex::Underflow(), RBinIndex(), 1);
   const RSliceBinIndexMapper mapper({range});
   ASSERT_EQ(mapper.GetMappedDimensionality(), 1);
   std::vector<RBinIndex> original(1);
   std::vector<RBinIndex> mapped(1);

   // Each index should be mapped to itself...
   for (auto index : {RBinIndex::Underflow(), RBinIndex(0), RBinIndex::Overflow()}) {
      original[0] = index;
      bool success = mapper.Map(original, mapped);
      EXPECT_TRUE(success);
      EXPECT_EQ(mapped[0], index);
   }
}

TEST(RSliceBinIndexMapper, MapRebin)
{
   const RSliceBinIndexMapper mapper({RSliceSpec::ROperationRebin(/*nGroup=*/2)});
   ASSERT_EQ(mapper.GetMappedDimensionality(), 1);
   std::vector<RBinIndex> original(1);
   std::vector<RBinIndex> mapped(1);

   // Underflow and overflow indices should be mapped to themselves...
   for (auto index : {RBinIndex::Underflow(), RBinIndex::Overflow()}) {
      original[0] = index;
      bool success = mapper.Map(original, mapped);
      EXPECT_TRUE(success);
      EXPECT_EQ(mapped[0], index);
   }

   // Normal bins are merged according to nGroup...
   for (std::uint64_t i = 0; i < 4; i++) {
      original[0] = RBinIndex(i);
      bool success = mapper.Map(original, mapped);
      EXPECT_TRUE(success);
      ASSERT_TRUE(mapped[0].IsNormal());
      EXPECT_EQ(mapped[0].GetIndex(), i / 2);
   }
}

TEST(RSliceBinIndexMapper, MapSliceRebin)
{
   const auto range = CreateBinIndexRange(RBinIndex(1), RBinIndex(4), 0);
   const RSliceBinIndexMapper mapper({RSliceSpec(range, RSliceSpec::ROperationRebin(/*nGroup=*/2))});
   ASSERT_EQ(mapper.GetMappedDimensionality(), 1);
   std::vector<RBinIndex> original(1);
   std::vector<RBinIndex> mapped(1);

   // Contained normal bins are first shifted and then merged according to nGroup...
   for (std::uint64_t i = 1; i < 4; i++) {
      original[0] = RBinIndex(i);
      bool success = mapper.Map(original, mapped);
      EXPECT_TRUE(success);
      ASSERT_TRUE(mapped[0].IsNormal());
      EXPECT_EQ(mapped[0].GetIndex(), (i - 1) / 2);
   }
}

TEST(RSliceBinIndexMapper, MapSum)
{
   const RSliceBinIndexMapper mapper({RSliceSpec::ROperationSum{}});
   ASSERT_EQ(mapper.GetMappedDimensionality(), 0);
   std::vector<RBinIndex> original(1);
   std::vector<RBinIndex> mapped(0);

   // All indices should be summed...
   for (auto index : {RBinIndex::Underflow(), RBinIndex(0), RBinIndex::Overflow()}) {
      original[0] = index;
      bool success = mapper.Map(original, mapped);
      EXPECT_TRUE(success);
   }
}

TEST(RSliceBinIndexMapper, MapProjection)
{
   const RSliceBinIndexMapper mapper({RSliceSpec{}, RSliceSpec::ROperationSum{}});
   ASSERT_EQ(mapper.GetMappedDimensionality(), 1);
   std::vector<RBinIndex> original(2);
   std::vector<RBinIndex> mapped(1);

   for (auto index0 : {RBinIndex::Underflow(), RBinIndex(0), RBinIndex::Overflow()}) {
      original[0] = index0;
      // The second dimension should be projected...
      for (auto index1 : {RBinIndex::Underflow(), RBinIndex(0), RBinIndex::Overflow()}) {
         original[1] = index1;
         // Reset mapped index, to be sure it's correctly set...
         mapped[0] = RBinIndex();
         bool success = mapper.Map(original, mapped);
         EXPECT_TRUE(success);
         EXPECT_EQ(mapped[0], index0);
      }
   }
}

TEST(RSliceBinIndexMapper, MapSliceSum)
{
   const auto range = CreateBinIndexRange(RBinIndex(1), RBinIndex(2), 0);
   const RSliceBinIndexMapper mapper({RSliceSpec(range, RSliceSpec::ROperationSum{})});
   ASSERT_EQ(mapper.GetMappedDimensionality(), 0);
   std::vector<RBinIndex> original(1);
   std::vector<RBinIndex> mapped(0);

   // Cut indices should be discarded...
   for (auto index : {RBinIndex::Underflow(), RBinIndex(0), RBinIndex(2), RBinIndex::Overflow()}) {
      original[0] = index;
      bool success = mapper.Map(original, mapped);
      EXPECT_FALSE(success);
   }

   {
      // Contained normal bins are summed...
      original[0] = RBinIndex(1);
      bool success = mapper.Map(original, mapped);
      EXPECT_TRUE(success);
   }
}

TEST(RSliceBinIndexMapper, MapSliceFullSum)
{
   const auto range = CreateBinIndexRange(RBinIndex::Underflow(), RBinIndex(), 1);
   const RSliceBinIndexMapper mapper({RSliceSpec(range, RSliceSpec::ROperationSum{})});
   ASSERT_EQ(mapper.GetMappedDimensionality(), 0);
   std::vector<RBinIndex> original(1);
   std::vector<RBinIndex> mapped(0);

   // All indices should be summed...
   for (auto index : {RBinIndex::Underflow(), RBinIndex(0), RBinIndex::Overflow()}) {
      original[0] = index;
      bool success = mapper.Map(original, mapped);
      EXPECT_TRUE(success);
   }
}

TEST(RHistEngine, SliceInvalidNumberOfArguments)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<int> engine1(axis);
   ASSERT_EQ(engine1.GetNDimensions(), 1);
   const RHistEngine<int> engine2(axis, axis);
   ASSERT_EQ(engine2.GetNDimensions(), 2);

   EXPECT_NO_THROW(engine1.Slice(RSliceSpec{}));
   EXPECT_THROW(engine1.Slice(RSliceSpec{}, RSliceSpec{}), std::invalid_argument);

   EXPECT_THROW(engine2.Slice(RSliceSpec{}), std::invalid_argument);
   EXPECT_NO_THROW(engine2.Slice(RSliceSpec{}, RSliceSpec{}));
   EXPECT_THROW(engine2.Slice(RSliceSpec{}, RSliceSpec{}, RSliceSpec{}), std::invalid_argument);
}

TEST(RHistEngine, SliceSumAll)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   const RHistEngine<int> engine1(axis);
   ASSERT_EQ(engine1.GetNDimensions(), 1);
   const RHistEngine<int> engine2(axis, axis);
   ASSERT_EQ(engine2.GetNDimensions(), 2);

   const RSliceSpec sum(RSliceSpec::ROperationSum{});
   EXPECT_THROW(engine1.Slice(sum), std::invalid_argument);
   EXPECT_THROW(engine2.Slice(sum, sum), std::invalid_argument);
}

TEST(RHistEngine, SliceFull)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine(axis);

   engine.SetBinContent(RBinIndex::Underflow(), 100);
   for (std::size_t i = 0; i < Bins; i++) {
      engine.SetBinContent(i, i + 1);
   }
   engine.SetBinContent(RBinIndex::Overflow(), 200);

   // Three different ways of "slicing" which will keep the entire axis.
   for (auto sliceSpec : {RSliceSpec{}, RSliceSpec(axis.GetFullRange()), RSliceSpec(axis.GetNormalRange())}) {
      const auto sliced = engine.Slice(sliceSpec);
      ASSERT_EQ(sliced.GetNDimensions(), 1);
      EXPECT_TRUE(sliced.GetAxes()[0].GetRegularAxis() != nullptr);
      EXPECT_EQ(sliced.GetAxes()[0].GetNNormalBins(), Bins);
      EXPECT_EQ(sliced.GetTotalNBins(), Bins + 2);

      EXPECT_EQ(sliced.GetBinContent(RBinIndex::Underflow()), 100);
      for (std::size_t i = 0; i < Bins; i++) {
         EXPECT_EQ(sliced.GetBinContent(i), i + 1);
      }
      EXPECT_EQ(sliced.GetBinContent(RBinIndex::Overflow()), 200);
   }
}

TEST(RHistEngine, SliceNormal)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine(axis);

   engine.SetBinContent(RBinIndex::Underflow(), 100);
   for (std::size_t i = 0; i < Bins; i++) {
      engine.SetBinContent(i, i + 1);
   }
   engine.SetBinContent(RBinIndex::Overflow(), 200);

   const auto sliced = engine.Slice(axis.GetNormalRange(1, 5));
   ASSERT_EQ(sliced.GetNDimensions(), 1);
   const auto *regular = sliced.GetAxes()[0].GetRegularAxis();
   ASSERT_TRUE(regular != nullptr);
   EXPECT_EQ(regular->GetNNormalBins(), 4);
   EXPECT_EQ(regular->GetLow(), 1);
   EXPECT_EQ(regular->GetHigh(), 5);
   EXPECT_EQ(sliced.GetTotalNBins(), 6);

   EXPECT_EQ(sliced.GetBinContent(RBinIndex::Underflow()), 101);
   for (std::size_t i = 0; i < 4; i++) {
      EXPECT_EQ(sliced.GetBinContent(i), i + 2);
   }
   EXPECT_EQ(sliced.GetBinContent(RBinIndex::Overflow()), 395);
}

TEST(RHistEngine, SliceRebin)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine(axis);

   engine.SetBinContent(RBinIndex::Underflow(), 100);
   for (std::size_t i = 0; i < Bins; i++) {
      engine.SetBinContent(i, i + 1);
   }
   engine.SetBinContent(RBinIndex::Overflow(), 200);

   const auto sliced = engine.Slice(RSliceSpec::ROperationRebin(2));
   ASSERT_EQ(sliced.GetNDimensions(), 1);
   const auto *regular = sliced.GetAxes()[0].GetRegularAxis();
   ASSERT_TRUE(regular != nullptr);
   EXPECT_EQ(regular->GetNNormalBins(), Bins / 2);
   EXPECT_EQ(regular->GetLow(), 0);
   EXPECT_EQ(regular->GetHigh(), Bins);
   EXPECT_EQ(sliced.GetTotalNBins(), Bins / 2 + 2);

   EXPECT_EQ(sliced.GetBinContent(RBinIndex::Underflow()), 100);
   for (std::size_t i = 0; i < Bins / 2; i++) {
      EXPECT_EQ(sliced.GetBinContent(i), 4 * i + 3);
   }
   EXPECT_EQ(sliced.GetBinContent(RBinIndex::Overflow()), 200);
}

TEST(RHistEngine, SliceRangeRebin)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine(axis);

   engine.SetBinContent(RBinIndex::Underflow(), 100);
   for (std::size_t i = 0; i < Bins; i++) {
      engine.SetBinContent(i, i + 1);
   }
   engine.SetBinContent(RBinIndex::Overflow(), 200);

   const RSliceSpec spec(axis.GetNormalRange(1, 5), RSliceSpec::ROperationRebin(2));
   const auto sliced = engine.Slice(spec);
   ASSERT_EQ(sliced.GetNDimensions(), 1);
   const auto *regular = sliced.GetAxes()[0].GetRegularAxis();
   ASSERT_TRUE(regular != nullptr);
   EXPECT_EQ(regular->GetNNormalBins(), 2);
   EXPECT_EQ(regular->GetLow(), 1);
   EXPECT_EQ(regular->GetHigh(), 5);
   EXPECT_EQ(sliced.GetTotalNBins(), 4);

   EXPECT_EQ(sliced.GetBinContent(RBinIndex::Underflow()), 101);
   for (std::size_t i = 0; i < 2; i++) {
      EXPECT_EQ(sliced.GetBinContent(i), 4 * i + 5);
   }
   EXPECT_EQ(sliced.GetBinContent(RBinIndex::Overflow()), 395);
}

TEST(RHistEngine, SliceSum)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine(axis, axis);

   engine.SetBinContent(RBinIndex::Underflow(), 0, 1000);
   engine.SetBinContent(RBinIndex::Underflow(), 2, 2000);
   for (std::size_t i = 0; i < Bins; i++) {
      for (std::size_t j = 0; j < Bins; j++) {
         engine.SetBinContent(i, RBinIndex::Underflow(), 100 * i);
         engine.SetBinContent(i, RBinIndex(j), i * Bins + j);
         engine.SetBinContent(i, RBinIndex::Overflow(), 200 * i);
      }
   }
   engine.SetBinContent(RBinIndex::Overflow(), 3, 3000);
   engine.SetBinContent(RBinIndex::Overflow(), 6, 4000);

   const auto sliced = engine.Slice(RSliceSpec{}, RSliceSpec::ROperationSum{});
   ASSERT_EQ(sliced.GetNDimensions(), 1);
   EXPECT_TRUE(sliced.GetAxes()[0].GetRegularAxis() != nullptr);
   EXPECT_EQ(sliced.GetAxes()[0].GetNNormalBins(), Bins);
   EXPECT_EQ(sliced.GetTotalNBins(), Bins + 2);

   EXPECT_EQ(sliced.GetBinContent(RBinIndex::Underflow()), 3000);
   for (std::size_t i = 0; i < Bins; i++) {
      EXPECT_EQ(sliced.GetBinContent(i), i * (100 + Bins * Bins + 200) + Bins * (Bins - 1) / 2);
   }
   EXPECT_EQ(sliced.GetBinContent(RBinIndex::Overflow()), 7000);
}

TEST(RHistEngine, SliceRangeSum)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<int> engine(axis, axis);

   engine.SetBinContent(RBinIndex::Underflow(), 0, 1000);
   engine.SetBinContent(RBinIndex::Underflow(), 2, 2000);
   for (std::size_t i = 0; i < Bins; i++) {
      for (std::size_t j = 0; j < Bins; j++) {
         engine.SetBinContent(i, RBinIndex::Underflow(), 100 * i);
         engine.SetBinContent(i, RBinIndex(j), i * Bins + j);
         engine.SetBinContent(i, RBinIndex::Overflow(), 200 * i);
      }
   }
   engine.SetBinContent(RBinIndex::Overflow(), 3, 3000);
   engine.SetBinContent(RBinIndex::Overflow(), 6, 4000);

   const RSliceSpec rangeSum(axis.GetNormalRange(1, 5), RSliceSpec::ROperationSum{});
   const auto sliced = engine.Slice(RSliceSpec{}, rangeSum);
   ASSERT_EQ(sliced.GetNDimensions(), 1);
   EXPECT_TRUE(sliced.GetAxes()[0].GetRegularAxis() != nullptr);
   EXPECT_EQ(sliced.GetAxes()[0].GetNNormalBins(), Bins);
   EXPECT_EQ(sliced.GetTotalNBins(), Bins + 2);

   EXPECT_EQ(sliced.GetBinContent(RBinIndex::Underflow()), 2000);
   for (std::size_t i = 0; i < Bins; i++) {
      EXPECT_EQ(sliced.GetBinContent(i), 4 * i * Bins + 10);
   }
   EXPECT_EQ(sliced.GetBinContent(RBinIndex::Overflow()), 3000);
}
