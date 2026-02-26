#include "hist_test.hxx"

#include <stdexcept>

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
