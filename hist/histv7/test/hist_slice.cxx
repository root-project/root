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
