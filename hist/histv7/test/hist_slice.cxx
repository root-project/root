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
