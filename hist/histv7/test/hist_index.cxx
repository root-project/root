#include "hist_test.hxx"

#include <cstdint>
#include <iterator>
#include <vector>

#ifndef TYPED_TEST_SUITE
#define TYPED_TEST_SUITE TYPED_TEST_CASE
#endif

TEST(RBinIndex, Constructor)
{
   const RBinIndex invalid;
   EXPECT_TRUE(invalid.IsInvalid());

   const RBinIndex index(0);
   EXPECT_TRUE(index.IsNormal());
   EXPECT_EQ(index.GetIndex(), 0);

   const auto underflow = RBinIndex::Underflow();
   EXPECT_FALSE(underflow.IsNormal());
   EXPECT_TRUE(underflow.IsUnderflow());

   const auto overflow = RBinIndex::Overflow();
   EXPECT_FALSE(overflow.IsNormal());
   EXPECT_TRUE(overflow.IsOverflow());
}

TEST(RBinIndex, Plus)
{
   const RBinIndex index1(1);
   ASSERT_EQ(index1.GetIndex(), 1);

   {
      auto index2 = index1;
      EXPECT_EQ((++index2).GetIndex(), 2);
      EXPECT_EQ(index2.GetIndex(), 2);
   }

   {
      auto index2 = index1;
      EXPECT_EQ((index2++).GetIndex(), 1);
      EXPECT_EQ(index2.GetIndex(), 2);
   }

   {
      auto index3 = index1;
      EXPECT_EQ((index3 += 2).GetIndex(), 3);
      EXPECT_EQ(index3.GetIndex(), 3);
   }

   {
      const auto index3 = index1 + 2;
      EXPECT_EQ(index3.GetIndex(), 3);
   }

   // Arithmetic operations on special values go to InvalidIndex.
   for (auto index : {RBinIndex::Underflow(), RBinIndex::Overflow(), RBinIndex()}) {
      index++;
      EXPECT_TRUE(index.IsInvalid());
   }

   // Matches RBinIndex::UnderflowIndex
   static constexpr auto UnderflowIndex = static_cast<std::uint64_t>(-3);
   EXPECT_TRUE((RBinIndex(0) + UnderflowIndex).IsInvalid());
   EXPECT_TRUE((RBinIndex(3) + UnderflowIndex).IsInvalid());
}

TEST(RBinIndex, Minus)
{
   const RBinIndex index3(3);
   ASSERT_EQ(index3.GetIndex(), 3);

   {
      auto index2 = index3;
      EXPECT_EQ((--index2).GetIndex(), 2);
      EXPECT_EQ(index2.GetIndex(), 2);
   }

   {
      auto index2 = index3;
      EXPECT_EQ((index2--).GetIndex(), 3);
      EXPECT_EQ(index2.GetIndex(), 2);
   }

   {
      auto index1 = index3;
      EXPECT_EQ((index1 -= 2).GetIndex(), 1);
      EXPECT_EQ(index1.GetIndex(), 1);
   }

   {
      const auto index1 = index3 - 2;
      EXPECT_EQ(index1.GetIndex(), 1);
   }

   // Arithmetic operations on special values go to InvalidIndex.
   for (auto index : {RBinIndex::Underflow(), RBinIndex::Overflow(), RBinIndex()}) {
      index--;
      EXPECT_TRUE(index.IsInvalid());
   }

   EXPECT_TRUE((RBinIndex(0) - 1).IsInvalid());
   EXPECT_TRUE((RBinIndex(0) - 4).IsInvalid());
}

TEST(RBinIndex, Equality)
{
   RBinIndex index(1);
   EXPECT_EQ(index, RBinIndex(1));
   index++;
   EXPECT_EQ(index, RBinIndex(2));
   EXPECT_NE(index, RBinIndex(3));

   const auto underflow = RBinIndex::Underflow();
   EXPECT_EQ(underflow, RBinIndex::Underflow());
   EXPECT_NE(index, underflow);

   const auto overflow = RBinIndex::Overflow();
   EXPECT_EQ(overflow, RBinIndex::Overflow());
   EXPECT_NE(index, overflow);
   EXPECT_NE(underflow, overflow);
}

TEST(RBinIndex, Relation)
{
   const RBinIndex index1(1);
   EXPECT_LE(index1, index1);
   EXPECT_GE(index1, index1);

   const RBinIndex index2(2);
   EXPECT_LT(index1, index2);
   EXPECT_LE(index1, index2);
   EXPECT_GT(index2, index1);
   EXPECT_GE(index2, index1);

   const auto underflow = RBinIndex::Underflow();
   EXPECT_LE(underflow, RBinIndex::Underflow());
   EXPECT_GE(underflow, RBinIndex::Underflow());
   EXPECT_FALSE(index1 < underflow);
   EXPECT_FALSE(index1 <= underflow);
   EXPECT_FALSE(index1 > underflow);
   EXPECT_FALSE(index1 >= underflow);

   const auto overflow = RBinIndex::Overflow();
   EXPECT_LE(overflow, RBinIndex::Overflow());
   EXPECT_GE(overflow, RBinIndex::Overflow());
   EXPECT_FALSE(index1 < overflow);
   EXPECT_FALSE(index1 <= overflow);
   EXPECT_FALSE(index1 > overflow);
   EXPECT_FALSE(index1 >= overflow);

   EXPECT_FALSE(underflow < overflow);
   EXPECT_FALSE(underflow <= overflow);
   EXPECT_FALSE(underflow > overflow);
   EXPECT_FALSE(underflow >= overflow);
}

template <typename T>
class RBinIndexConversion : public testing::Test {};

using IntegerTypes = testing::Types<signed char, unsigned char, short, unsigned short, int, unsigned int, long,
                                    unsigned long, long long, unsigned long long>;
TYPED_TEST_SUITE(RBinIndexConversion, IntegerTypes);

TYPED_TEST(RBinIndexConversion, Constructor)
{
   const TypeParam input = 1;
   const RBinIndex index(input);
   EXPECT_EQ(index.GetIndex(), 1);
}

using ROOT::Experimental::Internal::CreateBinIndexRange;

TEST(RBinIndexRange, ConstructorCreate)
{
   const RBinIndexRange invalid;
   EXPECT_TRUE(invalid.GetBegin().IsInvalid());
   EXPECT_TRUE(invalid.GetEnd().IsInvalid());

   const auto index0 = RBinIndex(0);
   const auto range0 = CreateBinIndexRange(index0, index0, 0);
   EXPECT_EQ(range0.GetBegin(), index0);
   EXPECT_EQ(range0.GetEnd(), index0);

   const auto range01 = CreateBinIndexRange(index0, RBinIndex(1), 1);
   EXPECT_EQ(range01.GetBegin(), index0);
   EXPECT_EQ(range01.GetEnd(), RBinIndex(1));
}

TEST(RBinIndexRange, Equality)
{
   const auto index0 = RBinIndex(0);
   const auto index1 = RBinIndex(1);
   const auto empty = CreateBinIndexRange(index0, index0, 0);
   const auto empty0 = CreateBinIndexRange(index0, index0, 0);
   const auto empty1 = CreateBinIndexRange(index1, index1, 0);
   EXPECT_EQ(empty, empty0);
   EXPECT_NE(empty0, empty1);

   const auto range01 = CreateBinIndexRange(index0, index1, 0);
   EXPECT_NE(empty, range01);

   const auto underflow = RBinIndex::Underflow();
   const RBinIndex invalid;
   const auto full1 = CreateBinIndexRange(underflow, invalid, /*nNormalBins=*/1);
   const auto full2 = CreateBinIndexRange(underflow, invalid, /*nNormalBins=*/2);
   EXPECT_NE(range01, full1);
   EXPECT_NE(full1, full2);
}

TEST(RBinIndexRange, Empty)
{
   const auto index0 = RBinIndex(0);
   const auto empty = CreateBinIndexRange(index0, index0, 0);
   EXPECT_EQ(empty.begin(), empty.end());
   EXPECT_EQ(std::distance(empty.begin(), empty.end()), 0);
}

TEST(RBinIndexRange, Normal)
{
   const auto index0 = RBinIndex(0);
   const auto range01 = CreateBinIndexRange(index0, RBinIndex(1), 0);
   EXPECT_EQ(std::distance(range01.begin(), range01.end()), 1);
   auto range01It = range01.begin();
   EXPECT_TRUE(range01It->IsNormal());
   EXPECT_EQ(*range01It, index0);
   range01It++;
   EXPECT_EQ(range01It, range01.end());
}

TEST(RBinIndexRange, Full)
{
   const auto underflow = RBinIndex::Underflow();
   const RBinIndex invalid;
   const auto full = CreateBinIndexRange(underflow, invalid, /*nNormalBins=*/10);
   EXPECT_EQ(full.GetBegin(), underflow);
   EXPECT_EQ(full.GetEnd(), invalid);
   EXPECT_EQ(std::distance(full.begin(), full.end()), 12);

   const std::vector binIndices(full.begin(), full.end());
   ASSERT_EQ(binIndices.size(), 12);
   EXPECT_TRUE(binIndices.front().IsUnderflow());
   EXPECT_TRUE(binIndices.back().IsOverflow());
}

TEST(RBinIndexMultiRange, Constructor)
{
   const RBinIndexMultiRange invalid;
   EXPECT_TRUE(invalid.GetRanges().empty());

   const auto index0 = RBinIndex(0);
   const auto range0 = CreateBinIndexRange(index0, index0, 0);
   const RBinIndexMultiRange multiRange0({range0});
   ASSERT_EQ(multiRange0.GetRanges().size(), 1);
   EXPECT_EQ(multiRange0.GetRanges()[0], range0);

   const auto range01 = CreateBinIndexRange(index0, RBinIndex(1), 1);
   const std::vector<RBinIndexRange> ranges = {range0, range01};
   const RBinIndexMultiRange multiRange001(ranges);
   ASSERT_EQ(multiRange001.GetRanges().size(), 2);
   EXPECT_EQ(multiRange001.GetRanges()[0], range0);
   EXPECT_EQ(multiRange001.GetRanges()[1], range01);
}

TEST(RBinIndexMultiRange, Equality)
{
   const auto index0 = RBinIndex(0);
   const auto range0 = CreateBinIndexRange(index0, index0, 0);
   const auto range01 = CreateBinIndexRange(index0, RBinIndex(1), 1);

   const RBinIndexMultiRange invalid;
   const RBinIndexMultiRange multiRange({range0});
   const RBinIndexMultiRange multiRange0({range0});
   const RBinIndexMultiRange multiRange001({range0, range01});

   EXPECT_NE(invalid, multiRange);
   EXPECT_EQ(multiRange, multiRange0);
   EXPECT_NE(multiRange, multiRange001);
}

TEST(RBinIndexMultiRange, Invalid)
{
   const RBinIndexMultiRange invalid;
   EXPECT_EQ(invalid.begin(), invalid.end());
   EXPECT_EQ(std::distance(invalid.begin(), invalid.end()), 0);
}

// For one-dimensional iteration, the behavior should be identical to RBinIndexRange.
TEST(RBinIndexMultiRange, Empty)
{
   const auto index0 = RBinIndex(0);
   const auto empty = CreateBinIndexRange(index0, index0, 0);
   ASSERT_EQ(empty.begin(), empty.end());
   const RBinIndexMultiRange emptyMulti({empty});
   EXPECT_EQ(emptyMulti.begin(), emptyMulti.end());
   EXPECT_EQ(std::distance(emptyMulti.begin(), emptyMulti.end()), 0);
}

TEST(RBinIndexMultiRange, Normal)
{
   const auto index0 = RBinIndex(0);
   const auto range01 = CreateBinIndexRange(index0, RBinIndex(1), 0);
   const RBinIndexMultiRange normal({range01});
   EXPECT_EQ(std::distance(normal.begin(), normal.end()), 1);

   auto normalIt = normal.begin();
   auto &indices = *normalIt;
   ASSERT_EQ(indices.size(), 1);
   EXPECT_TRUE(indices[0].IsNormal());
   EXPECT_EQ(indices[0], index0);
   normalIt++;
   EXPECT_EQ(normalIt, normal.end());
}

TEST(RBinIndexMultiRange, Full)
{
   const auto underflow = RBinIndex::Underflow();
   const RBinIndex invalid;
   const auto full = CreateBinIndexRange(underflow, invalid, /*nNormalBins=*/10);
   const RBinIndexMultiRange fullMulti({full});
   EXPECT_EQ(std::distance(fullMulti.begin(), fullMulti.end()), 12);

   const std::vector values(fullMulti.begin(), fullMulti.end());
   ASSERT_EQ(values.size(), 12);
   ASSERT_EQ(values.front().size(), 1);
   EXPECT_TRUE(values.front()[0].IsUnderflow());
   ASSERT_EQ(values.back().size(), 1);
   EXPECT_TRUE(values.back()[0].IsOverflow());
}

TEST(RBinIndexMultiRange, Multi)
{
   const auto index0 = RBinIndex(0);
   const auto normal = CreateBinIndexRange(index0, RBinIndex(10), 0);
   const auto underflow = RBinIndex::Underflow();
   const RBinIndex invalid;
   const auto full = CreateBinIndexRange(underflow, invalid, /*nNormalBins=*/10);
   const RBinIndexMultiRange multi({normal, full});
   EXPECT_EQ(std::distance(multi.begin(), multi.end()), 120);

   const std::vector values(multi.begin(), multi.end());
   ASSERT_EQ(values.size(), 120);
   ASSERT_EQ(values[0].size(), 2);
   EXPECT_EQ(values[0][0], index0);
   EXPECT_EQ(values[0][1], underflow);
   ASSERT_EQ(values[1].size(), 2);
   EXPECT_EQ(values[1][0], index0);
   EXPECT_EQ(values[1][1], index0);
   ASSERT_EQ(values.back().size(), 2);
   EXPECT_EQ(values.back()[0], RBinIndex(9));
   EXPECT_TRUE(values.back()[1].IsOverflow());
}

TEST(RBinIndexMultiRange, MultiEmpty)
{
   const auto normal = CreateBinIndexRange(RBinIndex(0), RBinIndex(10), 0);
   const auto empty = CreateBinIndexRange(RBinIndex(0), RBinIndex(0), 0);
   const RBinIndexMultiRange multi({normal, empty, normal});
   EXPECT_EQ(multi.begin(), multi.end());
   EXPECT_EQ(std::distance(multi.begin(), multi.end()), 0);
}
