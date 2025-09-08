#include "hist_test.hxx"

#include <iterator>
#include <vector>

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
   static constexpr std::size_t UnderflowIndex = -3;
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

   const std::vector binIndexes(full.begin(), full.end());
   ASSERT_EQ(binIndexes.size(), 12);
   EXPECT_TRUE(binIndexes.front().IsUnderflow());
   EXPECT_TRUE(binIndexes.back().IsOverflow());
}
