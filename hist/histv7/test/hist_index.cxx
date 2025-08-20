#include "hist_test.hxx"

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
