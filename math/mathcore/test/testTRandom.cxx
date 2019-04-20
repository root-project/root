#include "TRandom3.h"

#include "gtest/gtest.h"

TEST(TRandom3, Seeds)
{
   TRandom3 rand3;
   auto s0 = rand3.GetState();
   auto p0 = rand3.Poisson(10);
   auto s1 = rand3.GetState();
   auto p1 = rand3.Poisson(10);
   auto s2 = rand3.GetState();
   auto p2 = rand3.Poisson(10);
   
   rand3.SetState(s0);
   EXPECT_DOUBLE_EQ(p0, rand3.Poisson(10));

   rand3.SetState(s2);
   EXPECT_DOUBLE_EQ(p2, rand3.Poisson(10));

   rand3.SetState(s1);
   EXPECT_DOUBLE_EQ(p1, rand3.Poisson(10));

}