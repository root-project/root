#include <ROOT/RHistUtils.hxx>

#include <gtest/gtest.h>

#ifndef TYPED_TEST_SUITE
#define TYPED_TEST_SUITE TYPED_TEST_CASE
#endif

template <typename T>
class RHistAtomic : public testing::Test {};

using AtomicTypes = testing::Types<char, short, int, long, long long, float, double>;
TYPED_TEST_SUITE(RHistAtomic, AtomicTypes);

TYPED_TEST(RHistAtomic, AtomicInc)
{
   TypeParam a = 1;
   ROOT::Experimental::Internal::AtomicInc(&a);
   EXPECT_EQ(a, 2);
}

TYPED_TEST(RHistAtomic, AtomicAdd)
{
   TypeParam a = 1;
   const TypeParam b = 2;
   ROOT::Experimental::Internal::AtomicAdd(&a, b);
   EXPECT_EQ(a, 3);
}

TEST(AtomicAdd, FloatDouble)
{
   float a = 1;
   const double b = 2;
   ROOT::Experimental::Internal::AtomicAdd(&a, b);
   EXPECT_EQ(a, 3);
}
