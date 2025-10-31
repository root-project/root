#include "hist_test.hxx"

#include <cstddef>

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

// AtomicInc is implemented in terms of AtomicAdd, so it's sufficient to stress one of them.
TYPED_TEST(RHistAtomic, StressAtomicAdd)
{
   static constexpr TypeParam Addend = 1;
   static constexpr std::size_t NThreads = 4;
   // Reduce number of additions for char to avoid overflow.
   static constexpr std::size_t NAddsPerThread = sizeof(TypeParam) == 1 ? 20 : 8000;
   static constexpr std::size_t NAdds = NThreads * NAddsPerThread;

   TypeParam a = 0;
   StressInParallel(NThreads, [&] {
      for (std::size_t i = 0; i < NAddsPerThread; i++) {
         ROOT::Experimental::Internal::AtomicAdd(&a, Addend);
      }
   });

   EXPECT_EQ(a, NAdds * Addend);
}

TEST(AtomicAdd, FloatDouble)
{
   float a = 1;
   const double b = 2;
   ROOT::Experimental::Internal::AtomicAdd(&a, b);
   EXPECT_EQ(a, 3);
}
