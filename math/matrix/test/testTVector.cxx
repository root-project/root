#include "TVectorT.h"

#include "gtest/gtest.h"

TEST(TVectorT, ListInit)
{
   TVectorD vec{1., 2., 3.};
   ASSERT_EQ(vec.GetNoElements(), 3);
   for (int i = 0; i < 3; ++i)
      EXPECT_EQ(vec[i], (double)i + 1);

   // Get around SBO, force heap allocation
   TVectorF vec2{1., 2., 3., 4., 5., 6., 7.};
   ASSERT_EQ(vec2.GetNoElements(), 7);
   for (int i = 0; i < 7; ++i)
      EXPECT_EQ(vec2[i], (float)i + 1);
}
