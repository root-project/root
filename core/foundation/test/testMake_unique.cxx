#include "gtest/gtest.h"
#include <memory>

class A {
public:
static int fgN;
A(){fgN++;}
~A(){fgN--;}
};
int A::fgN = 0;


TEST(MakeUnique, Array)
{
   EXPECT_EQ(A::fgN, 0);
   {
      auto a = std::make_unique<A[]>(3u);
      EXPECT_EQ(A::fgN, 3);
   }
   EXPECT_EQ(A::fgN, 0);

   auto a = std::make_unique<int[]>(2u);
   a[0] = 42;
   a[1] = 7;
   EXPECT_EQ(a[0], 42);
   EXPECT_EQ(a[1], 7);
}