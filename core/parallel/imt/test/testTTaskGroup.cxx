#include "TROOT.h"

#include "gtest/gtest.h"

#ifdef R__USE_IMT
#include "ROOT/TTaskGroup.hxx"

using namespace ROOT::Experimental;

int Fibonacci(int n)
{
   if (n < 2) {
      return n;
   } else {
      int x, y;
      ROOT::Experimental::TTaskGroup tg;
      tg.Run([&] { x = Fibonacci(n - 1); });
      tg.Run([&] { y = Fibonacci(n - 2); });
      tg.Wait();
      return x + y;
   }
}

TEST(TTaskGroup, NestedExecution)
{
   ROOT::EnableImplicitMT(4);
   EXPECT_EQ(Fibonacci(7), 13);
}

#endif