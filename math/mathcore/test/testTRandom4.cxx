// std
#include <random>

// ROOT
#include "TRandom4.h"

#include "gtest/gtest.h"

TEST(TRandom4Test, TRandom4vsStdMt19937)
{
   // Test TRandom4 generator against other implementation of Mersenne Twister
   // 32-bit algoritm.
   //
   // Test inspired by code of Giovanni Cerretani in this issue:
   // https://its.cern.ch/jira/browse/ROOT-9733
   // https://github.com/root-project/root/issues/14581
   // https://github.com/root-project/root/pull/14702
   for (unsigned int seed = 1; seed < 10; ++seed) { // We do not test seed = 0 since it's a special case of TRandom4
      std::mt19937 gen_std(seed);
      TRandom4 gen_root(seed);
      unsigned short iteration = 0;
      while (++iteration) { // Let's test the first 65636 items in the sequence
         const double rnd_std = (gen_std() + 0.5) * 2.3283064365386963e-10;
         const double rnd_root = gen_root.Rndm();
         ASSERT_EQ(rnd_std, rnd_root);
      }
   }
}
