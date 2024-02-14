// std
#include <random>

// ROOT
#include "TRandom3.h"

#include "gtest/gtest.h"

TEST(TRandom3Test, TRandom3vsStdMt19937)
{
  unsigned int seed = 28u;

  std::mt19937 gen_std(seed);
  TRandom3 gen_root(seed);

  // Test TRandom3 generator against other implementation of Mersenne Twister
  // 32-bit algoritm.
  //
  // Test inspired by code of Giovanni Cerretani in this issue:
  // https://its.cern.ch/jira/browse/ROOT-9733
  // https://github.com/root-project/root/issues/14581
  for (size_t generation = 0; generation < 5; ++generation) {
    unsigned int iteration = 0;
    while (++iteration) {
      double rnd_std = gen_std() * 2.3283064365386963e-10;
      double rnd_root = gen_root.Rndm();

      ASSERT_EQ(rnd_std, rnd_root);
    }
  }
}
