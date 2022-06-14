
// ROOT
#include "TRandom.h"
#include "TRandom1.h"
#include "TRandom2.h"
#include "TRandom3.h"
#include "TRandomGen.h"

// TMVA
#include "TMVA/Tools.h"

// Stdlib
#include <algorithm>
#include <iostream>

// External
#include "gtest/gtest.h"

using TRandom1StdGen = TMVA::RandomGenerator<TRandom1, UInt_t, kMaxUInt << 2>;
using TRandom2StdGen = TMVA::RandomGenerator<TRandom2, UInt_t, kMaxUInt << 2>;
using TRandom3StdGen = TMVA::RandomGenerator<TRandom3>;
using TRandomMixMaxStdGen = TMVA::RandomGenerator<TRandomMixMax>;
using TRandomMixMax17StdGen = TMVA::RandomGenerator<TRandomMixMax17>;
using TRandomMixMax256StdGen = TMVA::RandomGenerator<TRandomMixMax256>;

template <typename URNG>
void print_urng(URNG urng, int n = 10)
{
   std::uniform_int_distribution<UInt_t> dist{2, 5};

   for (int i = 0; i < n; ++i) {
      std::cout << dist(urng) << ", ";
   }
   std::cout << std::endl;
}

// Test that generation compiles
void test_print(int n)
{

   std::cout << "Generating numbers from TRandom1StdGen: ";
   print_urng(TRandom1StdGen{1}, n);
   std::cout << "Generating numbers from TRandom2StdGen: ";
   print_urng(TRandom2StdGen{1}, n);
   std::cout << "Generating numbers from TRandom3StdGen: ";
   print_urng(TRandom3StdGen{1}, n);

   std::cout << "Generating numbers from TRandomMixMaxStdGen: ";
   print_urng(TRandomMixMaxStdGen{1}, n);
   std::cout << "Generating numbers from TRandomMixMax17StdGen: ";
   print_urng(TRandomMixMax17StdGen{1}, n);
   std::cout << "Generating numbers from TRandomMixMax256StdGen: ";
   print_urng(TRandomMixMax256StdGen{1}, n);

   std::cout << std::endl;
}

// Make sure that discard skips N steps
void test_discard(int n)
{
   TRandom3StdGen rng1{100};
   TRandom3StdGen rng2{100};

   std::uniform_int_distribution<UInt_t> dist{2, 5};

   for (int i = 0; i < n; ++i) {
      rng1();
   }
   rng2.discard(n);

   for (int i = 0; i < 10; ++i) {
      if (!(dist(rng1) == dist(rng2))) {
         throw std::runtime_error("");
      }
   }

   std::cout << "Discard test success!" << std::endl;
}

// Make sure the example code in the documentation compiles
void test_example()
{
   std::cout << std::endl;
   std::cout << "Shuffling vector {0, 1, 2, 3, 4, 5}" << std::endl;

   int seed = 0;
   std::vector<double> v{0, 1, 2, 3, 4, 5};
   TMVA::RandomGenerator<TRandom3> rng(seed);
   std::shuffle(v.begin(), v.end(), rng);

   for (auto e : v) {
      std::cout << e << ", ";
   }
   std::cout << std::endl;
}

TEST(RandomGenerator, print)
{
   test_print(10);
}

TEST(RandomGenerator, discard)
{
   test_discard(10);
}

TEST(RandomGenerator, example)
{
   test_example();
}
