#include "Math/Util.h"
#include <vector>
#include <random>

#include "gtest/gtest.h"

///This is ROOT's old implementation before the vectorised version appeared
///It's here now to test that the new one yields the same results.
template <class T>
class LegacyKahanSum {
  public:
    /// Constructor accepting a initial value for the summation as parameter
    LegacyKahanSum(const T &initialValue = T{}) : fSum(initialValue) {}

    /// Single element accumulated addition.
    void Add(const T &x)
    {
      auto y = x - fCorrection;
      auto t = fSum + y;
      fCorrection = (t - fSum) - y;
      fSum = t;
    }

    /// Iterate over a datastructure referenced by a pointer and accumulate on the exising result
    template <class Iterator>
    void Add(const Iterator begin, const Iterator end)
    {
        static_assert(!std::is_same<decltype(*begin), T>::value,
            "Iterator points to an element of the different type than the KahanSum class");
        for (auto it = begin; it != end; it++) this->Add(*it);
    }

    /// Iterate over a datastructure referenced by a pointer and return the result of its accumulation.
    /// Can take an initial value as third parameter.
    template <class Iterator>
    static T Accumulate(const Iterator begin, const Iterator end, const T &initialValue = T{})
    {
        static_assert(!std::is_same<decltype(*begin), T>::value,
            "Iterator points to an element of the different type than the KahanSum class");
        LegacyKahanSum init(initialValue);
        init.Add(begin, end);
        return init.fSum;
    }

    /// Return the result
    T Result() { return fSum; }

  private:
    T fSum{};
    T fCorrection{};
};


TEST(KahanTest, LegacySum)
{
   std::vector<double> numbers = {0.01, 0.001, 0.0001, 0.000001, 0.00000000001};
   LegacyKahanSum<double> k;
   k.Add(numbers.begin(), numbers.end());
   auto result = LegacyKahanSum<double>::Accumulate(numbers.begin(), numbers.end());
   EXPECT_FLOAT_EQ(k.Result(), result);

   LegacyKahanSum<double> k2;
   LegacyKahanSum<double> k3(1);
   k2.Add(1);
   k2.Add(numbers.begin(), numbers.end());
   k3.Add(numbers.begin(), numbers.end());
   EXPECT_FLOAT_EQ(k2.Result(), k3.Result());

}


TEST(KahanTest, Compensation)
{
  std::vector<double> numbers(10, 0.1); // = 1.
  numbers.resize(1010, 1.E-18);// = 1. + 1.E-15, if not catastrophic cancellation

  ASSERT_FLOAT_EQ(std::accumulate(numbers.begin(), numbers.end(), 0.), 1.)
      << "Compensation fails with standard sum.";

  auto result = LegacyKahanSum<double>::Accumulate(numbers.begin(), numbers.end());
  EXPECT_FLOAT_EQ(result, 1. + 1.E-15) << "Kahan compensation works";
}


TEST(KahanTest, VectorisableVsLegacy)
{
  std::default_random_engine engine(12345u);
  //Should more or less accumulate:
  std::uniform_real_distribution<double> realLarge(1.E-13, 1.);
  //None of these should be possible to accumulate:
  constexpr double a = 1.E-20, b = 1.E-16;
  std::uniform_real_distribution<double> realSmall(a, b);

  std::vector<double> summableNumbers;
  std::vector<double> allNumbers;

  for (unsigned int i=0; i<1000; ++i) {
    const double large = realLarge(engine);
    summableNumbers.push_back(large);
    allNumbers.push_back(large);
    for (unsigned int j=0; j<1000; ++j) {
      const double small = realSmall(engine);
      allNumbers.push_back(small);
    }
  }


  // Test that normal summation has catastrophic cancellation, we are actually testing something here:
  const double summableNormal = std::accumulate(summableNumbers.begin(), summableNumbers.end(), 0.);
  const double allNormal = std::accumulate(allNumbers.begin(), allNumbers.end(), 0.);
  ASSERT_FLOAT_EQ(summableNormal, allNormal) << "Assert that small numbers disappear because of catastrophic cancellation.";


  // Test that legacy implementation does better
  const double summableLegacy =
      LegacyKahanSum<double>::Accumulate(summableNumbers.begin(), summableNumbers.end());
  const double allLegacy =
        LegacyKahanSum<double>::Accumulate(allNumbers.begin(), allNumbers.end());
  EXPECT_FLOAT_EQ(summableNormal, summableLegacy)
      << "Test that legacy Kahan works on numbers summable without errors.";
  // Expect to miss 1.E6 numbers that are on average equal to mean({a,b})
  constexpr double expectedCancellationError = 1.E6 * 0.5*(a+b);
  EXPECT_NEAR(allLegacy-allNormal, expectedCancellationError, expectedCancellationError/100.)
      << "Test that legacy Kahan doesn't miss the numbers that std::accumulate misses.";


  // Test that vectorisable Kahan yields identical results when used with 1 accumulator
  auto Kahan1Acc = ROOT::Math::KahanSum<>::Accumulate(allNumbers.begin(), allNumbers.end());
  EXPECT_FLOAT_EQ(allLegacy, Kahan1Acc.Sum()) << "New implementation with 1 accumulator identical.";


  // Test with 4 accumulators
  ROOT::Math::KahanSum<double, 4> kahan4AccSummable;
  for (unsigned int i=0; i<summableNumbers.size(); ++i) {
    kahan4AccSummable.AddIndexed(summableNumbers[i], i);
  }

  ROOT::Math::KahanSum<double, 4> kahan4AccAll;
  for (unsigned int i=0; i<allNumbers.size(); ++i) {
    kahan4AccAll.AddIndexed(allNumbers[i], i);
  }
  EXPECT_FLOAT_EQ(summableLegacy, kahan4AccSummable.Sum()) << "Both Kahans identical on summable.";
  EXPECT_FLOAT_EQ(allLegacy, kahan4AccAll.Sum()) << "Both Kahans identical on numbers with cancellation.";


  // Test with 2 accumulators
  ROOT::Math::KahanSum<double, 2> kahan2AccAll;
  for (unsigned int i=0; i<allNumbers.size(); ++i) {
    kahan2AccAll.AddIndexed(allNumbers[i], i);
  }
  EXPECT_FLOAT_EQ(kahan2AccAll.Sum(), kahan4AccAll.Sum()) << "Kahan(2,4) identical.";
  EXPECT_NEAR(kahan2AccAll.Carry(), kahan4AccAll.Carry(), 1.E-12) << "Kahan(2,4) identical.";


  // Test with 8 accumulators
  ROOT::Math::KahanSum<double, 8> kahan8AccAll;
  for (unsigned int i=0; i<allNumbers.size(); ++i) {
    kahan8AccAll.AddIndexed(allNumbers[i], i);
  }
  EXPECT_FLOAT_EQ(kahan8AccAll.Sum(), kahan4AccAll.Sum()) << "Kahan(8,4) identical.";
  EXPECT_NEAR(kahan8AccAll.Carry(), kahan4AccAll.Carry(), 1.E-12) << "Kahan(8,4) identical.";


  // Test different filling methods
  ROOT::Math::KahanSum<double, 4> allVecKahan2;
  allVecKahan2.Add(allNumbers);
  EXPECT_FLOAT_EQ(allVecKahan2.Sum(), kahan4AccAll.Sum()) << "Kahan from container.";
  EXPECT_FLOAT_EQ(allVecKahan2.Carry(), kahan4AccAll.Carry()) << "Kahan from container.";


  ROOT::Math::KahanSum<double, 4> allVecKahan3;
  allVecKahan3.Add(allNumbers.begin(), allNumbers.end());
  EXPECT_FLOAT_EQ(allVecKahan3.Sum(), kahan4AccAll.Sum()) << "Kahan from iterators.";
  EXPECT_FLOAT_EQ(allVecKahan3.Carry(), kahan4AccAll.Carry()) << "Kahan from iterators.";


  auto allVecKahan4 = ROOT::Math::KahanSum<double, 4>::Accumulate(allNumbers.begin(), allNumbers.end());
  EXPECT_FLOAT_EQ(allVecKahan4.Sum(), kahan4AccAll.Sum()) << "Kahan from Accumulate().";
  EXPECT_FLOAT_EQ(allVecKahan4.Carry(), kahan4AccAll.Carry()) << "Kahan from Accumulate().";


  // Test adding an offset
  auto allVecKahan5 = ROOT::Math::KahanSum<double, 4>::Accumulate(allNumbers.begin(), allNumbers.end(), 10.);
  EXPECT_FLOAT_EQ(allVecKahan5.Sum(), kahan4AccAll.Sum() + 10.) << "Initial value works.";
}

