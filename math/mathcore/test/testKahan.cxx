#include "Math/Util.h"
#include <vector>
#include <random>
#include <cmath>  // std::nextafter, INFINITY

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

constexpr double smallLower = 1.E-20, smallUpper = 1.E-16;

std::tuple<std::vector<double>, std::vector<double>> generateSummableNumbers()
{
   std::default_random_engine engine(12345u);
   //Should more or less accumulate:
   std::uniform_real_distribution<double> realLarge(1.E-13, 1.);
   //None of these should be possible to accumulate:
   std::uniform_real_distribution<double> realSmall(smallLower, smallUpper);

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

   return {summableNumbers, allNumbers};
}


TEST(KahanTest, VectorisableVsLegacy)
{
   std::vector<double> summableNumbers;
   std::vector<double> allNumbers;
   std::tie(summableNumbers, allNumbers) = generateSummableNumbers();

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
  constexpr double expectedCancellationError = 1.E6 * 0.5*(smallLower+smallUpper);
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

// The KahanAddSubtractAssignTest suite tests some edge cases of operator+= and operator-= with another KahanSum as parameter

TEST(KahanAddSubtractAssignTest, BelowFloatPrecision)
{
   // When we add something and subsequently subtract the same (and vice versa), we expect to return
   // to the starting state
   double large_number = 1e18;
   double small_number = 1;
   ROOT::Math::KahanSum<double, 1> sum(large_number);
   ROOT::Math::KahanSum<double, 1> small_sum(small_number);

   sum += small_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), -small_number); // note that the carry stores the remainder of the sum as its negative
   sum -= small_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), 0);

   // the other way around
   sum -= small_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), small_number);
   sum += small_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), 0);
}

TEST(KahanAddSubtractAssignTest, LargestLargeEnoughCarry)
{
   // Check if the largest large enough carry is added to the sum instead of kept in the carry
   // when it could have been added to the sum
   double large_number = 1e18;
   ROOT::Math::KahanSum<double, 1> sum(large_number);

   double largest_large_enough_carry = std::nextafter(large_number, INFINITY) - large_number;

   ROOT::Math::KahanSum<double, 1> large_carry_sum( largest_large_enough_carry);

   sum += large_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number + largest_large_enough_carry);
   EXPECT_EQ(sum.Carry(), 0);
   sum -= large_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), 0);

   // the other way around
   sum -= large_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number - largest_large_enough_carry);
   EXPECT_EQ(sum.Carry(), 0);
   sum += large_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), 0);
}

TEST(KahanAddSubtractAssignTest, SmallestLargeEnoughCarry)
{
   // Check if the smallest large enough carry is added to the number instead of kept in the
   // carry when it could have been added to the sum. Unlike test LargestLargeEnoughCarry, in this
   // case there will also be a non-zero carry.
   double large_number = 1e18;
   ROOT::Math::KahanSum<double, 1> sum(large_number);

   double largest_large_enough_carry = std::nextafter(large_number, INFINITY) - large_number;
   double a_bit_too_small_carry = largest_large_enough_carry / 2;
   double smallest_large_enough_carry = std::nextafter(a_bit_too_small_carry, INFINITY);
   // here we check that these are the right numbers:
   EXPECT_EQ(large_number + largest_large_enough_carry, large_number + smallest_large_enough_carry);
   EXPECT_GT(large_number + largest_large_enough_carry, large_number);

   ROOT::Math::KahanSum<double, 1> smallest_large_carry_sum(smallest_large_enough_carry);

   sum += smallest_large_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number + smallest_large_enough_carry);
   // Note: because carry terms are stored negatively, the positive carry value here actually means that
   // the summed floating point result is higher than the corresponding algebraic result!
   EXPECT_EQ(sum.Carry(), largest_large_enough_carry - smallest_large_enough_carry);
   sum -= smallest_large_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), 0);

   // the other way around
   sum -= smallest_large_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number - smallest_large_enough_carry);
   EXPECT_EQ(sum.Carry(), -(largest_large_enough_carry - smallest_large_enough_carry));
   sum += smallest_large_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), 0);
}

TEST(KahanAddSubtractAssignTest, ABitTooSmallCarry)
{
   // Check what happens with the value just below the limit of being large enough to influence the normal floating point sum
   double large_number = 1e18;
   ROOT::Math::KahanSum<double, 1> sum(large_number);

   double largest_large_enough_carry = std::nextafter(large_number, INFINITY) - large_number;
   double a_bit_too_small_carry = largest_large_enough_carry / 2;

   ROOT::Math::KahanSum<double, 1> small_carry_sum(a_bit_too_small_carry);

   sum += small_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   // This time, the stored carry in sum is again negative because it still has to be added. This is
   // in contrast to the case above where the floating point sum already yielded a higher Sum value,
   // but in fact should be corrected downwards by the Carry term (which was hence positive in
   // SmallestLargeEnoughCarry).
   EXPECT_EQ(sum.Carry(), -a_bit_too_small_carry);
   sum += small_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number + 2 * a_bit_too_small_carry);
   EXPECT_EQ(sum.Carry(), 0);
   sum -= small_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), -a_bit_too_small_carry);
   sum -= small_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), 0);

   // compare to when just adding regular numbers (not KahanSum objects)
   auto compare_sum {sum};
   sum += small_carry_sum;
   compare_sum += a_bit_too_small_carry;
   EXPECT_EQ(sum.Sum(), compare_sum.Sum());
   EXPECT_EQ(sum.Carry(), compare_sum.Carry());
   sum += small_carry_sum;
   compare_sum += a_bit_too_small_carry;
   EXPECT_EQ(sum.Sum(), compare_sum.Sum());
   EXPECT_EQ(sum.Carry(), compare_sum.Carry());
   // reset state
   sum -= small_carry_sum;
   sum -= small_carry_sum;

   // the other way around
   sum -= small_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), a_bit_too_small_carry);
   sum -= small_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number - 2 * a_bit_too_small_carry);
   EXPECT_EQ(sum.Carry(), 0);
   sum += small_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), a_bit_too_small_carry);
   sum += small_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), 0);
}

TEST(KahanAddSubtractAssignTest, SubtractWithABitTooSmallCarry)
{
   // Subtract a large value with half of the largest large enough carry, which
   // is exactly one bit too small to be added to the large number in normal
   // floating point addition. Compare this (where possible) to behavior of a
   // regular Kahan sum.
   double large_number = 1e18;
   ROOT::Math::KahanSum<double, 1> sum(large_number);

   double largest_large_enough_carry = std::nextafter(large_number, INFINITY) - large_number;
   double a_bit_too_small_carry = largest_large_enough_carry / 2;

   // note: we initialize the carry with negative sign, meaning it should have been added, but hasn't yet
   ROOT::Math::KahanSum<double, 1> large_carry_sum(large_number, -a_bit_too_small_carry);

   sum -= large_carry_sum;
   // The carry completely disappears in this case, never to be seen again:
   EXPECT_EQ(sum.Sum(), 0);
   EXPECT_EQ(sum.Carry(), 0);
   sum += large_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   // Surprisingly, it also doesn't turn up in the carry after adding it back in:
   EXPECT_EQ(sum.Carry(), 0);

   // With the terms reversed, it gives the same resulting values as above:

   large_carry_sum -= sum;
   // This time we also do another sum to compare behavior of += for KahanSum vs double
   // arguments.
   ROOT::Math::KahanSum<double, 1> another_large_carry_sum(large_number, -a_bit_too_small_carry);
   another_large_carry_sum += -large_number;
   EXPECT_EQ(large_carry_sum.Sum(), 0);
   EXPECT_EQ(large_carry_sum.Carry(), 0);
   EXPECT_EQ(another_large_carry_sum.Sum(), 0);
   EXPECT_EQ(another_large_carry_sum.Carry(), 0);
   large_carry_sum += sum;
   another_large_carry_sum += large_number;
   EXPECT_EQ(large_carry_sum.Sum(), large_number);
   EXPECT_EQ(another_large_carry_sum.Sum(), large_number);
   // ... meaning we do not return to the original state after two opposite operations!
   EXPECT_EQ(large_carry_sum.Carry(), 0);
   // However, this is expected behavior, because it also happens to the regular
   // Kahan summation:
   EXPECT_EQ(another_large_carry_sum.Carry(), large_carry_sum.Carry());

   // the other way around (first + then -) also loses the small carry:
   ROOT::Math::KahanSum<double, 1> minus_large_carry_sum(-large_number, a_bit_too_small_carry);

   sum += minus_large_carry_sum;
   EXPECT_EQ(sum.Sum(), 0);
   EXPECT_EQ(sum.Carry(), 0);
   sum -= minus_large_carry_sum;
   EXPECT_EQ(sum.Sum(), large_number);
   EXPECT_EQ(sum.Carry(), 0);
}

TEST(KahanAddSubtractAssignTest, XMinusXIsZero)
{
   // x - x should always be zero
   double large_number = 1e18;
   double small_number = 1;
   ROOT::Math::KahanSum<double, 1> sum(large_number, -small_number);
   ROOT::Math::KahanSum<double, 1> sum2(large_number, -small_number);

   auto diff = sum - sum2;
   EXPECT_EQ(diff.Sum(), 0);
   EXPECT_EQ(diff.Carry(), 0);
}

TEST(KahanAddSubtractAssignTest, AddMinusXEqualsSubtractX)
{
   // y + (-x) should equal y - x
   double large_number = 1e18;
   double small_number = 1;
   ROOT::Math::KahanSum<double, 1> sum(large_number);
   ROOT::Math::KahanSum<double, 1> small_sum(small_number);

   auto addMinusX = sum + (-small_sum);
   auto subtractX = sum - small_sum;

   EXPECT_EQ(addMinusX.Sum(), subtractX.Sum());
   EXPECT_EQ(addMinusX.Carry(), subtractX.Carry());
}

TEST(KahanAddSubtractAssignTest, MultipleAccumulators)
{
   // Adding and subtracting also works for multiple accumulators, even mixing different N.
   // It doesn't vectorize, but that's only a performance issue, not a precision issue.
   ROOT::Math::KahanSum<double, 4> sum4Acc;
   ROOT::Math::KahanSum<double, 2> sum2Acc;

   std::vector<double> _;
   std::vector<double> allNumbers;
   std::tie(_, allNumbers) = generateSummableNumbers();

   sum4Acc.Add(allNumbers);
   sum2Acc.Add(allNumbers);

   auto total2_4 = sum2Acc + sum4Acc;
   auto total4_2 = sum4Acc + sum2Acc;

   // note that with different numbers of accumulators we expect floating-point equality,
   // like in the legacy test above, not exact equality
   EXPECT_FLOAT_EQ(total2_4.Sum(), total4_2.Sum());
   EXPECT_NEAR(total2_4.Carry(), total4_2.Carry(), 1e-12);

   auto diff2_4 = sum2Acc - sum4Acc;
   auto diff4_2 = sum4Acc - sum2Acc;

   // now the result should be (almost) zero
   EXPECT_NEAR(diff2_4.Sum(), 0, 1e-12);
   EXPECT_NEAR(diff4_2.Sum(), 0, 1e-12);
   // the carries as well
   EXPECT_NEAR(diff2_4.Carry(), 0, 1e-12);
   EXPECT_NEAR(diff4_2.Carry(), 0, 1e-12);
}
