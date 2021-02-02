// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 14 15:44:38 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Utility functions for all ROOT Math classes

#ifndef ROOT_Math_Util
#define ROOT_Math_Util

#include <string>
#include <sstream>

#include <cmath>
#include <limits>
#include <numeric>


// This can be protected against by defining ROOT_Math_VecTypes
// This is only used for the R__HAS_VECCORE define
// and a single VecCore function in EvalLog
#ifndef ROOT_Math_VecTypes
#include "Types.h"
#endif


// for defining unused variables in the interfaces
//  and have still them in the documentation
#define MATH_UNUSED(var)   (void)var


namespace ROOT {

   namespace Math {

   /**
      namespace defining Utility functions needed by mathcore
   */
   namespace Util {

   /**
      Utility function for conversion to strings
   */
   template <class T>
   std::string ToString(const T &val)
   {
      std::ostringstream buf;
      buf << val;

      std::string ret = buf.str();
      return ret;
   }

   /// safe evaluation of log(x) with a protections against negative or zero argument to the log
   /// smooth linear extrapolation below function values smaller than  epsilon
   /// (better than a simple cut-off)

   template<class T>
   inline T EvalLog(T x) {
      static const T epsilon = T(2.0 * std::numeric_limits<double>::min());
#ifdef R__HAS_VECCORE
      T logval = vecCore::Blend<T>(x <= epsilon, x / epsilon + std::log(epsilon) - T(1.0), std::log(x));
#else
      T logval = x <= epsilon ? x / epsilon + std::log(epsilon) - T(1.0) : std::log(x);
#endif
      return logval;
   }

   } // end namespace Util

   /// \class KahanSum
   /// The Kahan summation is a compensated summation algorithm, which significantly reduces numerical errors
   /// when adding a sequence of finite-precision floating point numbers.
   /// This is done by keeping a separate running compensation (a variable to accumulate small errors).
   ///
   /// ### Auto-vectorisable accumulation
   /// This class can internally use multiple accumulators (template parameter `N`).
   /// When filled from a collection that supports index access from a *contiguous* block of memory,
   /// compilers such as gcc, clang and icc can auto-vectorise the accumulation. This happens by cycling
   /// through the internal accumulators based on the value of "`index % N`", so `N` accumulators can be filled from a block
   /// of `N` numbers in a single instruction.
   ///
   /// The usage of multiple accumulators might slightly increase the precision in comparison to the single-accumulator version
   /// with `N = 1`.
   /// This depends on the order and magnitude of the numbers being accumulated. Therefore, in rare cases, the accumulation
   /// result can change *in dependence of N*, even when the data are identical.
   /// The magnitude of such differences is well below the precision of the floating point type, and will therefore mostly show
   /// in the compensation sum(see Carry()). Increasing the number of accumulators therefore only makes sense to
   /// speed up the accumulation, but not to increase precision.
   ///
   /// \param T The type of the values to be accumulated.
   /// \param N Number of accumulators. Defaults to 1. Ideal values are the widths of a vector register on the relevant architecture.
   /// Depending on the instruction set, good values are:
   /// - AVX2-float: 8
   /// - AVX2-double: 4
   /// - AVX512-float: 16
   /// - AVX512-double: 8
   ///
   /// ### Examples
   ///
   /// ~~~{.cpp}
   /// std::vector<double> numbers(1000);
   /// for (std::size_t i=0; i<1000; ++i) {
   ///    numbers[i] = rand();
   /// }
   ///
   /// ROOT::Math::KahanSum<double, 4> k;
   /// k.Add(numbers.begin(), numbers.end());
   /// // or
   /// k.Add(numbers);
   /// ~~~
   /// ~~~{.cpp}
   /// double offset = 10.;
   /// auto result = ROOT::Math::KahanSum<double, 4>::Accumulate(numbers.begin(), numbers.end(), offset);
   /// ~~~
   template<typename T = double, unsigned int N = 1>
   class KahanSum {
     public:
       /// Initialise the sum.
       /// \param[in] initialValue Initialise with this value. Defaults to 0.
       KahanSum(T initialValue = T{}) {
         fSum[0] = initialValue;
         std::fill(std::begin(fSum)+1, std::end(fSum), 0.);
         std::fill(std::begin(fCarry), std::end(fCarry), 0.);
       }

       /// Constructor to create a KahanSum from another KahanSum with a different number of accumulators
       template <unsigned int M>
       KahanSum(KahanSum<T,M> const& other) {
         fSum[0] = other.Sum();
         fCarry[0] = other.Carry();
         std::fill(std::begin(fSum)+1, std::end(fSum), 0.);
         std::fill(std::begin(fCarry)+1, std::end(fCarry), 0.);
       }

       /// Single-element accumulation. Will not vectorise.
       void Add(T x) {
          auto y = x - fCarry[0];
          auto t = fSum[0] + y;
          fCarry[0] = (t - fSum[0]) - y;
          fSum[0] = t;
       }


       /// Accumulate from a range denoted by iterators.
       ///
       /// This function will auto-vectorise with random-access iterators.
       /// \param[in] begin Beginning of a range. Needs to be a random access iterator for automatic
       /// vectorisation, because a contiguous block of memory needs to be read.
       /// \param[in] end   End of the range.
       template <class Iterator>
       void Add(Iterator begin, Iterator end) {
           static_assert(std::is_floating_point<
               typename std::remove_reference<decltype(*begin)>::type>::value,
               "Iterator needs to point to floating-point values.");
           const std::size_t n = std::distance(begin, end);

           for (std::size_t i=0; i<n; ++i) {
             AddIndexed(*(begin++), i);
           }
       }


       /// Fill from a container that supports index access.
       /// \param[in] inputs Container with index access such as std::vector or array.
       template<class Container_t>
       void Add(const Container_t& inputs) {
         static_assert(std::is_floating_point<typename Container_t::value_type>::value,
             "Container does not hold floating-point values.");
         for (std::size_t i=0; i < inputs.size(); ++i) {
           AddIndexed(inputs[i], i);
         }
       }


       /// Iterate over a range and return an instance of a KahanSum.
       ///
       /// See Add(Iterator,Iterator) for details.
       /// \param[in] begin Beginning of a range.
       /// \param[in] end   End of the range.
       /// \param[in] initialValue Optional initial value.
       template <class Iterator>
       static KahanSum<T, N> Accumulate(Iterator begin, Iterator end,
           T initialValue = T{}) {
           KahanSum<T, N> theSum(initialValue);
           theSum.Add(begin, end);

           return theSum;
       }


       /// Add `input` to the sum.
       ///
       /// Particularly helpful when filling from a for loop.
       /// This function can be inlined and auto-vectorised if
       /// the index parameter is used to enumerate *consecutive* fills.
       /// Use Add() or Accumulate() when no index is available.
       /// \param[in] input Value to accumulate.
       /// \param[in] index Index of the value. Determines internal accumulator that this
       /// value is added to. Make sure that consecutive fills have consecutive indices
       /// to make a loop auto-vectorisable. The actual value of the index does not matter,
       /// as long as it is consecutive.
       void AddIndexed(T input, std::size_t index) {
         const unsigned int i = index % N;
         const T y = input - fCarry[i];
         const T t = fSum[i] + y;
         fCarry[i] = (t - fSum[i]) - y;
         fSum[i] = t;
       }

       /// \return Compensated sum.
       T Sum() const {
         return std::accumulate(std::begin(fSum), std::end(fSum), 0.);
       }

       /// \return Compensated sum.
       T Result() const {
         return Sum();
       }

       /// Auto-convert to type T
       operator T() const {
         return Sum();
       }

       /// \return The sum used for compensation.
       T Carry() const {
         return std::accumulate(std::begin(fCarry), std::end(fCarry), 0.);
       }

       /// Add `arg` into accumulator. Does not vectorise.
       KahanSum<T, N>& operator+=(T arg) {
         Add(arg);
         return *this;
       }

       /// Add `arg` into accumulator. Does not vectorise.
       template<typename U>
       KahanSum& operator+=(const KahanSum<U>& arg) {
         Add(arg.Sum());
         fCarry[0] += arg.Carry();
         return *this;
       }

       /// Subtract other KahanSum. Does not vectorise.
       ///
       /// This is only meaningfull when both the sum and carry of each operand are of similar order of magnitude.
       KahanSum<T, N>& operator-=(KahanSum<T, N> const& other) {
         fSum[0]   -= other.Sum();
         fCarry[0] -= other.Carry();
         // add zero such that if the summed carry is large enough to be partly represented in the sum,
         // it will happen
         Add(T{});
         return *this;
       }

     private:
       T fSum[N];
       T fCarry[N];
   };

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Util */
