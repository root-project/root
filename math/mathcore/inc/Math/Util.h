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

   ///\class KahanSum
   /// The Kahan compensate summation algorithm significantly reduces the numerical error in the total obtained
   /// by adding a sequence of finite precision floating point numbers.
   /// This is done by keeping a separate running compensation (a variable to accumulate small errors).\n
   ///
   /// The intial values of the result and the correction are set to the default value of the type it hass been
   /// instantiated with.\n
   /// ####Examples:
   /// ~~~{.cpp}
   /// std::vector<double> numbers = {0.01, 0.001, 0.0001, 0.000001, 0.00000000001};
   /// ROOT::Math::KahanSum<double> k;
   /// k.Add(numbers);
   /// ~~~
   /// ~~~{.cpp}
   /// auto result = ROOT::Math::KahanSum<double>::Accumulate(numbers);
   /// ~~~
   template <class T>
   class KahanSum {
   public:
      /// Constructor accepting a initial value for the summation as parameter
      KahanSum(const T &initialValue = T{}) : fSum(initialValue) {}

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
         KahanSum init(initialValue);
         init.Add(begin, end);
         return init.fSum;
      }

      /// Return the result
      T Result() { return fSum; }

   private:
      T fSum{};
      T fCorrection{};
   };

   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Util */
