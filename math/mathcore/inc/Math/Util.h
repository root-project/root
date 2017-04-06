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
   inline double EvalLog(double x)
   {
   // evaluate the log
#ifdef __CINT__
      static const double epsilon = 2. * 2.2250738585072014e-308;
#else
      static const double epsilon = 2. * std::numeric_limits<double>::min();
#endif
      if (x <= epsilon)
         return x / epsilon + std::log(epsilon) - 1;
      else
         return std::log(x);
   }

   } // end namespace Util

   ///\class KahanSum
   /// The Kahan compensate summation algorithm significantly reduces the numerical error in the total obtained
   /// by adding a sequence of finite precision floating point numbers.
   /// This is done by keeping a separate running compensation (a variable to accumulate small errors).\n
   ///
   /// The intial values of the result and the correction are set to the default value of the type it hass been instantiated with.\n
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
      /// Single element accumulated addition.
      void Add(const T &x)
      {
         auto y = x - fCorrection;
         auto t = fSum + y;
         fCorrection = (t - fSum) - y;
         fSum = t;
      }

      /// Iterate over an iterable container of values and accumulate on the exising result 
      template<class Container>
      void Add(const Container &elements)
      {
         static_assert(!std::is_same<decltype(++(elements.begin()), elements.end(), elements.front()), T>::value, "argument is not a container of the same type as the KahanSum class");
         for (auto e : elements) this->Add(e);
      }

      /// Iterate over an iterable and return the result of its accumulation
      template<class Container>
      static T Accumulate(const Container &elements)
      {
         static_assert(!std::is_same<decltype(++(elements.begin()), elements.end(), elements.front()), T>::value, "argument is not a container of the same type as the KahanSum class");
         KahanSum init;
         init.Add(elements);
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
