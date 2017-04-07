/* @(#)root/core/cont:$Id$ */
// Author: Danilo Piparo November 2015

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSeq
#define ROOT_TSeq

#include <iterator>
#include <type_traits>

/**
\class ROOT::TSeq
\brief A pseudo container class which is a generator of indices.

\tparam T Type of the numerical sequence.
\ingroup Containers
A pseudo container class which is a generator of indices. The model is the `xrange`
built-in function of Python.
Possible usages:
Loop on a sequence of integers
~~~{.cpp}
   for (auto i : TSeqI(10)) {
      cout << "Element " << i << endl;
   }
~~~
Loop on a sequence of integers in steps
~~~{.cpp}
   for (auto i : TSeqI(-5, 29, 6)) {
      cout << "Element " << i << endl;
   }
~~~
Loop backwards on a sequence of integers
~~~{.cpp}
   for (auto i : TSeqI(50, 30, -3)) {
      cout << "Element " << i << endl;
   }
~~~
Use an stl algorithm, for_each
~~~{.cpp}
   TSeqUL ulSeq(2,30,3);
   std::for_each(std::begin(ulSeq),std::end(ulSeq),[](ULong_t i){cout << "For each: " << i <<endl;});
~~~
Random access:
~~~{.cpp}
   cout << "Random access: 3rd element is " << ulSeq[2] << endl;
~~~
A function to create sequences inferring the type:
~~~{.cpp}
   for (auto i : MakeSeq(1000000000000UL, 1000000000003UL)) {
      cout << "Element " << i << endl;
   }
~~~

**/

namespace ROOT {

   template<class T>
   class TSeq {
   private:
      void checkIntegralType() {
         static_assert(std::is_integral<T>::value, "Only integral types are supported.");
      }
      const T fBegin;
      const T fEnd;
      const T fStep;
   public:
      using value_type = T;
      using difference_type = typename std::make_signed<T>::type;

      TSeq(T theEnd): fBegin(), fEnd(theEnd), fStep(1) {
         checkIntegralType();
      }
      TSeq(T theBegin, T theEnd, T theStep = 1):
        fBegin(theBegin), fEnd(theEnd), fStep(theStep) {
         checkIntegralType();
      }

      class iterator: public std::iterator<std::random_access_iterator_tag, T, difference_type> {
      private:
         T fCounter;
         T fStep;
      public:
         iterator(T start, T step): fCounter(start), fStep(step) {}
         T operator*() const {
            return fCounter;
         }
         // equality
         bool operator==(const iterator &other) const {
            return fCounter == other.fCounter;
         }
         // inequality
         bool operator!=(const iterator &other) const {
            return fCounter != other.fCounter;
         }
         // sum with integer
         iterator operator+(difference_type v) const {
            return iterator(fCounter + v * fStep, fStep);
         }
         // difference with integer
         iterator operator-(difference_type v) const {
            return iterator(fCounter - v * fStep, fStep);
         }
         // distance
         difference_type operator-(const iterator &other) const {
            return (fCounter - other.fCounter) / fStep;
         }
         // increments
         iterator &operator++() {
            fCounter += fStep;
            return *this;
         }
         iterator operator++(int) {
            iterator tmp(*this);
            operator++();
            return tmp;
         }
         // decrements
         iterator &operator--() {
            fCounter -= fStep;
            return *this;
         }
         iterator operator--(int) {
            iterator tmp(*this);
            operator--();
            return tmp;
         }
         // compound assignments
         iterator &operator+=(const difference_type& v) {
            *this = *this + v;
            return *this;
         }
         iterator &operator-=(const difference_type& v) {
            *this = *this - v;
            return *this;
         }
         // comparison operators
         bool operator <(const iterator &other) const {
             return (other - *this) > 0;
         }
         bool operator >(const iterator &other) const {
             return other < *this;
         }
         bool operator <=(const iterator &other) const {
             return !(*this > other);
         }
         bool operator >=(const iterator &other) const {
             return !(other > *this);
         }
         // subscript operator
         const T operator[](const difference_type& v) const{
             return *(*this + v);
         }
      };

      iterator begin() const {
         return iterator(fBegin, fStep);
      }
      iterator end() const {
         auto isStepMultiple = (fEnd - fBegin) % fStep == 0;
         auto theEnd = isStepMultiple ? fEnd : fStep * (((fEnd - fBegin) / fStep) + 1) + fBegin;
         return iterator(theEnd, fStep);
      }

      T const &front() const {
         return fBegin;
      }

      T operator[](T s) const {
         return s * fStep + fBegin;
      }

      std::size_t size() const {
         return end() - begin();
      }

      T step() const {
         return fStep;
      }

      bool empty() const {
         return fEnd == fBegin;
      }

   };

   using TSeqI = TSeq<int>;
   using TSeqU = TSeq<unsigned int>;
   using TSeqL = TSeq<long>;
   using TSeqUL = TSeq<unsigned long>;

   template<class T>
   TSeq<T> MakeSeq(T end)
   {
      return TSeq<T>(end);
   }

   template<class T>
   TSeq<T> MakeSeq(T begin, T end, T step = 1)
   {
      return TSeq<T>(begin, end, step);
   }

}

#include <sstream>

////////////////////////////////////////////////////////////////////////////////
/// Print a TSeq at the prompt:

namespace cling {
   template<class T>
   std::string printValue(ROOT::TSeq<T> *val)
   {
      std::ostringstream ret;
      ret << "A sequence of values: " << *val->begin()
          << " <= i < " << *val->end();
      auto step = val->step();
      if (1 != step)
          ret << " in steps of " <<  step;
      return ret.str();
   }
}

#endif
