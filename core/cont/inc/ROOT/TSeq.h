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

      TSeq(T end): fBegin(), fEnd(end), fStep(1) {
         checkIntegralType();
      }
      TSeq(T begin, T end, T step = 1):
         fBegin(begin), fEnd(end), fStep(step) {
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
         iterator &operator++() {
            fCounter += fStep;
            return *this;
         };
         iterator operator++(int) {
            iterator tmp(*this);
            operator++();
            return tmp;
         }
         bool operator==(const iterator &other) {
            return fCounter == other.fCounter;
         }
         bool operator!=(const iterator &other) {
            return fCounter != other.fCounter;
         }
         T operator+(int v) {
            return fCounter + v;
         }
         T operator-(int v) {
            return fCounter - v;
         }
         T operator+(const iterator &other) {
            return fCounter + other.fCounter;
         }
         T operator-(const iterator &other) {
            return fCounter - other.fCounter;
         }
         iterator &operator--() {
            fCounter -= fStep;
            return *this;
         }
         iterator operator--(int) {
            iterator tmp(*this);
            operator--();
            return tmp;
         }
      };

      iterator begin() const {
         return iterator(fBegin, fStep);
      }
      iterator end() const {
         auto isStepMultiple = (fEnd - fBegin) % fStep == 0;
         auto theEnd = isStepMultiple ? fEnd : fStep * ((T)((fEnd - fBegin) / fStep) + 1) + fBegin;
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
      ret << "Sequence of values. Begin: " << *val->begin()
          << " - End: " << *val->end()
          << " - Step: " <<  val->step();
      return ret.str();
   }
}

#endif
