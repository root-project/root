/// \file ROOT/RRangeCast.hxx
/// \ingroup Base StdExt
/// \author Jonas Rembser <jonas.rembser@cern.ch>
/// \date 2021-08-04

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RRangeCast
#define ROOT_RRangeCast

#include "ROOT/RSpan.hxx"

#include <cassert>
#include <iterator>
#include <type_traits>
#include <utility>

namespace ROOT {
namespace Internal {

template <typename T>
struct RBaseType {
   using type = typename std::remove_pointer<typename std::decay<T>::type>::type;
};

#if (__cplusplus < 201700L)

template <typename T, bool isDynamic = true, bool isPolymorphic = std::is_polymorphic<RBaseType<T>>::value>
struct RCast {
   template <typename U>
   static T cast(U &&u)
   {
      return dynamic_cast<T>(u);
   }
};

template <typename T>
struct RCast<T, false, false> {
   template <typename U>
   static T cast(U &&u)
   {
      return static_cast<T>(u);
   }
};

template <typename T>
struct RCast<T, false, true> {
   template <typename U>
   static T cast(U &&u)
   {
      assert(dynamic_cast<T>(u));
      return static_cast<T>(u);
   }
};

#endif

// For SFINAE-based checks for the existence of the `begin` and `end` methods.
template <typename T>
constexpr auto hasBeginEnd(int) -> decltype(std::begin(std::declval<T>()), std::end(std::declval<T>()), true)
{
   return true;
}

template <typename>
constexpr bool hasBeginEnd(...)
{
   return false;
}

template <typename T, typename WrappedIterator_t, bool isDynamic>
class TypedIter {

public:
   TypedIter(WrappedIterator_t const &iter) : fIter{iter} {}

   TypedIter &operator++()
   {
      ++fIter;
      return *this;
   }
   TypedIter operator++(int)
   {
      TypedIter tmp(*this);
      operator++();
      return tmp;
   }
   bool operator==(const TypedIter &rhs) const { return fIter == rhs.fIter; }
   bool operator!=(const TypedIter &rhs) const { return fIter != rhs.fIter; }

   void swap(TypedIter &other) { fIter.swap(other.fIter); }

   // We want to know at compile time whether dynamic_cast or static_cast is
   // used. First of all to avoid overhead, but also to avoid a compiler
   // error when using dynamic_cast on a non-polymorphic class. In C++17,
   // this can be done easily with `if constexpr`, but for the older
   // standards we have to use a more verbose alternative. Both ways are
   // explicitely implemented for different standards, so that when the
   // minimum C++ standard for ROOT is raised to C++17 it's easy to remember
   // that we can avoid much boilerplate code in this file.
#if (__cplusplus < 201700L)
   T operator*() { return ROOT::Internal::RCast<T, isDynamic>::cast(*fIter); }
#else
   T operator*()
   {
      if constexpr (isDynamic) {
         return dynamic_cast<T>(*fIter);
      } else {
         if constexpr (std::is_polymorphic<RBaseType<T>>::value) {
            assert(dynamic_cast<T>(*fIter));
         }
         return static_cast<T>(*fIter);
      }
   }
#endif

private:
   WrappedIterator_t fIter;
};

} // namespace Internal

/// Wraps any collection that can be used in range-based loops and applies
/// `static_cast<T>` or `dynamic_cast<T>` to each element.
/// \tparam T The new type to convert to.
/// \tparam isDynamic If `true`, `dynamic_cast` is used, otherwise `static_cast` is used.
/// \tparam Range_t The type of the input range, which should be usually a reference type to avoid copying.
template <typename T, bool isDynamic, typename Range_t>
class RRangeCast {

public:
   RRangeCast(Range_t &&inputRange) : fInputRange{inputRange}
   {
      static_assert(ROOT::Internal::hasBeginEnd<Range_t>(0),
                    "Type with no `begin` or `end` method passed to `RRangeCast`");
   }

   using const_iterator = Internal::TypedIter<T, decltype(std::cbegin(std::declval<Range_t>())), isDynamic>;
   const_iterator begin() const { return std::cbegin(fInputRange); }
   const_iterator end() const { return std::cend(fInputRange); }

   using iterator = Internal::TypedIter<T, decltype(std::begin(std::declval<Range_t>())), isDynamic>;
   iterator begin() { return std::begin(fInputRange); }
   iterator end() { return std::end(fInputRange); }

private:
   Range_t fInputRange;
};

/// Takes any collection that can be used in range-based loops and applies
/// static_cast<T> to each element. This function can be used for example to
/// cast all objects in a RooAbsCollection when iterating over them.
/// Example:
/// ~~~{.cpp}
/// class ClassA {
/// public:
///    virtual ~ClassA() {}
/// };
/// class ClassB : public ClassA {
/// };
///
/// B b1, b2, b3;
/// std::vector<A const*> vec{&b1, &b2, &b3};
///
/// for(auto *b : ROOT::RangeStaticCast<B const*>(vec)) {
///    // do something with b
/// }
/// ~~~
/// Make sure to not use `auto const&` in the range-based loop, as this will
/// cause a range-loop-bind-reference warning with the clang compiler.
template <typename T, typename Range_t>
RRangeCast<T, false, Range_t> RangeStaticCast(Range_t &&coll)
{
   return std::forward<Range_t>(coll);
}
// Overload for C-style arrays. It's not possible to make an overload of the
// RRangeCast constructor itself, because when the C-style array is forwarded
// it might decay depending on the compiler version.
template <typename T, typename U, std::size_t N>
RRangeCast<T, false, std::span<U>> RangeStaticCast(U (&arr)[N])
{
   return std::span<U>(arr, arr + N);
}

/// Takes any collection that can be used in range-based loops and applies
/// dynamic_cast<T> to each element. This function can be used for example to
/// cast all objects in a RooAbsCollection when iterating over them.
/// Example:
/// ~~~{.cpp}
///
/// class ClassA {
/// public:
///    virtual ~ClassA() {}
/// };
/// class ClassB : public ClassA {
/// };
///
/// A a1, a2;
/// B b1, b2, b3;
/// std::vector<A const*> vec{&b1, &a1, &b2, &a2, &b3};
///
/// for(auto *b : ROOT::RangeDynCast<B const*>(vec)) {
///    if(b) {
///       // do something with b
///    }
/// }
/// ~~~
/// Make sure to not use `auto const&` in the range-based loop, as this will
/// cause a range-loop-bind-reference warning with the clang compiler.
template <typename T, typename Range_t>
RRangeCast<T, true, Range_t> RangeDynCast(Range_t &&coll)
{
   return std::forward<Range_t>(coll);
}
// Overload for C-style arrays. It's not possible to make an overload of the
// RRangeCast constructor itself, because when the C-style array is forwarded
// it might decay depending on the compiler version.
template <typename T, typename U, std::size_t N>
RRangeCast<T, true, std::span<U>> RangeDynCast(U (&arr)[N])
{
   return std::span<U>(arr, arr + N);
}

} // namespace ROOT

#endif
