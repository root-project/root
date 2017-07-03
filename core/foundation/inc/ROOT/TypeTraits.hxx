// @(#)root/foundation:
// Author: Axel Naumann, Enrico Guiraud, June 2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TypeTraits
#define ROOT_TypeTraits

#include <memory> // shared_ptr, unique_ptr for IsSmartOrDumbPtr
#include <type_traits>
#include <vector> // for IsContainer

namespace ROOT {

/// ROOT type_traits extensions
namespace TypeTraits {

///\class ROOT::TypeTraits::
template <class T>
class IsSmartOrDumbPtr : public std::integral_constant<bool, std::is_pointer<T>::value> {
};

template <class P>
class IsSmartOrDumbPtr<std::shared_ptr<P>> : public std::true_type {
};

template <class P>
class IsSmartOrDumbPtr<std::unique_ptr<P>> : public std::true_type {
};

/// Check for container traits.
template <typename T>
struct IsContainer {
   using Test_t = typename std::decay<T>::type;

   template <typename A>
   static constexpr bool Test(A *pt, A const *cpt = nullptr, decltype(pt->begin()) * = nullptr,
                              decltype(pt->end()) * = nullptr, decltype(cpt->begin()) * = nullptr,
                              decltype(cpt->end()) * = nullptr, typename A::iterator *pi = nullptr,
                              typename A::const_iterator *pci = nullptr)
   {
      using It_t = typename A::iterator;
      using CIt_t = typename A::const_iterator;
      using V_t = typename A::value_type;
      return std::is_same<Test_t, std::vector<bool>>::value ||
             (std::is_same<decltype(pt->begin()), It_t>::value && std::is_same<decltype(pt->end()), It_t>::value &&
              std::is_same<decltype(cpt->begin()), CIt_t>::value && std::is_same<decltype(cpt->end()), CIt_t>::value &&
              std::is_same<decltype(**pi), V_t &>::value && std::is_same<decltype(**pci), V_t const &>::value);
   }

   template <typename A>
   static constexpr bool Test(...)
   {
      return false;
   }

   static constexpr bool value = Test<Test_t>(nullptr);
};

/// Lightweight storage for a collection of types.
/// Differently from std::tuple, no instantiation of objects of stored types is performed
template <typename... Types>
struct TypeList {
   static constexpr std::size_t list_size = sizeof...(Types);
};

/// Extract types from the signature of a callable object.
template <typename T>
struct CallableTraits {
   using arg_types = typename CallableTraits<decltype(&T::operator())>::arg_types;
   using arg_types_nodecay = typename CallableTraits<decltype(&T::operator())>::arg_types_nodecay;
   using ret_type = typename CallableTraits<decltype(&T::operator())>::ret_type;
};

// lambdas and std::function
template <typename R, typename T, typename... Args>
struct CallableTraits<R (T::*)(Args...) const> {
   using arg_types = TypeList<typename std::decay<Args>::type...>;
   using arg_types_nodecay = TypeList<Args...>;
   using ret_type = R;
};

// mutable lambdas and functor classes
template <typename R, typename T, typename... Args>
struct CallableTraits<R (T::*)(Args...)> {
   using arg_types = TypeList<typename std::decay<Args>::type...>;
   using arg_types_nodecay = TypeList<Args...>;
   using ret_type = R;
};

// function pointers
template <typename R, typename... Args>
struct CallableTraits<R (*)(Args...)> {
   using arg_types = TypeList<typename std::decay<Args>::type...>;
   using arg_types_nodecay = TypeList<Args...>;
   using ret_type = R;
};

// free functions
template <typename R, typename... Args>
struct CallableTraits<R(Args...)> {
   using arg_types = TypeList<typename std::decay<Args>::type...>;
   using arg_types_nodecay = TypeList<Args...>;
   using ret_type = R;
};

// Return first of a variadic list of types.
template <typename T, typename... Rest>
struct TakeFirstType {
   using type = T;
};

template <typename... Types>
using TakeFirstType_t = typename TakeFirstType<Types...>::type;

// Remove first type from a variadic list of types, return a TypeList containing the rest.
// e.g. RemoveFirst_t<A,B,C> is TypeList<B,C>
template <typename T, typename... Rest>
struct RemoveFirst {
   using type = TypeList<Rest...>;
};

template <typename... Args>
using RemoveFirst_t = typename RemoveFirst<Args...>::type;

/// Return first of possibly many template parameters.
/// For non-template types, the result is the type itself.
/// e.g. TakeFirstParameter<U<A,B>> is A
///      TakeFirstParameter<T> is T
template <typename T>
struct TakeFirstParameter {
   using type = void;
};

template <template <typename...> class Template, typename T, typename... Rest>
struct TakeFirstParameter<Template<T, Rest...>> {
   using type = T;
};

template <typename T>
using TakeFirstParameter_t = typename TakeFirstParameter<T>::type;

/// Remove first of possibly many template parameters.
/// e.g. RemoveFirstParameter_t<U<A,B>> is U<B>
template <typename>
struct RemoveFirstParameter {
};

template <typename T, template <typename...> class U, typename... Rest>
struct RemoveFirstParameter<U<T, Rest...>> {
   using type = U<Rest...>;
};

template <typename T>
using RemoveFirstParameter_t = typename RemoveFirstParameter<T>::type;

} // ns TypeTraits
} // ns ROOT
#endif // ROOT_TTypeTraits
