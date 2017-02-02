// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDFTRAITSUTILS
#define ROOT_TDFTRAITSUTILS

#include <functional>
#include <vector>
#include <type_traits> // std::decay

/// \cond HIDDEN_SYMBOLS

namespace ROOT {
namespace Internal {
namespace TDFTraitsUtils {
template <typename... Types>
struct TTypeList {
   static constexpr std::size_t fgSize = sizeof...(Types);
};

// extract parameter types from a callable object
template <typename T>
struct TFunctionTraits {
   using ArgTypes_t = typename TFunctionTraits<decltype(&T::operator())>::ArgTypes_t;
   using ArgTypesNoDecay_t = typename TFunctionTraits<decltype(&T::operator())>::ArgTypesNoDecay_t;
   using RetType_t = typename TFunctionTraits<decltype(&T::operator())>::RetType_t;
};

// lambdas and std::function
template <typename R, typename T, typename... Args>
struct TFunctionTraits<R (T::*)(Args...) const> {
   using ArgTypes_t = TTypeList<typename std::decay<Args>::type...>;
   using ArgTypesNoDecay_t = TTypeList<Args...>;
   using RetType_t = R;
};

// mutable lambdas and functor classes
template <typename R, typename T, typename... Args>
struct TFunctionTraits<R (T::*)(Args...)> {
   using ArgTypes_t = TTypeList<typename std::decay<Args>::type...>;
   using ArgTypesNoDecay_t = TTypeList<Args...>;
   using RetType_t = R;
};

// function pointers
template <typename R, typename... Args>
struct TFunctionTraits<R (*)(Args...)> {
   using ArgTypes_t = TTypeList<typename std::decay<Args>::type...>;
   using ArgTypesNoDecay_t = TTypeList<Args...>;
   using RetType_t = R;
};

// free functions
template <typename R, typename... Args>
struct TFunctionTraits<R (Args...)> {
   using ArgTypes_t = TTypeList<typename std::decay<Args>::type...>;
   using ArgTypesNoDecay_t = TTypeList<Args...>;
   using RetType_t = R;
};

// remove first type from TypeList
template <typename>
struct TRemoveFirst { };

template <typename T, typename... Args>
struct TRemoveFirst<TTypeList<T, Args...>> {
   using Types_t = TTypeList<Args...>;
};

// return wrapper around f that prepends an `unsigned int slot` parameter
template <typename R, typename F, typename... Args>
std::function<R(unsigned int, Args...)> AddSlotParameter(F f, TTypeList<Args...>)
{
   return [f](unsigned int, Args... a) -> R { return f(a...); };
}

// compile-time integer sequence generator
// e.g. calling TGenStaticSeq<3>::type() instantiates a TStaticSeq<0,1,2>
template <int...>
struct TStaticSeq { };

template <int N, int... S>
struct TGenStaticSeq : TGenStaticSeq<N - 1, N - 1, S...> { };

template <int... S>
struct TGenStaticSeq<0, S...> {
   using Type_t = TStaticSeq<S...>;
};

template <typename T>
struct TIsContainer {
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
             (std::is_same<decltype(pt->begin()), It_t>::value &&
              std::is_same<decltype(pt->end()), It_t>::value &&
              std::is_same<decltype(cpt->begin()), CIt_t>::value &&
              std::is_same<decltype(cpt->end()), CIt_t>::value &&
              std::is_same<decltype(**pi), V_t &>::value &&
              std::is_same<decltype(**pci), V_t const &>::value);
   }

   template <typename A>
   static constexpr bool Test(...)
   {
      return false;
   }

   static const bool fgValue = Test<Test_t>(nullptr);
};

} // end NS TDFTraitsUtils

} // end NS Internal

} // end NS ROOT

/// \endcond

#endif
