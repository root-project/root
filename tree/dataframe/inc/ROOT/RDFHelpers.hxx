// Author: Enrico Guiraud, Danilo Piparo CERN  02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// This header contains helper free functions that slim down RDataFrame's programming model

#ifndef ROOT_RDF_HELPERS
#define ROOT_RDF_HELPERS

#include <ROOT/TypeTraits.hxx>

#include <functional>
#include <type_traits>

namespace ROOT {
namespace Internal {
namespace RDF {
template <typename... ArgTypes, typename F>
std::function<bool(ArgTypes...)> NotHelper(ROOT::TypeTraits::TypeList<ArgTypes...>, F &&f)
{
   return std::function<bool(ArgTypes...)>([=](ArgTypes... args) mutable { return !f(args...); });
}

template <typename... ArgTypes, typename Ret, typename... Args>
std::function<bool(ArgTypes...)> NotHelper(ROOT::TypeTraits::TypeList<ArgTypes...>, Ret (*f)(Args...))
{
   return std::function<bool(ArgTypes...)>([=](ArgTypes... args) mutable { return !f(args...); });
}

template <typename I, typename T>
struct MakeVecHelper;

template <typename T, std::size_t... N>
struct MakeVecHelper<std::index_sequence<N...>, T> {
   template <std::size_t>
   using RepeatT = T;

   static std::vector<T> func(RepeatT<N>... args) { return {args...}; }
};
} // namespace RDF
} // namespace Internal

namespace RDF {
namespace RDFInternal = ROOT::Internal::RDF;
// clag-format off
/// Given a callable with signature bool(T1, T2, ...) return a callable with same signature that returns the negated result
///
/// The callable must have one single non-template definition of operator(). This is a limitation with respect to
/// std::not_fn, required for interoperability with RDataFrame.
// clang-format on
template <typename F,
          typename Args = typename ROOT::TypeTraits::CallableTraits<typename std::decay<F>::type>::arg_types_nodecay,
          typename Ret = typename ROOT::TypeTraits::CallableTraits<typename std::decay<F>::type>::ret_type>
auto Not(F &&f) -> decltype(RDFInternal::NotHelper(Args(), std::forward<F>(f)))
{
   static_assert(std::is_same<Ret, bool>::value, "RDF::Not requires a callable that returns a bool.");
   return RDFInternal::NotHelper(Args(), std::forward<F>(f));
}

#if R__HAS_VARIABLE_TEMPLATES
// clang-format off
/// MakeVec<N, T> is a callable that takes N arguments of type T and returns a std::vector<T> containing copies of the arguments.
///
/// Only available if ROOT has been compiled with C++14 support.
/// Note that the type of all columns that the callable is applied to must be exactly T.
/// Example usage ("triggerX" columns must be `float`, "triggers" is a `std::vector<float>`:
/// \code
/// df.Define("triggers", MakeVec<3, float>, {"trigger1", "trigger2", "trigger3"})
/// \endcode
// clang-format on
template <int N, typename T>
auto MakeVec = RDFInternal::MakeVecHelper<std::make_index_sequence<N>, T>::func;
#endif

} // namespace RDF
} // namespace ROOT
#endif
