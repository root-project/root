/// \file ROOT/RNotFn.hxx
/// \ingroup Base StdExt
/// \author Danilo Piparo, Enrico Guiraud
/// \date 2018-01-19

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNotFn
#define ROOT_RNotFn

#include <functional>

// Backport if not_fn is not available.
// libc++ does not define __cpp_lib_not_fn.
// Assume we have not_fn if libc++ is compiled with C++14 and up.
#if !defined(__cpp_lib_not_fn) && !(defined(_LIBCPP_VERSION) && __cplusplus > 201103L)

#define R__NOTFN_BACKPORT

#include <type_traits> // std::decay
#include <utility>     // std::forward, std::declval

namespace std {

namespace __ROOT_noinline {
template <typename F>
class not_fn_t {
   std::decay_t<F> fFun;

public:
   explicit not_fn_t(F &&f) : fFun(std::forward<F>(f)) {}
   not_fn_t(not_fn_t &&h) = default;
   not_fn_t(const not_fn_t &f) = default;

   template <class... Args>
   auto operator()(Args &&... args) & -> decltype(
      !std::declval<std::result_of_t<std::decay_t<F>(Args...)>>())
   {
      return !fFun(std::forward<Args>(args)...);
   }
   template <class... Args>
   auto operator()(Args &&... args) const & -> decltype(
      !std::declval<std::result_of_t<std::decay_t<F> const(Args...)>>())
   {
      return !fFun(std::forward<Args>(args)...);
   }
};
}


template <typename F>
__ROOT_noinline::not_fn_t<F> not_fn(F &&f)
{
   return __ROOT_noinline::not_fn_t<F>(std::forward<F>(f));
}
}

#endif

#endif
