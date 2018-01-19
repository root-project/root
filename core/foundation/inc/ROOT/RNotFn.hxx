/// \file ROOT/RNotFn.h
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

#if __cplusplus < 201703L && !defined(_MSC_VER)

#include <type_traits> // std::decay
#include <utility>     // std::forward, std::declval

namespace std {

namespace Detail {
template <typename F>
class not_fn_t {
   typename std::decay<F>::type fFun;

public:
   explicit not_fn_t(F &&f) : fFun(std::forward<F>(f)) {}
   not_fn_t(not_fn_t &&h) = default;
   not_fn_t(const not_fn_t &f) = default;

   template <class... Args>
   auto operator()(Args &&... args) & -> decltype(
      !std::declval<typename std::result_of<typename std::decay<F>::type(Args...)>::type>())
   {
      return !fFun(std::forward<Args>(args)...);
   }
   template <class... Args>
   auto operator()(Args &&... args) const & -> decltype(
      !std::declval<typename std::result_of<typename std::decay<F>::type const(Args...)>::type>())
   {
      return !fFun(std::forward<Args>(args)...);
   }
};
}


template <typename F>
Detail::not_fn_t<F> not_fn(F &&f)
{
   return Detail::not_fn_t<F>(std::forward<F>(f));
}
}

#else
#include <functional>
#endif

#endif
