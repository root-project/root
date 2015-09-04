/// \file RArrayView.h
/// \ingroup Base StdExt
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-09-06

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RTupleApply
#define ROOT7_RTupleApply

#ifdef R__HAS_STD_TUPLE_APPLY

#include <tuple>

#elif defined(R__HAS_STD_EXPERIMENTAL_TUPLE_APPLY)

#include <experimental/tuple>
namespace std {
  using template <class F, class Tuple> constexpr decltype(auto) apply(F&& f, Tuple&& t);
}

#else
# include "ROOT/impl_tuple_apply.h"
#endif


#endif
