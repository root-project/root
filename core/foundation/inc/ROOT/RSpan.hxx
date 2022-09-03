/// \file ROOT/RSpan.hxx
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

#ifndef ROOT_RSpan
#define ROOT_RSpan

#include "RConfigure.h"

#if defined(R__HAS_STD_SPAN) || __cplusplus >= 202002L

#include <span>

#elif defined(R__HAS_STD_EXPERIMENTAL_SPAN)

#include <experimental/span>
namespace std {
  using template<class T> span = experimental::span<T>;

  // TODO: using make_view() overloads
}

#else
# include "ROOT/span.hxx"
#endif

#endif
