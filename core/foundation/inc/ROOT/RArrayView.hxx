/// \file ROOT/RArrayView.h
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

#ifndef ROOT_RArrayView
#define ROOT_RArrayView

#include "RConfigure.h"

#ifdef R__HAS_STD_ARRAY_VIEW

#include <array_view>

#elif defined(R__HAS_STD_EXPERIMENTAL_ARRAY_VIEW)

#include <experimental/array_view>
namespace std {
  using template<class T> array_view = experimental::array_view<T>;

  // TODO: using make_view() overloads
}

#else
# include "ROOT/rhysd_array_view.hxx"
#endif

#endif
