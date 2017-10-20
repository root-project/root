/// \file ROOT/RMakeUnique.h
/// \ingroup Base StdExt
/// \author Danilo Piparo
/// \date 2017-09-22

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RMakeUnique
#define ROOT_RMakeUnique

#include <memory>

#if __cplusplus < 201402L && !defined(_MSC_VER)

#include <utility>

namespace std {

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args)
{
   return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
}
#endif

#endif
