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

#if __cplusplus < 201402L && !defined(_MSC_VER)

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

// Implementation taken from https://isocpp.org/files/papers/N3656.txt

namespace std {
template <class T>
struct _Unique_if {
   typedef unique_ptr<T> _Single_object;
};

template <class T>
struct _Unique_if<T[]> {
   typedef unique_ptr<T[]> _Unknown_bound;
};

template <class T, size_t N>
struct _Unique_if<T[N]> {
   typedef void _Known_bound;
};

template <class T, class... Args>
typename _Unique_if<T>::_Single_object make_unique(Args &&... args)
{
   return unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
typename _Unique_if<T>::_Unknown_bound make_unique(size_t n)
{
   typedef typename remove_extent<T>::type U;
   return unique_ptr<T>(new U[n]());
}

template <class T, class... Args>
typename _Unique_if<T>::_Known_bound make_unique(Args &&...) = delete;
}

#endif

#endif
