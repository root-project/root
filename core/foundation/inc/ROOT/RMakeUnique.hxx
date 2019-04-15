/// \file ROOT/RMakeUnique.hxx
/// \ingroup Base StdExt
/// \author Danilo Piparo
/// \date 2017-09-22

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RMakeUnique
#define ROOT_RMakeUnique

#include <memory>

#if __cplusplus < 201402L && !defined(_MSC_VER)

#include <type_traits>
#include <utility>

namespace ROOT {
namespace Detail {
// Inspired from abseil
template <typename T>
struct RMakeUniqueResult {
   using scalar = std::unique_ptr<T>;
};
template <typename T>
struct RMakeUniqueResult<T[]> {
   using array = std::unique_ptr<T[]>;
};
template <typename T, size_t N>
struct RMakeUniqueResult<T[N]> {
   using invalid = void;
};
} // namespace Detail
} // namespace ROOT

namespace std {

// template <typename T, typename... Args, typename std::enable_if<!std::is_array<T>::value, int>::type = 0>
template <typename T, typename... Args>
typename ROOT::Detail::RMakeUniqueResult<T>::scalar make_unique(Args &&... args)
{
   return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T>
typename ROOT::Detail::RMakeUniqueResult<T>::array make_unique(std::size_t size)
{
   return std::unique_ptr<T>(new typename std::remove_extent<T>::type[size]());
}

template <typename T, typename... Args>
typename ROOT::Detail::RMakeUniqueResult<T>::invalid make_unique(Args &&...) = delete;


} // namespace std
#endif

#endif
