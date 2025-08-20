/// \file ROOT/RNTupleUtils.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2025-07-31

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleUtils
#define ROOT_RNTupleUtils

#include <ROOT/RError.hxx>

#include <cstddef>
#include <memory>
#include <string_view>

namespace ROOT {

class RLogChannel;

namespace Internal {

/// Log channel for RNTuple diagnostics.
ROOT::RLogChannel &NTupleLog();

template <typename T>
auto MakeAliasedSharedPtr(T *rawPtr)
{
   const static std::shared_ptr<T> fgRawPtrCtrlBlock;
   return std::shared_ptr<T>(fgRawPtrCtrlBlock, rawPtr);
}

/// Make an array of default-initialized elements. This is useful for buffers that do not need to be initialized.
///
/// With C++20, this function can be replaced by std::make_unique_for_overwrite<T[]>.
template <typename T>
std::unique_ptr<T[]> MakeUninitArray(std::size_t size)
{
   // DO NOT use std::make_unique<T[]>, the array elements are value-initialized!
   return std::unique_ptr<T[]>(new T[size]);
}

/// Check whether a given string is a valid name according to the RNTuple specification
RResult<void> EnsureValidNameForRNTuple(std::string_view name, std::string_view where);

} // namespace Internal
} // namespace ROOT

#endif
