/// \file ROOT/RNTupleUtils.hxx
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
#include <string>
#include <string_view>
#include <vector>

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

/// Used for string interning of repeated type names, field names, etc. The call operator returns for every
/// given string the one immutable pointer to that string in the pool. After calling Freeze(), no new string
/// must be added and the pool is safe to use from multiple threads.
class RStringPool {
   std::vector<std::unique_ptr<std::string>> fStrings; // String pointers are always valid and sorted by their pointee
   bool fFrozen = false;

   static bool Less(const std::unique_ptr<std::string> &a, const std::string_view &b) { return *a < b; }

public:
   RStringPool() = default;
   ~RStringPool() = default;
   RStringPool(const RStringPool &other) = delete;
   RStringPool(RStringPool &&other) = default;
   RStringPool &operator=(RStringPool &&other) = default;
   RStringPool &operator=(const RStringPool &other) = delete;

   /// Searches for str in the pool. If not present, adds the string to the pool before returning a pointer to it.
   /// Throws if frozen and str is not yet in the pool.
   const std::string *Intern(std::string_view str);
   void Freeze() { fFrozen = true; }
};

} // namespace Internal
} // namespace ROOT

#endif
