/// \file ROOT/RPagePool.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPagePool
#define ROOT7_RPagePool

#include <ROOT/RPage.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <cstddef>
#include <vector>

namespace ROOT {
namespace Experimental {

namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RPagePool
\ingroup NTuple
\brief A thread-safe cache of column pages.

The page pool provides memory tracking for data written into an ntuple or read from an ntuple. Adding and removing
pages is thread-safe. The page pool does not allocate the memory -- allocation and deallocation is performed by the
page storage, which might do it in a way optimized to the backing store (e.g., mmap()).
Multiple page caches can coexist.

TODO(jblomer): it should be possible to register pages and to find them by column and index; this would
facilitate pre-filling a cache, e.g. by read-ahead.
*/
// clang-format on
class RPagePool {
private:
   /// TODO(jblomer): should be an efficient index structure that allows
   ///   - random insert
   ///   - random delete
   ///   - searching by page
   ///   - searching by tree index
   std::vector<RPage> fPages;
   std::vector<std::uint32_t> fReferences;

public:
   RPagePool() = default;
   RPagePool(const RPagePool&) = delete;
   RPagePool& operator =(const RPagePool&) = delete;
   ~RPagePool() = default;

   /// Adds a new page to the pool. The new page has its reference counter set to 1.
   void RegisterPage(const RPage &page);
   /// Tries to find the page corresponding to column and index in the cache. If the page is found, its reference
   /// counter is increased
   RPage GetPage(ColumnId_t columnId, NTupleSize_t index);
   /// Give back a page to the pool and decrease the reference counter. There must not be any pointers anymore into
   /// this page. If the reference counter drops to zero, the page is removed from the page pool and the return value
   /// is true, indicating that its memory can be freed.
   bool ReturnPage(const RPage &page);
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
