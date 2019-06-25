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

class RColumn;

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
   void* fMemory;
   std::size_t fPageSize;
   std::size_t fNPages;
   /// TODO(jblomer): should be an efficient index structure that allows
   ///   - random insert
   ///   - random delete
   ///   - searching by page
   ///   - searching by tree index
   std::vector<RPage> fPages;
   std::vector<std::uint32_t> fReferences;

public:
   RPagePool(std::size_t pageSize, std::size_t nPages);
   RPagePool(const RPagePool&) = delete;
   RPagePool& operator =(const RPagePool&) = delete;
   ~RPagePool();

   /// Get a new, empty page from the cache. Return a "null page" if there is no more free space.  Used for writing.
   RPage ReservePage(RColumn* column);
   /// Registers a page that has previously been acquired by ReservePage() and was meanwhile filled with content.
   void CommitPage(const RPage& page);
   /// Tries to find the page corresponding to column and index in the cache. On cache miss, load the page
   /// from the PageSource attached to the column and put it in the cache.
   RPage GetPage(RColumn* column, NTupleSize_t index);
   /// Give back a page to the pool. There must not be any pointers anymore into this page.
   void ReleasePage(const RPage &page);
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
