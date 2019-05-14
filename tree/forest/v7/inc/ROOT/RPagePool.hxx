/// \file ROOT/RPagePool.hxx
/// \ingroup Forest ROOT7
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
#include <ROOT/RForestUtil.hxx>

#include <cstddef>
#include <vector>

namespace ROOT {
namespace Experimental {

namespace Detail {

class RColumn;

// clang-format off
/**
\class ROOT::Experimental::Detail::RPagePool
\ingroup Forest
\brief A thread-safe cache of column pages.

The page pool encapsulated memory management for data written into a tree or read from a tree. Adding and removing
pages is thread-safe. All pages have the same size, which means different pages do not necessarily contain the same
number of elements. Multiple page caches can coexist.

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

   /// Get a new, empty page from the cache. Return a "null Page" if there is no more free space.
   RPage ReservePage(RColumn* column);
   /// Registers a page that has previously been acquired by ReservePage() and was meanwhile filled with content.
   void CommitPage(const RPage& page);
   /// Tries to find the page corresponding to column and index in the cache. On cache miss, load the page
   /// from the PageSource attached to the column and put it in the cache.
   RPage GetPage(RColumn* column, ForestSize_t index);
   /// Give back a page to the pool. There must not be any pointers anymore into this page.
   void ReleasePage(const RPage &page);
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
