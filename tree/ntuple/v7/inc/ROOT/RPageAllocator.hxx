/// \file ROOT/RPageAllocator.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-06-25
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageAllocator
#define ROOT7_RPageAllocator

#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPage.hxx>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <functional>

namespace ROOT {
namespace Experimental {
namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageDeleter
\ingroup NTuple
\brief A closure that can free the memory associated with a mapped page

The page pool, once taken ownership of pages, must know how to free them. When registering a new page with
the page pool, the passed page deleter encapsulates that knowledge.
*/
// clang-format on
class RPageDeleter {
private:
   /// The callable that is suppped to free the given page; it is called with fUserData as the second argument.
   std::function<void(const RPage &page, void *userData)> fFnDelete;
   /// Optionally additional information necessary to free resources associated with a page.  For instance,
   /// when the page is read from a TKey, user data points to the ROOT object created for reading, which needs to be
   /// freed as well.
   void *fUserData;

public:
   RPageDeleter() : fFnDelete(), fUserData(nullptr) {}
   explicit RPageDeleter(decltype(fFnDelete) fnDelete) : fFnDelete(fnDelete), fUserData(nullptr) {}
   RPageDeleter(decltype(fFnDelete) fnDelete, void *userData) : fFnDelete(fnDelete), fUserData(userData) {}
   RPageDeleter(const RPageDeleter &other) = default;
   RPageDeleter &operator =(const RPageDeleter &other) = default;
   ~RPageDeleter() = default;

   void operator()(const RPage &page) { fFnDelete(page, fUserData); }
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageAllocatorHeap
\ingroup NTuple
\brief Uses standard C++ memory allocation for the column data pages

The page allocator acquires and releases memory for pages.  It does not populate the pages, the returned pages
are empty but guaranteed to have enough contiguous space for the given number of elements.  While a common
concrete implementation uses the heap, other implementations are possible, e.g. using arenas or mmap().
*/
// clang-format on
class RPageAllocatorHeap {
public:
   /// Reserves memory large enough to hold nElements of the given size. The page is immediately tagged with
   /// a column id.
   static RPage NewPage(ColumnId_t columnId, std::size_t elementSize, std::size_t nElements);
   /// Releases the memory pointed to by page and resets the page's information
   static void DeletePage(const RPage &page);
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageAllocatorCache
\ingroup NTuple
\brief Reuse memory allocations from a small thread-local cache

This allocator returns pages that have at least the required capacity. `DeletePage()` does not immediately deallocate
memory; instead, pages are returned to a thread-local cache, dropping the smallest allocated buffer if the cache is full.
If a previous page cannot be recycled, `AllocT` is used to allocate memory.
*/
// clang-format on
template <typename AllocT = RPageAllocatorHeap, std::size_t CacheSize = 16>
class RPageAllocatorCache {
   thread_local static struct RPageCache {
      std::deque<RPage> fPages;
      ~RPageCache()
      {
         for (const auto &page : fPages)
            AllocT::DeletePage(page);
      }
   } gCache;

public:
   /// Request a page from the cache with at least the specified capacity. If there isn't a suitable page in the cache,
   /// a new allocation is made.
   static RPage NewPage(ColumnId_t columnId, std::size_t elementSize, std::size_t nElements)
   {
      const auto szPage = elementSize * nElements;
      auto it = std::find_if(gCache.fPages.begin(), gCache.fPages.end(),
                             [szPage](const auto &p) { return p.GetMaxBytes() >= szPage; });
      if (it == gCache.fPages.end())
         return AllocT::NewPage(columnId, elementSize, nElements);
      RPage page(columnId, it->GetBuffer(), it->GetElementSize(), it->GetMaxElements());
      gCache.fPages.erase(it);
      return page;
   }
   /// Return a page to the cache for reuse. If the cache is full, the smallest memory buffer is deallocated.
   static void DeletePage(const RPage &page)
   {
      gCache.fPages.emplace_front(kInvalidColumnId, page.GetBuffer(), page.GetElementSize(), page.GetMaxElements());
      if (gCache.fPages.size() <= CacheSize)
         return;
      auto itSmallest = std::min_element(gCache.fPages.begin(), gCache.fPages.end(), [](const auto &a, const auto &b) {
         return a.GetMaxBytes() < b.GetMaxBytes();
      });
      AllocT::DeletePage(*itSmallest);
      gCache.fPages.erase(itSmallest);
   }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
