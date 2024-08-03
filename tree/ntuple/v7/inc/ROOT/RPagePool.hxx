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
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <cstddef>
#include <mutex>
#include <vector>

namespace ROOT {
namespace Experimental {

namespace Internal {

// clang-format off
/**
\class ROOT::Experimental::Internal::RPagePool
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
   friend class RPageRef;

   /// TODO(jblomer): should be an efficient index structure that allows
   ///   - random insert
   ///   - random delete
   ///   - searching by page
   ///   - searching by tree index
   std::vector<RPage> fPages;
   std::vector<std::int32_t> fReferences;
   RPageAllocator *fPageAllocator; ///< The allocator is used to release the added pages
   std::mutex fLock;

   /// Give back a page to the pool and decrease the reference counter. There must not be any pointers anymore into
   /// this page. If the reference counter drops to zero, the page pool might decide to call the deleter given in
   /// during registration. Called by the RPageRef destructor.
   void ReturnPage(RPage &page);

public:
   explicit RPagePool(RPageAllocator *pageAllocator) : fPageAllocator(pageAllocator) {}
   RPagePool(const RPagePool&) = delete;
   RPagePool& operator =(const RPagePool&) = delete;
   ~RPagePool();

   /// Adds a new page to the pool. Upon registration, the page pool takes ownership of the page's memory.
   /// The new page has its reference counter set to 1.
   RPageRef RegisterPage(RPage &page);
   /// Like RegisterPage() but the reference counter is initialized to 0
   void PreloadPage(RPage &page);
   /// Tries to find the page corresponding to column and index in the cache. If the page is found, its reference
   /// counter is increased
   RPageRef GetPage(ColumnId_t columnId, NTupleSize_t globalIndex);
   RPageRef GetPage(ColumnId_t columnId, RClusterIndex clusterIndex);
};

// clang-format off
/**
\class ROOT::Experimental::Internal::RPageRef
\ingroup NTuple
\brief Reference to a page stored in the page pool

The referenced page knows about its page pool and decreases the reference counter on destruction.
*/
// clang-format on
class RPageRef {
   friend class RPagePool;

   RPage fPage;
   RPagePool *fPagePool = nullptr;

   // Called as delegated constructor and directly by the page pool
   RPageRef(const RPage &page, RPagePool *pagePool) : fPagePool(pagePool)
   {
      // We leave the fPage::fPageAllocator member unset (nullptr), since fPage is a non-owning view on the page
      fPage.fColumnId = page.fColumnId;
      fPage.fBuffer = page.fBuffer;
      fPage.fElementSize = page.fElementSize;
      fPage.fNElements = page.fNElements;
      fPage.fMaxElements = page.fMaxElements;
      fPage.fRangeFirst = page.fRangeFirst;
      fPage.fClusterInfo = page.fClusterInfo;
   }

public:
   RPageRef() = default;
   RPageRef(const RPageRef &other) = delete;
   RPageRef &operator=(const RPageRef &other) = delete;

   RPageRef(RPageRef &&other) : RPageRef(other.fPage, other.fPagePool) { other.fPagePool = nullptr; }

   RPageRef &operator=(RPageRef &&other)
   {
      if (this != &other) {
         std::swap(fPage, other.fPage);
         std::swap(fPagePool, other.fPagePool);
      }
      return *this;
   }

   ~RPageRef()
   {
      if (fPagePool)
         fPagePool->ReturnPage(fPage);
   }

   /// Used by the friend virtual page source to map the physical column and cluster IDs to ther virtual counterparts
   void ChangeIds(DescriptorId_t columnId, DescriptorId_t clusterId)
   {
      fPage.fColumnId = columnId;
      fPage.fClusterInfo = RPage::RClusterInfo(clusterId, fPage.fClusterInfo.GetIndexOffset());
   }

   const RPage &Get() const { return fPage; }
};

} // namespace Internal

} // namespace Experimental
} // namespace ROOT

#endif
