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
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace ROOT {
namespace Experimental {

namespace Internal {

// clang-format off
/**
\class ROOT::Experimental::Internal::RPagePool
\ingroup NTuple
\brief A thread-safe cache of pages loaded from the page source.

The page pool is used as a cache for pages loaded from a page source.
In this way, identical page needed at the same time, only need to be loaded once.
Page sources also use the page pool to stage (preload) pages unsealed by IMT tasks.
*/
// clang-format on
class RPagePool {
   friend class RPageRef;

public:
   // Search key for a set of pages covering the same column and in-memory target type.
   // Within the set of pages, one needs to find the page of a given index.
   struct RKey {
      DescriptorId_t fColumnId = kInvalidDescriptorId;
      std::type_index fInMemoryType = std::type_index(typeid(void));

      bool operator==(const RKey &other) const
      {
         return this->fColumnId == other.fColumnId && this->fInMemoryType == other.fInMemoryType;
      }

      bool operator!=(const RKey &other) const { return !(*this == other); }
   };

private:
   /// Hash function to be used in the unordered map fLookupByKey
   struct RKeyHasher {
      /// Like boost::hash_combine
      std::size_t operator()(const RKey &k) const
      {
         auto seed = std::hash<DescriptorId_t>()(k.fColumnId);
         return seed ^ (std::hash<std::type_index>()(k.fInMemoryType) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
      }
   };

   /// Every page in the page pool is annotated with a search key and a reference counter.
   struct REntry {
      RPage fPage;
      RKey fKey;
      std::int64_t fRefCounter = 0;
   };

   std::vector<REntry> fEntries; ///< All cached pages in the page pool
   /// Used in ReleasePage() to find the page index in fPages
   std::unordered_map<void *, std::size_t> fLookupByBuffer;
   /// Used in GetPage() to find the right page in fEntries. Lookup for the key (pair of on-disk and in-memory type)
   /// takes place in O(1). The selected pages are identified by index into the fEntries vector. Access in this
   /// selected set is linear. We may consider replacing this in the future by a sorted list of pages (e.g., std::set).
   /// That comes at a code complexity cost though.
   std::unordered_map<RKey, std::vector<std::size_t>, RKeyHasher> fLookupByKey;
   std::mutex fLock; ///< The page pool is accessed concurrently due to parallel decompression

   /// Add a new page to the fLookupByBuffer and fLookupByKey data structures.
   REntry &AddPage(RPage page, const RKey &key, std::int64_t initialRefCounter);

   /// Give back a page to the pool and decrease the reference counter. There must not be any pointers anymore into
   /// this page. If the reference counter drops to zero, the page pool might decide to call the deleter given in
   /// during registration. Called by the RPageRef destructor.
   void ReleasePage(const RPage &page);

public:
   RPagePool() = default;
   RPagePool(const RPagePool&) = delete;
   RPagePool& operator =(const RPagePool&) = delete;
   ~RPagePool() = default;

   /// Adds a new page to the pool. Upon registration, the page pool takes ownership of the page's memory.
   /// The new page has its reference counter set to 1.
   RPageRef RegisterPage(RPage page, RKey key);
   /// Like RegisterPage() but the reference counter is initialized to 0
   void PreloadPage(RPage page, RKey key);
   /// Tries to find the page corresponding to column and index in the cache. If the page is found, its reference
   /// counter is increased
   RPageRef GetPage(RKey key, NTupleSize_t globalIndex);
   RPageRef GetPage(RKey key, RClusterIndex clusterIndex);
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
         fPagePool->ReleasePage(fPage);
   }

   /// Used by the friend virtual page source to map the cluster ID to its virtual counterpart
   void ChangeClusterId(DescriptorId_t clusterId)
   {
      fPage.fClusterInfo = RPage::RClusterInfo(clusterId, fPage.fClusterInfo.GetIndexOffset());
   }

   const RPage &Get() const { return fPage; }
};

} // namespace Internal

} // namespace Experimental
} // namespace ROOT

#endif
