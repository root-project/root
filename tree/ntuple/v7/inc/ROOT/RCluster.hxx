/// \file ROOT/RCluster.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2020-03-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleUtil.hxx>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef ROOT7_RCluster
#define ROOT7_RCluster

namespace ROOT {
namespace Experimental {
namespace Detail {


// clang-format off
/**
\class ROOT::Experimental::ROnDiskPage
\ingroup NTuple
\brief A page as being stored on disk, that is packed and compressed

Used by the cluster pool to cache pages from the physical storage. Such pages generally need to be
uncompressed and unpacked before they can be used by RNTuple upper layers.
*/
// clang-format on
class ROnDiskPage {
private:
   /// The memory location of the bytes
   const void *fAddress = nullptr;
   /// The compressed and packed size of the page
   std::size_t fSize = 0;

public:
   /// On disk pages within a page source are identified by the column and page number. The key is used for
   /// associative collections of on disk pages.
   struct Key {
      DescriptorId_t fColumnId;
      NTupleSize_t fPageNo;
      Key(DescriptorId_t columnId, NTupleSize_t pageNo) : fColumnId(columnId), fPageNo(pageNo) {}
      bool operator ==(const Key &other) const {
         return fColumnId == other.fColumnId && fPageNo == other.fPageNo;
      }
   };

   ROnDiskPage() = default;
   ROnDiskPage(void *address, std::size_t size) : fAddress(address), fSize(size) {}

   const void *GetAddress() const { return fAddress; }
   std::size_t GetSize() const { return fSize; }

   bool IsNull() const { return fAddress == nullptr; }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

// For hash maps ROnDiskPage::Key --> ROnDiskPage
namespace std
{
   template <>
   struct hash<ROOT::Experimental::Detail::ROnDiskPage::Key>
   {
      size_t operator()(const ROOT::Experimental::Detail::ROnDiskPage::Key &key) const
      {
         return ((std::hash<ROOT::Experimental::DescriptorId_t>()(key.fColumnId) ^
                 (hash<ROOT::Experimental::NTupleSize_t>()(key.fPageNo) << 1)) >> 1);
      }
   };
}


namespace ROOT {
namespace Experimental {
namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::ROnDiskPageMap
\ingroup NTuple
\brief A memory region that contains packed and compressed pages

Derived classes implement how the on disk pages are stored in memory, e.g. mmap'd or in a special area.
*/
// clang-format on
class ROnDiskPageMap {
   friend class RCluster;

protected:
   /// The memory region containing the on-disk pages. Ownership of the memory region is passed to the page map.
   /// Therefore, the region needs to be allocated in a way that fits the derived class and its destructor.
   void *fMemory;
   std::unordered_map<ROnDiskPage::Key, ROnDiskPage> fOnDiskPages;

public:
   explicit ROnDiskPageMap(void *memory) : fMemory(memory) {}
   ROnDiskPageMap(const ROnDiskPageMap &other) = delete;
   ROnDiskPageMap(ROnDiskPageMap &&other);
   ROnDiskPageMap &operator =(const ROnDiskPageMap &other) = delete;
   ROnDiskPageMap &operator =(ROnDiskPageMap &&other);
   virtual ~ROnDiskPageMap();

   /// Inserts information about a page stored in fMemory.  Therefore, the address referenced by onDiskPage
   /// needs to be owned by the fMemory block.  If a page map contains a page of a given column, it is expected
   /// that _all_ the pages of that column in that cluster are part of the page map.
   void Register(const ROnDiskPage::Key &key, const ROnDiskPage &onDiskPage) { fOnDiskPages.emplace(key, onDiskPage); }
};


// clang-format off
/**
\class ROOT::Experimental::Detail::ROnDiskPageMapHeap
\ingroup NTuple
\brief An ROnDiskPageMap that is used for an fMemory allocated as an array of unsigned char.
*/
// clang-format on
class ROnDiskPageMapHeap : public ROnDiskPageMap {
public:
   explicit ROnDiskPageMapHeap(void *memory) : ROnDiskPageMap(memory) {}
   ~ROnDiskPageMapHeap();
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RCluster
\ingroup NTuple
\brief An in-memory subset of the packed and compressed pages of a cluster

Binds together several page maps that represent all the pages of certain columns of a cluster
*/
// clang-format on
class RCluster {
protected:
   /// References the cluster identifier in the page source that created the cluster
   DescriptorId_t fClusterId;
   /// Multiple page maps can be combined in a single RCluster
   std::vector<ROnDiskPageMap> fPageMaps;
   /// List of the (complete) columns represented by the RCluster
   std::unordered_set<DescriptorId_t> fAvailColumns;
   /// Lookup table for the on-disk pages
   std::unordered_map<ROnDiskPage::Key, ROnDiskPage> fOnDiskPages;

public:
   explicit RCluster(DescriptorId_t clusterId) : fClusterId(clusterId) {}
   RCluster(const RCluster &other) = delete;
   RCluster(RCluster &&other) = default;
   RCluster &operator =(const RCluster &other) = delete;
   RCluster &operator =(RCluster &&other) = default;
   ~RCluster() = default;

   /// Move the given page map into this cluster; overlapping page ranges are ignored
   void MergeColumns(ROnDiskPageMap &&pageMap);
   /// Move the contents of other into this cluster; overlapping page ranges are ignored
   void MergeCluster(RCluster &&other);
   const ROnDiskPage *GetOnDiskPage(const ROnDiskPage::Key &key) const;

   DescriptorId_t GetId() const { return fClusterId; }
   const std::unordered_set<DescriptorId_t> &GetAvailColumns() const { return fAvailColumns; }
   bool ContainsColumn(DescriptorId_t columnId) const { return fAvailColumns.count(columnId) > 0; }
   size_t GetNOnDiskPages() const { return fOnDiskPages.size(); }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
