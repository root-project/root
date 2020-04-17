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
\class ROOT::Experimental::RCluster
\ingroup NTuple
\brief A map of on disk pages for a particular cluster

Derived classes implement how the on disk pages are stored in memory, e.g. mmap'd or in a special area.
*/
// clang-format on
class RCluster {
protected:
   /// The memory region containing the on-disk pages. Ownership of the memory region is passed to the cluster.
   /// Therefore, the region needs to be allocated in a way that fits the derived class and its destructor.
   void *fMemory;
   DescriptorId_t fClusterId;
   std::unordered_map<ROnDiskPage::Key, ROnDiskPage> fOnDiskPages;

public:
   RCluster(void *memory, DescriptorId_t clusterId) : fMemory(memory), fClusterId(clusterId) {}
   RCluster(const RCluster &other) = delete;
   RCluster &operator =(const RCluster &other) = delete;
   virtual ~RCluster();

   void Insert(const ROnDiskPage::Key &key, const ROnDiskPage &onDiskPage) { fOnDiskPages[key] = onDiskPage; }

   DescriptorId_t GetId() const { return fClusterId; }
   const ROnDiskPage *GetOnDiskPage(const ROnDiskPage::Key &key) const;

   size_t GetNOnDiskPages() const { return fOnDiskPages.size(); }
};

// clang-format off
/**
\class ROOT::Experimental::RHeapCluster
\ingroup NTuple
\brief An RCluster that provides the on-disk pages using new[]
*/
// clang-format on
class RHeapCluster : public RCluster {
public:
   RHeapCluster(void *memory, DescriptorId_t clusterId) : RCluster(memory, clusterId) {}
   ~RHeapCluster();
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
