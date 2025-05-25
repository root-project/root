/// \file ROOT/RPage.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RPage
#define ROOT_RPage

#include <ROOT/RNTupleUtil.hxx>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ROOT {
namespace Internal {

class RPageAllocator;
class RPageRef;

// clang-format off
/**
\class ROOT::Internal::RPage
\ingroup NTuple
\brief A page is a slice of a column that is mapped into memory

The page provides an opaque memory buffer for uncompressed, unpacked data. It does not interpret
the contents but it does now about the size (and thus the number) of the elements inside as well as the element
number range within the backing column/cluster.
For reading, pages are allocated and filled by the page source and then registered with the page pool.
For writing, the page sink allocates uninitialized pages of a given size.
The page has a pointer to its memory allocator so that it can release itself.
*/
// clang-format on
class RPage {
   friend class RPageRef;

public:
   static constexpr size_t kPageZeroSize = 64 * 1024;

   /**
    * Stores information about the cluster in which this page resides.
    */
   class RClusterInfo {
   private:
      /// The cluster number
      ROOT::DescriptorId_t fId = 0;
      /// The first element index of the column in this cluster
      ROOT::NTupleSize_t fIndexOffset = 0;

   public:
      RClusterInfo() = default;
      RClusterInfo(ROOT::NTupleSize_t id, ROOT::NTupleSize_t indexOffset) : fId(id), fIndexOffset(indexOffset) {}
      ROOT::NTupleSize_t GetId() const { return fId; }
      ROOT::NTupleSize_t GetIndexOffset() const { return fIndexOffset; }
   };

private:
   void *fBuffer = nullptr;
   /// The allocator used to allocate fBuffer. Can be null if the buffer doesn't need to be freed.
   RPageAllocator *fPageAllocator = nullptr;
   std::uint32_t fElementSize = 0;
   std::uint32_t fNElements = 0;
   /// The capacity of the page in number of elements
   std::uint32_t fMaxElements = 0;
   ROOT::NTupleSize_t fRangeFirst = 0;
   RClusterInfo fClusterInfo;

public:
   RPage() = default;
   RPage(void *buffer, RPageAllocator *pageAllocator, std::uint32_t elementSize, std::uint32_t maxElements)
      : fBuffer(buffer), fPageAllocator(pageAllocator), fElementSize(elementSize), fMaxElements(maxElements)
   {}
   RPage(const RPage &) = delete;
   RPage &operator=(const RPage &) = delete;
   RPage(RPage &&other)
   {
      fBuffer = other.fBuffer;
      fPageAllocator = other.fPageAllocator;
      fElementSize = other.fElementSize;
      fNElements = other.fNElements;
      fMaxElements = other.fMaxElements;
      fRangeFirst = other.fRangeFirst;
      fClusterInfo = other.fClusterInfo;
      other.fPageAllocator = nullptr;
   }
   RPage &operator=(RPage &&other)
   {
      if (this != &other) {
         std::swap(fBuffer, other.fBuffer);
         std::swap(fPageAllocator, other.fPageAllocator);
         std::swap(fElementSize, other.fElementSize);
         std::swap(fNElements, other.fNElements);
         std::swap(fMaxElements, other.fMaxElements);
         std::swap(fRangeFirst, other.fRangeFirst);
         std::swap(fClusterInfo, other.fClusterInfo);
      }
      return *this;
   }
   ~RPage();

   /// The space taken by column elements in the buffer
   std::size_t GetNBytes() const
   {
      return static_cast<std::size_t>(fElementSize) * static_cast<std::size_t>(fNElements);
   }
   std::size_t GetCapacity() const
   {
      return static_cast<std::size_t>(fElementSize) * static_cast<std::size_t>(fMaxElements);
   }
   std::uint32_t GetElementSize() const { return fElementSize; }
   std::uint32_t GetNElements() const { return fNElements; }
   std::uint32_t GetMaxElements() const { return fMaxElements; }
   ROOT::NTupleSize_t GetGlobalRangeFirst() const { return fRangeFirst; }
   ROOT::NTupleSize_t GetGlobalRangeLast() const { return fRangeFirst + ROOT::NTupleSize_t(fNElements) - 1; }
   ROOT::NTupleSize_t GetLocalRangeFirst() const { return fRangeFirst - fClusterInfo.GetIndexOffset(); }
   ROOT::NTupleSize_t GetLocalRangeLast() const { return GetLocalRangeFirst() + ROOT::NTupleSize_t(fNElements) - 1; }
   const RClusterInfo& GetClusterInfo() const { return fClusterInfo; }

   bool Contains(ROOT::NTupleSize_t globalIndex) const
   {
      return (globalIndex >= fRangeFirst) && (globalIndex < fRangeFirst + ROOT::NTupleSize_t(fNElements));
   }

   bool Contains(RNTupleLocalIndex localIndex) const
   {
      if (fClusterInfo.GetId() != localIndex.GetClusterId())
         return false;
      auto clusterRangeFirst = fRangeFirst - fClusterInfo.GetIndexOffset();
      return (localIndex.GetIndexInCluster() >= clusterRangeFirst) &&
             (localIndex.GetIndexInCluster() < clusterRangeFirst + fNElements);
   }

   void* GetBuffer() const { return fBuffer; }
   /// Increases the number elements in the page. The caller is responsible to respect the page capacity,
   /// i.e. to ensure that fNElements + nElements <= fMaxElements.
   /// Returns a pointer after the last element, which is used during writing in anticipation of the caller filling
   /// nElements in the page.
   /// When reading a page from disk, GrowUnchecked is used to set the actual number of elements. In this case, the
   /// return value is ignored.
   void *GrowUnchecked(std::uint32_t nElements)
   {
      assert(fNElements + nElements <= fMaxElements);
      auto offset = GetNBytes();
      fNElements += nElements;
      return static_cast<unsigned char *>(fBuffer) + offset;
   }
   /// Seek the page to a certain position of the column
   void SetWindow(const ROOT::NTupleSize_t rangeFirst, const RClusterInfo &clusterInfo)
   {
      fClusterInfo = clusterInfo;
      fRangeFirst = rangeFirst;
   }
   /// Forget all currently stored elements (size == 0) and set a new starting index.
   void Reset(ROOT::NTupleSize_t rangeFirst)
   {
      fNElements = 0;
      fRangeFirst = rangeFirst;
   }
   void ResetCluster(const RClusterInfo &clusterInfo) { fNElements = 0; fClusterInfo = clusterInfo; }

   /// Return a pointer to the page zero buffer used if there is no on-disk data for a particular deferred column
   static const void *GetPageZeroBuffer();

   bool IsNull() const { return fBuffer == nullptr; }
   bool IsEmpty() const { return fNElements == 0; }
   bool operator ==(const RPage &other) const { return fBuffer == other.fBuffer; }
   bool operator !=(const RPage &other) const { return !(*this == other); }
}; // class RPage

} // namespace Internal
} // namespace ROOT

#endif
