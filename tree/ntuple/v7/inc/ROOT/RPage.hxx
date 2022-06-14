/// \file ROOT/RPage.hxx
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

#ifndef ROOT7_RPage
#define ROOT7_RPage

#include <ROOT/RNTupleUtil.hxx>

#include <cstddef>
#include <cstdint>

namespace ROOT {
namespace Experimental {

namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RPage
\ingroup NTuple
\brief A page is a slice of a column that is mapped into memory

The page provides an opaque memory buffer for uncompressed, unpacked data. It does not interpret
the contents but it does now about the size (and thus the number) of the elements inside as well as the element
number range within the backing column/cluster. The memory buffer is not managed by the page. It is normally registered
with the page pool and allocated/freed by the page storage.
*/
// clang-format on
class RPage {
public:
   /**
    * Stores information about the cluster in which this page resides.
    */
   class RClusterInfo {
   private:
      /// The cluster number
      DescriptorId_t fId = 0;
      /// The first element index of the column in this cluster
      NTupleSize_t fIndexOffset = 0;
   public:
      RClusterInfo() = default;
      RClusterInfo(NTupleSize_t id, NTupleSize_t indexOffset) : fId(id), fIndexOffset(indexOffset) {}
      NTupleSize_t GetId() const { return fId; }
      NTupleSize_t GetIndexOffset() const { return fIndexOffset; }
   };

private:
   ColumnId_t fColumnId;
   void *fBuffer;
   ClusterSize_t::ValueType fElementSize;
   ClusterSize_t::ValueType fNElements;
   /// The capacity of the page in number of elements
   ClusterSize_t::ValueType fMaxElements;
   NTupleSize_t fRangeFirst;
   RClusterInfo fClusterInfo;

public:
   RPage()
      : fColumnId(kInvalidColumnId), fBuffer(nullptr), fElementSize(0), fNElements(0), fMaxElements(0), fRangeFirst(0)
   {}
   RPage(ColumnId_t columnId, void* buffer, ClusterSize_t::ValueType elementSize, ClusterSize_t::ValueType maxElements)
      : fColumnId(columnId), fBuffer(buffer), fElementSize(elementSize), fNElements(0), fMaxElements(maxElements),
        fRangeFirst(0)
   {}
   ~RPage() = default;

   ColumnId_t GetColumnId() const { return fColumnId; }
   /// The space taken by column elements in the buffer
   ClusterSize_t::ValueType GetNBytes() const { return fElementSize * fNElements; }
   ClusterSize_t::ValueType GetElementSize() const { return fElementSize; }
   ClusterSize_t::ValueType GetNElements() const { return fNElements; }
   ClusterSize_t::ValueType GetMaxElements() const { return fMaxElements; }
   NTupleSize_t GetGlobalRangeFirst() const { return fRangeFirst; }
   NTupleSize_t GetGlobalRangeLast() const { return fRangeFirst + NTupleSize_t(fNElements) - 1; }
   ClusterSize_t::ValueType GetClusterRangeFirst() const { return fRangeFirst - fClusterInfo.GetIndexOffset(); }
   ClusterSize_t::ValueType GetClusterRangeLast() const {
      return GetClusterRangeFirst() + NTupleSize_t(fNElements) - 1;
   }
   const RClusterInfo& GetClusterInfo() const { return fClusterInfo; }

   bool Contains(NTupleSize_t globalIndex) const {
      return (globalIndex >= fRangeFirst) && (globalIndex < fRangeFirst + NTupleSize_t(fNElements));
   }

   bool Contains(const RClusterIndex &clusterIndex) const {
      if (fClusterInfo.GetId() != clusterIndex.GetClusterId())
         return false;
      auto clusterRangeFirst = ClusterSize_t(fRangeFirst - fClusterInfo.GetIndexOffset());
      return (clusterIndex.GetIndex() >= clusterRangeFirst) &&
             (clusterIndex.GetIndex() < clusterRangeFirst + fNElements);
    }

   void* GetBuffer() const { return fBuffer; }
   /// Called during writing: returns a pointer after the last element and increases the element counter
   /// in anticipation of the caller filling nElements in the page. It is the responsibility of the caller
   /// to prevent page overflows, i.e. that fNElements + nElements <= fMaxElements
   void* GrowUnchecked(ClusterSize_t::ValueType nElements) {
      auto offset = GetNBytes();
      fNElements += nElements;
      return static_cast<unsigned char *>(fBuffer) + offset;
   }
   /// Seek the page to a certain position of the column
   void SetWindow(const NTupleSize_t rangeFirst, const RClusterInfo &clusterInfo) {
      fClusterInfo = clusterInfo;
      fRangeFirst = rangeFirst;
   }
   /// Forget all currently stored elements (size == 0) and set a new starting index.
   void Reset(NTupleSize_t rangeFirst) { fNElements = 0; fRangeFirst = rangeFirst; }
   void ResetCluster(const RClusterInfo &clusterInfo) { fNElements = 0; fClusterInfo = clusterInfo; }

   /// Used by virtual page sources to map the physical column and cluster IDs to ther virtual counterparts
   void ChangeIds(DescriptorId_t columnId, DescriptorId_t clusterId)
   {
      fColumnId = columnId;
      fClusterInfo = RClusterInfo(clusterId, fClusterInfo.GetIndexOffset());
   }

   bool IsNull() const { return fBuffer == nullptr; }
   bool IsEmpty() const { return fNElements == 0; }
   bool operator ==(const RPage &other) const { return fBuffer == other.fBuffer; }
   bool operator !=(const RPage &other) const { return !(*this == other); }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
