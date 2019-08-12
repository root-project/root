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
      NTupleSize_t fId;
      /// The first element index of the column in this cluster
      NTupleSize_t fSelfOffset;
      /// For offset columns, also store the cluster offset of the column being referenced
      NTupleSize_t fPointeeOffset;
   public:
      RClusterInfo() : fId(0), fSelfOffset(0), fPointeeOffset(0) {}
      RClusterInfo(NTupleSize_t id, NTupleSize_t selfOffset, NTupleSize_t pointeeOffset)
         : fId(id), fSelfOffset(selfOffset), fPointeeOffset(pointeeOffset) {}
      NTupleSize_t GetId() const { return fId; }
      NTupleSize_t GetSelfOffset() const { return fSelfOffset; }
      NTupleSize_t GetPointeeOffset() const { return fPointeeOffset; }
   };

private:
   ColumnId_t fColumnId;
   void* fBuffer;
   std::size_t fCapacity;
   std::size_t fSize;
   std::size_t fElementSize;
   NTupleSize_t fRangeFirst;
   NTupleSize_t fNElements;
   RClusterInfo fClusterInfo;

public:
   RPage()
     : fColumnId(kInvalidColumnId), fBuffer(nullptr), fCapacity(0), fSize(0), fElementSize(0),
       fRangeFirst(0), fNElements(0)
   {}
   RPage(ColumnId_t columnId, void* buffer, std::size_t capacity, std::size_t elementSize)
      : fColumnId(columnId), fBuffer(buffer), fCapacity(capacity), fSize(0), fElementSize(elementSize),
        fRangeFirst(0), fNElements(0)
      {}
   ~RPage() = default;

   std::int64_t GetColumnId() { return fColumnId; }
   /// The total space available in the page
   std::size_t GetCapacity() const { return fCapacity; }
   /// The space taken by column elements in the buffer
   std::size_t GetSize() const { return fSize; }
   std::size_t GetElementSize() const { return fElementSize; }
   NTupleSize_t GetNElements() const { return fSize / fElementSize; }
   NTupleSize_t GetRangeFirst() const { return fRangeFirst; }
   NTupleSize_t GetRangeLast() const { return fRangeFirst + fNElements - 1; }
   const RClusterInfo& GetClusterInfo() const { return fClusterInfo; }
   bool Contains(NTupleSize_t index) const {
      return (index >= fRangeFirst) && (index < fRangeFirst + fNElements);
   }
   void* GetBuffer() const { return fBuffer; }
   /// Return a pointer after the last element that has space for nElements new elements. If there is not enough capacity,
   /// return nullptr
   void* TryGrow(std::size_t nElements) {
      size_t offset = fSize;
      size_t nbyte = nElements * fElementSize;
      if (offset + nbyte > fCapacity) {
        return nullptr;
      }
      fSize += nbyte;
      fNElements = nElements;
      return static_cast<unsigned char *>(fBuffer) + offset;
   }
   /// Seek the page to a certain position of the column
   void SetWindow(const NTupleSize_t rangeFirst, const RClusterInfo &clusterInfo) {
      fClusterInfo = clusterInfo;
      fRangeFirst = rangeFirst;
   }
   /// Forget all currently stored elements (size == 0) and set a new starting index.
   void Reset(NTupleSize_t rangeFirst) { fSize = 0; fRangeFirst = rangeFirst; }
   void ResetCluster(const RClusterInfo &clusterInfo) { fSize = 0; fClusterInfo = clusterInfo; }

   bool IsNull() const { return fBuffer == nullptr; }
   bool operator ==(const RPage &other) const { return fBuffer == other.fBuffer; }
   bool operator !=(const RPage &other) const { return !(*this == other); }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
