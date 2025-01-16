/// \file ROOT/RColumn.hxx
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

#ifndef ROOT7_RColumn
#define ROOT7_RColumn

#include <ROOT/RConfig.hxx> // for R__likely
#include <ROOT/RColumnElementBase.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageStorage.hxx>

#include <TError.h>

#include <cstring> // for memcpy
#include <memory>
#include <utility>

namespace ROOT::Experimental::Internal {

// clang-format off
/**
\class ROOT::Experimental::Internal::RColumn
\ingroup NTuple
\brief A column is a storage-backed array of a simple, fixed-size type, from which pages can be mapped into memory.
*/
// clang-format on
class RColumn {
private:
   ENTupleColumnType fType;
   /// Columns belonging to the same field are distinguished by their order.  E.g. for an std::string field, there is
   /// the offset column with index 0 and the character value column with index 1.
   std::uint32_t fIndex;
   /// Fields can have multiple column representations, distinguished by representation index
   std::uint16_t fRepresentationIndex;
   RPageSink *fPageSink = nullptr;
   RPageSource *fPageSource = nullptr;
   RPageStorage::ColumnHandle_t fHandleSink;
   RPageStorage::ColumnHandle_t fHandleSource;
   /// The page into which new elements are being written. The page will initially be small
   /// (RNTupleWriteOptions::fInitialUnzippedPageSize, which corresponds to fInitialElements) and expand as needed and
   /// as memory for page buffers is still available (RNTupleWriteOptions::fPageBufferBudget) or the maximum page
   /// size is reached (RNTupleWriteOptions::fMaxUnzippedPageSize).
   RPage fWritePage;
   /// The initial number of elements in a page
   NTupleSize_t fInitialNElements = 1;
   /// The number of elements written resp. available in the column
   NTupleSize_t fNElements = 0;
   /// The currently mapped page for reading
   RPageRef fReadPageRef;
   /// The column id in the column descriptor, once connected to a sink or source
   DescriptorId_t fOnDiskId = kInvalidDescriptorId;
   /// Global index of the first element in this column; usually == 0, unless it is a deferred column
   NTupleSize_t fFirstElementIndex = 0;
   /// Used to pack and unpack pages on writing/reading
   std::unique_ptr<RColumnElementBase> fElement;
   /// The column team is a set of columns that serve the same column index for different representation IDs.
   /// Initially, the team has only one member, the very column it belongs to. Through MergeTeams(), two columns
   /// can join forces. The team is used to react on suppressed columns: if the current team member has a suppressed
   /// column for a MapPage() call, it get the page from the active column in the corresponding cluster.
   std::vector<RColumn *> fTeam;
   /// Points into fTeam to the column that successfully returned the last page.
   std::size_t fLastGoodTeamIdx = 0;

   RColumn(ENTupleColumnType type, std::uint32_t columnIndex, std::uint16_t representationIndex);

   /// Used when trying to append to a full write page. If possible, expand the page. Otherwise, flush and reset
   /// to the minimal size.
   void HandleWritePageIfFull()
   {
      auto newMaxElements = fWritePage.GetMaxElements() * 2;
      if (newMaxElements * fElement->GetSize() > fPageSink->GetWriteOptions().GetMaxUnzippedPageSize()) {
         newMaxElements = fPageSink->GetWriteOptions().GetMaxUnzippedPageSize() / fElement->GetSize();
      }

      if (newMaxElements == fWritePage.GetMaxElements()) {
         // Maximum page size reached, flush and reset
         Flush();
      } else {
         auto expandedPage = fPageSink->ReservePage(fHandleSink, newMaxElements);
         if (expandedPage.IsNull()) {
            Flush();
         } else {
            memcpy(expandedPage.GetBuffer(), fWritePage.GetBuffer(), fWritePage.GetNBytes());
            expandedPage.Reset(fNElements);
            expandedPage.GrowUnchecked(fWritePage.GetNElements());
            fWritePage = std::move(expandedPage);
         }
      }

      assert(fWritePage.GetNElements() < fWritePage.GetMaxElements());
   }

public:
   template <typename CppT>
   static std::unique_ptr<RColumn>
   Create(ENTupleColumnType type, std::uint32_t columnIdx, std::uint16_t representationIdx)
   {
      auto column = std::unique_ptr<RColumn>(new RColumn(type, columnIdx, representationIdx));
      column->fElement = RColumnElementBase::Generate<CppT>(type);
      return column;
   }

   RColumn(const RColumn &) = delete;
   RColumn &operator=(const RColumn &) = delete;
   ~RColumn();

   /// Connect the column to a page sink.  `firstElementIndex` can be used to specify the first column element index
   /// with backing storage for this column.  On read back, elements before `firstElementIndex` will cause the zero page
   /// to be mapped.
   void ConnectPageSink(DescriptorId_t fieldId, RPageSink &pageSink, NTupleSize_t firstElementIndex = 0U);
   /// Connect the column to a page source.
   void ConnectPageSource(DescriptorId_t fieldId, RPageSource &pageSource);

   void Append(const void *from)
   {
      if (fWritePage.GetNElements() == fWritePage.GetMaxElements()) {
         HandleWritePageIfFull();
      }

      void *dst = fWritePage.GrowUnchecked(1);

      std::memcpy(dst, from, fElement->GetSize());
      fNElements++;
   }

   void AppendV(const void *from, std::size_t count)
   {
      auto src = reinterpret_cast<const unsigned char *>(from);
      // TODO(jblomer): A future optimization should grow the page in one go, up to the maximum unzipped page size
      while (count > 0) {
         std::size_t nElementsRemaining = fWritePage.GetMaxElements() - fWritePage.GetNElements();
         if (nElementsRemaining == 0) {
            HandleWritePageIfFull();
            nElementsRemaining = fWritePage.GetMaxElements() - fWritePage.GetNElements();
         }

         assert(nElementsRemaining > 0);
         auto nBatch = std::min(count, nElementsRemaining);

         void *dst = fWritePage.GrowUnchecked(nBatch);
         std::memcpy(dst, src, nBatch * fElement->GetSize());
         src += nBatch * fElement->GetSize();
         count -= nBatch;
         fNElements += nBatch;
      }
   }

   void Read(const NTupleSize_t globalIndex, void *to)
   {
      if (!fReadPageRef.Get().Contains(globalIndex)) {
         MapPage(globalIndex);
      }
      const auto elemSize = fElement->GetSize();
      void *from = static_cast<unsigned char *>(fReadPageRef.Get().GetBuffer()) +
                   (globalIndex - fReadPageRef.Get().GetGlobalRangeFirst()) * elemSize;
      std::memcpy(to, from, elemSize);
   }

   void Read(RNTupleLocalIndex localIndex, void *to)
   {
      if (!fReadPageRef.Get().Contains(localIndex)) {
         MapPage(localIndex);
      }
      const auto elemSize = fElement->GetSize();
      void *from = static_cast<unsigned char *>(fReadPageRef.Get().GetBuffer()) +
                   (localIndex.GetIndexInCluster() - fReadPageRef.Get().GetClusterRangeFirst()) * elemSize;
      std::memcpy(to, from, elemSize);
   }

   void ReadV(NTupleSize_t globalIndex, NTupleSize_t count, void *to)
   {
      const auto elemSize = fElement->GetSize();
      auto tail = static_cast<unsigned char *>(to);

      while (count > 0) {
         if (!fReadPageRef.Get().Contains(globalIndex)) {
            MapPage(globalIndex);
         }
         const NTupleSize_t idxInPage = globalIndex - fReadPageRef.Get().GetGlobalRangeFirst();

         const void *from = static_cast<unsigned char *>(fReadPageRef.Get().GetBuffer()) + idxInPage * elemSize;
         const NTupleSize_t nBatch = std::min(fReadPageRef.Get().GetNElements() - idxInPage, count);

         std::memcpy(tail, from, elemSize * nBatch);

         tail += nBatch * elemSize;
         count -= nBatch;
         globalIndex += nBatch;
      }
   }

   void ReadV(RNTupleLocalIndex localIndex, NTupleSize_t count, void *to)
   {
      const auto elemSize = fElement->GetSize();
      auto tail = static_cast<unsigned char *>(to);

      while (count > 0) {
         if (!fReadPageRef.Get().Contains(localIndex)) {
            MapPage(localIndex);
         }
         NTupleSize_t idxInPage = localIndex.GetIndexInCluster() - fReadPageRef.Get().GetClusterRangeFirst();

         const void *from = static_cast<unsigned char *>(fReadPageRef.Get().GetBuffer()) + idxInPage * elemSize;
         const NTupleSize_t nBatch = std::min(count, fReadPageRef.Get().GetNElements() - idxInPage);

         std::memcpy(tail, from, elemSize * nBatch);

         tail += nBatch * elemSize;
         count -= nBatch;
         localIndex = RNTupleLocalIndex(localIndex.GetClusterId(), localIndex.GetIndexInCluster() + nBatch);
      }
   }

   template <typename CppT>
   CppT *Map(const NTupleSize_t globalIndex)
   {
      NTupleSize_t nItems;
      return MapV<CppT>(globalIndex, nItems);
   }

   template <typename CppT>
   CppT *Map(RNTupleLocalIndex localIndex)
   {
      NTupleSize_t nItems;
      return MapV<CppT>(localIndex, nItems);
   }

   template <typename CppT>
   CppT *MapV(const NTupleSize_t globalIndex, NTupleSize_t &nItems)
   {
      if (R__unlikely(!fReadPageRef.Get().Contains(globalIndex))) {
         MapPage(globalIndex);
      }
      // +1 to go from 0-based indexing to 1-based number of items
      nItems = fReadPageRef.Get().GetGlobalRangeLast() - globalIndex + 1;
      return reinterpret_cast<CppT *>(static_cast<unsigned char *>(fReadPageRef.Get().GetBuffer()) +
                                      (globalIndex - fReadPageRef.Get().GetGlobalRangeFirst()) * sizeof(CppT));
   }

   template <typename CppT>
   CppT *MapV(RNTupleLocalIndex localIndex, NTupleSize_t &nItems)
   {
      if (!fReadPageRef.Get().Contains(localIndex)) {
         MapPage(localIndex);
      }
      // +1 to go from 0-based indexing to 1-based number of items
      nItems = fReadPageRef.Get().GetClusterRangeLast() - localIndex.GetIndexInCluster() + 1;
      return reinterpret_cast<CppT *>(static_cast<unsigned char *>(fReadPageRef.Get().GetBuffer()) +
                                      (localIndex.GetIndexInCluster() - fReadPageRef.Get().GetClusterRangeFirst()) *
                                         sizeof(CppT));
   }

   NTupleSize_t GetGlobalIndex(RNTupleLocalIndex clusterIndex)
   {
      if (!fReadPageRef.Get().Contains(clusterIndex)) {
         MapPage(clusterIndex);
      }
      return fReadPageRef.Get().GetClusterInfo().GetIndexOffset() + clusterIndex.GetIndexInCluster();
   }

   RNTupleLocalIndex GetClusterIndex(NTupleSize_t globalIndex)
   {
      if (!fReadPageRef.Get().Contains(globalIndex)) {
         MapPage(globalIndex);
      }
      return RNTupleLocalIndex(fReadPageRef.Get().GetClusterInfo().GetId(),
                               globalIndex - fReadPageRef.Get().GetClusterInfo().GetIndexOffset());
   }

   /// For offset columns only, look at the two adjacent values that define a collection's coordinates
   void
   GetCollectionInfo(const NTupleSize_t globalIndex, RNTupleLocalIndex *collectionStart, NTupleSize_t *collectionSize)
   {
      NTupleSize_t idxStart = 0;
      NTupleSize_t idxEnd;
      // Try to avoid jumping back to the previous page and jumping back to the previous cluster
      if (R__likely(globalIndex > 0)) {
         if (R__likely(fReadPageRef.Get().Contains(globalIndex - 1))) {
            idxStart = *Map<RColumnIndex>(globalIndex - 1);
            idxEnd = *Map<RColumnIndex>(globalIndex);
            if (R__unlikely(fReadPageRef.Get().GetClusterInfo().GetIndexOffset() == globalIndex))
               idxStart = 0;
         } else {
            idxEnd = *Map<RColumnIndex>(globalIndex);
            auto selfOffset = fReadPageRef.Get().GetClusterInfo().GetIndexOffset();
            idxStart = (globalIndex == selfOffset) ? 0 : *Map<RColumnIndex>(globalIndex - 1);
         }
      } else {
         idxEnd = *Map<RColumnIndex>(globalIndex);
      }
      *collectionSize = idxEnd - idxStart;
      *collectionStart = RNTupleLocalIndex(fReadPageRef.Get().GetClusterInfo().GetId(), idxStart);
   }

   void
   GetCollectionInfo(RNTupleLocalIndex localIndex, RNTupleLocalIndex *collectionStart, NTupleSize_t *collectionSize)
   {
      auto index = localIndex.GetIndexInCluster();
      auto idxStart = (index == 0) ? 0 : *Map<RColumnIndex>(localIndex - 1);
      auto idxEnd = *Map<RColumnIndex>(localIndex);
      *collectionSize = idxEnd - idxStart;
      *collectionStart = RNTupleLocalIndex(localIndex.GetClusterId(), idxStart);
   }

   /// Get the currently active cluster id
   void GetSwitchInfo(NTupleSize_t globalIndex, RNTupleLocalIndex *varIndex, std::uint32_t *tag)
   {
      auto varSwitch = Map<RColumnSwitch>(globalIndex);
      *varIndex = RNTupleLocalIndex(fReadPageRef.Get().GetClusterInfo().GetId(), varSwitch->GetIndex());
      *tag = varSwitch->GetTag();
   }

   void Flush();
   void CommitSuppressed();

   void MapPage(NTupleSize_t globalIndex) { R__ASSERT(TryMapPage(globalIndex)); }
   void MapPage(RNTupleLocalIndex localIndex) { R__ASSERT(TryMapPage(localIndex)); }
   bool TryMapPage(NTupleSize_t globalIndex);
   bool TryMapPage(RNTupleLocalIndex localIndex);

   bool ReadPageContains(NTupleSize_t globalIndex) const { return fReadPageRef.Get().Contains(globalIndex); }
   bool ReadPageContains(RNTupleLocalIndex localIndex) const { return fReadPageRef.Get().Contains(localIndex); }

   void MergeTeams(RColumn &other);

   NTupleSize_t GetNElements() const { return fNElements; }
   RColumnElementBase *GetElement() const { return fElement.get(); }
   ENTupleColumnType GetType() const { return fType; }
   std::uint16_t GetBitsOnStorage() const
   {
      assert(fElement);
      return static_cast<std::uint16_t>(fElement->GetBitsOnStorage());
   }
   std::optional<std::pair<double, double>> GetValueRange() const
   {
      assert(fElement);
      return fElement->GetValueRange();
   }
   std::uint32_t GetIndex() const { return fIndex; }
   std::uint16_t GetRepresentationIndex() const { return fRepresentationIndex; }
   DescriptorId_t GetOnDiskId() const { return fOnDiskId; }
   NTupleSize_t GetFirstElementIndex() const { return fFirstElementIndex; }
   RPageSource *GetPageSource() const { return fPageSource; }
   RPageSink *GetPageSink() const { return fPageSink; }
   RPageStorage::ColumnHandle_t GetHandleSource() const { return fHandleSource; }
   RPageStorage::ColumnHandle_t GetHandleSink() const { return fHandleSink; }

   void SetBitsOnStorage(std::size_t bits) { fElement->SetBitsOnStorage(bits); }
   std::size_t GetWritePageCapacity() const { return fWritePage.GetCapacity(); }
   void SetValueRange(double min, double max) { fElement->SetValueRange(min, max); }
}; // class RColumn

} // namespace ROOT::Experimental::Internal

#endif
