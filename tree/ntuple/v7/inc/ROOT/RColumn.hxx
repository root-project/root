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
#include <ROOT/RColumnElement.hxx>
#include <ROOT/RColumnModel.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageStorage.hxx>

#include <TError.h>

#include <cstring> // for memcpy
#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {
namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::RColumn
\ingroup NTuple
\brief A column is a storage-backed array of a simple, fixed-size type, from which pages can be mapped into memory.
*/
// clang-format on
class RColumn {
private:
   RColumnModel fModel;
   /**
    * Columns belonging to the same field are distinguished by their order.  E.g. for an std::string field, there is
    * the offset column with index 0 and the character value column with index 1.
    */
   std::uint32_t fIndex;
   RPageSink *fPageSink = nullptr;
   RPageSource *fPageSource = nullptr;
   RPageStorage::ColumnHandle_t fHandleSink;
   RPageStorage::ColumnHandle_t fHandleSource;
   /// A set of open pages into which new elements are being written. The pages are used
   /// in rotation. They are 50% bigger than the target size given by the write options.
   /// The current page is filled until the target size, but it is only committed once the other
   /// write page is filled at least 50%. If a flush occurs earlier, a slightly oversized, single
   /// page will be committed.
   RPage fWritePage[2];
   /// Index of the current write page
   int fWritePageIdx = 0;
   /// For writing, the targeted number of elements, given by `fApproxNElementsPerPage` (in the write options) and the element size.
   /// We ensure this value to be >= 2 in Connect() so that we have meaningful
   /// "page full" and "page half full" events when writing the page.
   std::uint32_t fApproxNElementsPerPage = 0;
   /// The number of elements written resp. available in the column
   NTupleSize_t fNElements = 0;
   /// The currently mapped page for reading
   RPage fReadPage;
   /// The column id is used to find matching pages with content when reading
   ColumnId_t fColumnIdSource = kInvalidColumnId;
   /// Global index of the first element in this column; usually == 0, unless it is a deferred column
   NTupleSize_t fFirstElementIndex = 0;
   /// Used to pack and unpack pages on writing/reading
   std::unique_ptr<RColumnElementBase> fElement;

   RColumn(const RColumnModel &model, std::uint32_t index);

   /// Used in Append() and AppendV() to switch pages when the main page reached the target size
   /// The other page has been flushed when the main page reached 50%.
   void SwapWritePagesIfFull() {
      if (R__likely(fWritePage[fWritePageIdx].GetNElements() < fApproxNElementsPerPage))
         return;

      fWritePageIdx = 1 - fWritePageIdx; // == (fWritePageIdx + 1) % 2
      R__ASSERT(fWritePage[fWritePageIdx].IsEmpty());
      fWritePage[fWritePageIdx].Reset(fNElements);
   }

   /// When the main write page surpasses the 50% fill level, the (full) shadow write page gets flushed
   void FlushShadowWritePage() {
      auto otherIdx = 1 - fWritePageIdx;
      if (fWritePage[otherIdx].IsEmpty())
         return;
      fPageSink->CommitPage(fHandleSink, fWritePage[otherIdx]);
      // Mark the page as flushed; the rangeFirst is zero for now but will be reset to
      // fNElements in SwapWritePagesIfFull() when the pages swap
      fWritePage[otherIdx].Reset(0);
   }

public:
   template <typename CppT>
   static std::unique_ptr<RColumn> Create(const RColumnModel &model, std::uint32_t index)
   {
      auto column = std::unique_ptr<RColumn>(new RColumn(model, index));
      column->fElement = RColumnElementBase::Generate<CppT>(model.GetType());
      return column;
   }

   RColumn(const RColumn&) = delete;
   RColumn &operator =(const RColumn&) = delete;
   ~RColumn();

   /// Connect the column to a page storage.  If `pageStorage` is a page sink, `firstElementIndex` can be used to
   /// specify the first column element index with backing storage for this column.  On read back, elements before
   /// `firstElementIndex` will cause the zero page to be mapped.
   /// TODO(jalopezg): at this point it would be nicer to distinguish between connecting a page sink and a page source
   void Connect(DescriptorId_t fieldId, RPageStorage *pageStorage, NTupleSize_t firstElementIndex = 0U);

   void Append(const void *from)
   {
      void *dst = fWritePage[fWritePageIdx].GrowUnchecked(1);

      if (fWritePage[fWritePageIdx].GetNElements() == fApproxNElementsPerPage / 2) {
         FlushShadowWritePage();
      }

      std::memcpy(dst, from, fElement->GetSize());
      fNElements++;

      SwapWritePagesIfFull();
   }

   void AppendV(const void *from, std::size_t count)
   {
      // We might not have enough space in the current page. In this case, fall back to one by one filling.
      if (fWritePage[fWritePageIdx].GetNElements() + count > fApproxNElementsPerPage) {
         // TODO(jblomer): use (fewer) calls to AppendV to write the data page-by-page
         for (unsigned i = 0; i < count; ++i) {
            Append(static_cast<const unsigned char *>(from) + fElement->GetSize() * i);
         }
         return;
      }

      // The check for flushing the shadow page is more complicated than for the Append() case
      // because we don't necessarily fill up to exactly fApproxNElementsPerPage / 2 elements;
      // we might instead jump over the 50% fill level.
      // This check should be done before calling `RPage::GrowUnchecked()` as the latter affects the return value of
      // `RPage::GetNElements()`.
      if ((fWritePage[fWritePageIdx].GetNElements() < fApproxNElementsPerPage / 2) &&
          (fWritePage[fWritePageIdx].GetNElements() + count >= fApproxNElementsPerPage / 2))
      {
         FlushShadowWritePage();
      }

      void *dst = fWritePage[fWritePageIdx].GrowUnchecked(count);

      std::memcpy(dst, from, fElement->GetSize() * count);
      fNElements += count;

      // Note that by the very first check in AppendV, we cannot have filled more than fApproxNElementsPerPage elements
      SwapWritePagesIfFull();
   }

   void Read(const NTupleSize_t globalIndex, void *to)
   {
      if (!fReadPage.Contains(globalIndex)) {
         MapPage(globalIndex);
      }
      const auto elemSize = fElement->GetSize();
      void *from = static_cast<unsigned char *>(fReadPage.GetBuffer()) +
                   (globalIndex - fReadPage.GetGlobalRangeFirst()) * elemSize;
      std::memcpy(to, from, elemSize);
   }

   void Read(const RClusterIndex &clusterIndex, void *to)
   {
      if (!fReadPage.Contains(clusterIndex)) {
         MapPage(clusterIndex);
      }
      const auto elemSize = fElement->GetSize();
      void *from = static_cast<unsigned char *>(fReadPage.GetBuffer()) +
                   (clusterIndex.GetIndex() - fReadPage.GetClusterRangeFirst()) * elemSize;
      std::memcpy(to, from, elemSize);
   }

   void ReadV(const NTupleSize_t globalIndex, const ClusterSize_t::ValueType count, void *to)
   {
      if (!fReadPage.Contains(globalIndex)) {
         MapPage(globalIndex);
      }
      NTupleSize_t idxInPage = globalIndex - fReadPage.GetGlobalRangeFirst();

      const auto elemSize = fElement->GetSize();
      const void *from = static_cast<unsigned char *>(fReadPage.GetBuffer()) + idxInPage * elemSize;
      if (globalIndex + count <= fReadPage.GetGlobalRangeLast() + 1) {
         std::memcpy(to, from, elemSize * count);
      } else {
         ClusterSize_t::ValueType nBatch = fReadPage.GetNElements() - idxInPage;
         std::memcpy(to, from, elemSize * nBatch);
         auto tail = static_cast<unsigned char *>(to) + nBatch * elemSize;
         ReadV(globalIndex + nBatch, count - nBatch, tail);
      }
   }

   void ReadV(const RClusterIndex &clusterIndex, const ClusterSize_t::ValueType count, void *to)
   {
      if (!fReadPage.Contains(clusterIndex)) {
         MapPage(clusterIndex);
      }
      NTupleSize_t idxInPage = clusterIndex.GetIndex() - fReadPage.GetClusterRangeFirst();

      const auto elemSize = fElement->GetSize();
      const void *from = static_cast<unsigned char *>(fReadPage.GetBuffer()) + idxInPage * elemSize;
      if (clusterIndex.GetIndex() + count <= fReadPage.GetClusterRangeLast() + 1) {
         std::memcpy(to, from, elemSize * count);
      } else {
         ClusterSize_t::ValueType nBatch = fReadPage.GetNElements() - idxInPage;
         std::memcpy(to, from, elemSize * nBatch);
         auto tail = static_cast<unsigned char *>(to) + nBatch * elemSize;
         ReadV(RClusterIndex(clusterIndex.GetClusterId(), clusterIndex.GetIndex() + nBatch), count - nBatch, tail);
      }
   }

   template <typename CppT>
   CppT *Map(const NTupleSize_t globalIndex) {
      NTupleSize_t nItems;
      return MapV<CppT>(globalIndex, nItems);
   }

   template <typename CppT>
   CppT *Map(const RClusterIndex &clusterIndex) {
      NTupleSize_t nItems;
      return MapV<CppT>(clusterIndex, nItems);
   }

   template <typename CppT>
   CppT *MapV(const NTupleSize_t globalIndex, NTupleSize_t &nItems) {
      if (R__unlikely(!fReadPage.Contains(globalIndex))) {
         MapPage(globalIndex);
      }
      // +1 to go from 0-based indexing to 1-based number of items
      nItems = fReadPage.GetGlobalRangeLast() - globalIndex + 1;
      return reinterpret_cast<CppT*>(
         static_cast<unsigned char *>(fReadPage.GetBuffer()) +
         (globalIndex - fReadPage.GetGlobalRangeFirst()) * RColumnElement<CppT>::kSize);
   }

   template <typename CppT>
   CppT *MapV(const RClusterIndex &clusterIndex, NTupleSize_t &nItems) {
      if (!fReadPage.Contains(clusterIndex)) {
         MapPage(clusterIndex);
      }
      // +1 to go from 0-based indexing to 1-based number of items
      nItems = fReadPage.GetClusterRangeLast() - clusterIndex.GetIndex() + 1;
      return reinterpret_cast<CppT*>(
         static_cast<unsigned char *>(fReadPage.GetBuffer()) +
         (clusterIndex.GetIndex() - fReadPage.GetClusterRangeFirst()) * RColumnElement<CppT>::kSize);
   }

   NTupleSize_t GetGlobalIndex(const RClusterIndex &clusterIndex) {
      if (!fReadPage.Contains(clusterIndex)) {
         MapPage(clusterIndex);
      }
      return fReadPage.GetClusterInfo().GetIndexOffset() + clusterIndex.GetIndex();
   }

   RClusterIndex GetClusterIndex(NTupleSize_t globalIndex) {
      if (!fReadPage.Contains(globalIndex)) {
         MapPage(globalIndex);
      }
      return RClusterIndex(fReadPage.GetClusterInfo().GetId(),
                           globalIndex - fReadPage.GetClusterInfo().GetIndexOffset());
   }

   /// For offset columns only, look at the two adjacent values that define a collection's coordinates
   void GetCollectionInfo(const NTupleSize_t globalIndex, RClusterIndex *collectionStart, ClusterSize_t *collectionSize)
   {
      NTupleSize_t idxStart = 0;
      NTupleSize_t idxEnd;
      // Try to avoid jumping back to the previous page and jumping back to the previous cluster
      if (R__likely(globalIndex > 0)) {
         if (R__likely(fReadPage.Contains(globalIndex - 1))) {
            idxStart = *Map<ClusterSize_t>(globalIndex - 1);
            idxEnd = *Map<ClusterSize_t>(globalIndex);
            if (R__unlikely(fReadPage.GetClusterInfo().GetIndexOffset() == globalIndex))
               idxStart = 0;
         } else {
            idxEnd = *Map<ClusterSize_t>(globalIndex);
            auto selfOffset = fReadPage.GetClusterInfo().GetIndexOffset();
            idxStart = (globalIndex == selfOffset) ? 0 : *Map<ClusterSize_t>(globalIndex - 1);
         }
      } else {
         idxEnd = *Map<ClusterSize_t>(globalIndex);
      }
      *collectionSize = idxEnd - idxStart;
      *collectionStart = RClusterIndex(fReadPage.GetClusterInfo().GetId(), idxStart);
   }

   void GetCollectionInfo(const RClusterIndex &clusterIndex,
                          RClusterIndex *collectionStart, ClusterSize_t *collectionSize)
   {
      auto index = clusterIndex.GetIndex();
      auto idxStart = (index == 0) ? 0 : *Map<ClusterSize_t>(clusterIndex - 1);
      auto idxEnd = *Map<ClusterSize_t>(clusterIndex);
      *collectionSize = idxEnd - idxStart;
      *collectionStart = RClusterIndex(clusterIndex.GetClusterId(), idxStart);
   }

   /// Get the currently active cluster id
   void GetSwitchInfo(NTupleSize_t globalIndex, RClusterIndex *varIndex, std::uint32_t *tag) {
      auto varSwitch = Map<RColumnSwitch>(globalIndex);
      *varIndex = RClusterIndex(fReadPage.GetClusterInfo().GetId(), varSwitch->GetIndex());
      *tag = varSwitch->GetTag();
   }

   void Flush();
   void MapPage(const NTupleSize_t index);
   void MapPage(const RClusterIndex &clusterIndex);
   NTupleSize_t GetNElements() const { return fNElements; }
   RColumnElementBase *GetElement() const { return fElement.get(); }
   const RColumnModel &GetModel() const { return fModel; }
   std::uint32_t GetIndex() const { return fIndex; }
   ColumnId_t GetColumnIdSource() const { return fColumnIdSource; }
   NTupleSize_t GetFirstElementIndex() const { return fFirstElementIndex; }
   RPageSource *GetPageSource() const { return fPageSource; }
   RPageSink *GetPageSink() const { return fPageSink; }
   RPageStorage::ColumnHandle_t GetHandleSource() const { return fHandleSource; }
   RPageStorage::ColumnHandle_t GetHandleSink() const { return fHandleSink; }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
