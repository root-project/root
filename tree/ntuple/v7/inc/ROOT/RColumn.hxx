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

#include <ROOT/RColumnElement.hxx>
#include <ROOT/RColumnModel.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageStorage.hxx>

#include <TError.h>

#include <memory>

namespace ROOT {
namespace Experimental {
namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::RColumn
\ingroup NTuple
\brief A column is a storage-backed array of a simple, fixed-size type, from which pages can be mapped into memory.

On the primitives data layer, the RColumn and RColumnElement are the equivalents to RField and RFieldValue on the
logical data layer.
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
   /// head page is filled at least 50%. If a flush occurs earlier, a slightly oversize, single
   /// page will be committed.
   RPage fHeadPage[2];
   /// Index of the current head page
   int fHeadPageIdx = 0;
   /// For writing, the targeted number of elements, given by `fApproxNElementsPerPage` (in the write options) and the element size.
   /// We ensure this value to be >= 2 in Connect() so that we have meaningful
   /// "page full" and "page half full" events when writing the page.
   std::uint32_t fApproxNElementsPerPage = 0;
   /// The number of elements written resp. available in the column
   NTupleSize_t fNElements = 0;
   /// The currently mapped page for reading
   RPage fCurrentPage;
   /// The column id is used to find matching pages with content when reading
   ColumnId_t fColumnIdSource = kInvalidColumnId;
   /// Used to pack and unpack pages on writing/reading
   std::unique_ptr<RColumnElementBase> fElement;

   RColumn(const RColumnModel &model, std::uint32_t index);

   /// Used in Append() and AppendV() to switch pages when the main page reached the target size
   /// The other page has been flushed when the main page reached 50%.
   void SwapHeadPages() {
      fHeadPageIdx = 1 - fHeadPageIdx; // == (fHeadPageIdx + 1) % 2
      R__ASSERT(fHeadPage[fHeadPageIdx].IsEmpty());
      fHeadPage[fHeadPageIdx].Reset(fNElements);
   }

   /// When the main head page surpasses the 50% fill level, the (full) shadow head page gets flushed
   void FlushShadowHeadPage() {
      auto otherIdx = 1 - fHeadPageIdx;
      if (fHeadPage[otherIdx].IsEmpty())
         return;
      fPageSink->CommitPage(fHandleSink, fHeadPage[otherIdx]);
      // Mark the page as flushed; the rangeFirst is zero for now but will be reset to
      // fNElements in SwapHeadPages() when the pages swap
      fHeadPage[otherIdx].Reset(0);
   }

public:
   template <typename CppT, EColumnType ColumnT>
   static RColumn *Create(const RColumnModel &model, std::uint32_t index) {
      R__ASSERT(model.GetType() == ColumnT);
      auto column = new RColumn(model, index);
      column->fElement = std::unique_ptr<RColumnElementBase>(new RColumnElement<CppT, ColumnT>(nullptr));
      return column;
   }

   RColumn(const RColumn&) = delete;
   RColumn &operator =(const RColumn&) = delete;
   ~RColumn();

   void Connect(DescriptorId_t fieldId, RPageStorage *pageStorage);

   void Append(const RColumnElementBase &element) {
      void *dst = fHeadPage[fHeadPageIdx].GrowUnchecked(1);

      if (fHeadPage[fHeadPageIdx].GetNElements() == fApproxNElementsPerPage / 2) {
         FlushShadowHeadPage();
      }

      element.WriteTo(dst, 1);
      fNElements++;

      if (fHeadPage[fHeadPageIdx].GetNElements() == fApproxNElementsPerPage)
         SwapHeadPages();
   }

   void AppendV(const RColumnElementBase &elemArray, std::size_t count) {
      // We might not have enough space in the current page. In this case, fall back to one by one filling.
      if (fHeadPage[fHeadPageIdx].GetNElements() + count > fApproxNElementsPerPage) {
         // TODO(jblomer): use (fewer) calls to AppendV to write the data page-by-page
         for (unsigned i = 0; i < count; ++i) {
            Append(RColumnElementBase(elemArray, i));
         }
         return;
      }

      void *dst = fHeadPage[fHeadPageIdx].GrowUnchecked(count);

      // The check for flushing the shadow page is more complicated than for the Append() case
      // because we don't necessarily fill up to exactly fApproxNElementsPerPage / 2 elements;
      // we might instead jump over the 50% fill level
      if ((fHeadPage[fHeadPageIdx].GetNElements() <= fApproxNElementsPerPage / 2) &&
          (fHeadPage[fHeadPageIdx].GetNElements() + count > fApproxNElementsPerPage / 2))
      {
         FlushShadowHeadPage();
      }

      elemArray.WriteTo(dst, count);
      fNElements += count;

      // Note that by the very first check, we cannot have filled more than fApproxNElementsPerPage elements
      if (fHeadPage[fHeadPageIdx].GetNElements() == fApproxNElementsPerPage)
         SwapHeadPages();
   }

   void Read(const NTupleSize_t globalIndex, RColumnElementBase *element) {
      if (!fCurrentPage.Contains(globalIndex)) {
         MapPage(globalIndex);
         R__ASSERT(fCurrentPage.Contains(globalIndex));
      }
      void *src = static_cast<unsigned char *>(fCurrentPage.GetBuffer()) +
                  (globalIndex - fCurrentPage.GetGlobalRangeFirst()) * element->GetSize();
      element->ReadFrom(src, 1);
   }

   void Read(const RClusterIndex &clusterIndex, RColumnElementBase *element) {
      if (!fCurrentPage.Contains(clusterIndex)) {
         MapPage(clusterIndex);
      }
      void *src = static_cast<unsigned char *>(fCurrentPage.GetBuffer()) +
                  (clusterIndex.GetIndex() - fCurrentPage.GetClusterRangeFirst()) * element->GetSize();
      element->ReadFrom(src, 1);
   }

   void ReadV(const NTupleSize_t globalIndex, const ClusterSize_t::ValueType count, RColumnElementBase *elemArray) {
      R__ASSERT(count > 0);
      if (!fCurrentPage.Contains(globalIndex)) {
         MapPage(globalIndex);
      }
      NTupleSize_t idxInPage = globalIndex - fCurrentPage.GetGlobalRangeFirst();

      void *src = static_cast<unsigned char *>(fCurrentPage.GetBuffer()) + idxInPage * elemArray->GetSize();
      if (globalIndex + count <= fCurrentPage.GetGlobalRangeLast() + 1) {
         elemArray->ReadFrom(src, count);
      } else {
         ClusterSize_t::ValueType nBatch = fCurrentPage.GetNElements() - idxInPage;
         elemArray->ReadFrom(src, nBatch);
         RColumnElementBase elemTail(*elemArray, nBatch);
         ReadV(globalIndex + nBatch, count - nBatch, &elemTail);
      }
   }

   void ReadV(const RClusterIndex &clusterIndex, const ClusterSize_t::ValueType count, RColumnElementBase *elemArray)
   {
      if (!fCurrentPage.Contains(clusterIndex)) {
         MapPage(clusterIndex);
      }
      NTupleSize_t idxInPage = clusterIndex.GetIndex() - fCurrentPage.GetClusterRangeFirst();

      void* src = static_cast<unsigned char *>(fCurrentPage.GetBuffer()) + idxInPage * elemArray->GetSize();
      if (clusterIndex.GetIndex() + count <= fCurrentPage.GetClusterRangeLast() + 1) {
         elemArray->ReadFrom(src, count);
      } else {
         ClusterSize_t::ValueType nBatch = fCurrentPage.GetNElements() - idxInPage;
         elemArray->ReadFrom(src, nBatch);
         RColumnElementBase elemTail(*elemArray, nBatch);
         ReadV(RClusterIndex(clusterIndex.GetClusterId(), clusterIndex.GetIndex() + nBatch), count - nBatch, &elemTail);
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
      if (!fCurrentPage.Contains(globalIndex)) {
         MapPage(globalIndex);
      }
      // +1 to go from 0-based indexing to 1-based number of items
      nItems = fCurrentPage.GetGlobalRangeLast() - globalIndex + 1;
      return reinterpret_cast<CppT*>(
         static_cast<unsigned char *>(fCurrentPage.GetBuffer()) +
         (globalIndex - fCurrentPage.GetGlobalRangeFirst()) * RColumnElement<CppT>::kSize);
   }

   template <typename CppT>
   CppT *MapV(const RClusterIndex &clusterIndex, NTupleSize_t &nItems) {
      if (!fCurrentPage.Contains(clusterIndex)) {
         MapPage(clusterIndex);
      }
      // +1 to go from 0-based indexing to 1-based number of items
      nItems = fCurrentPage.GetClusterRangeLast() - clusterIndex.GetIndex() + 1;
      return reinterpret_cast<CppT*>(
         static_cast<unsigned char *>(fCurrentPage.GetBuffer()) +
         (clusterIndex.GetIndex() - fCurrentPage.GetClusterRangeFirst()) * RColumnElement<CppT>::kSize);
   }

   NTupleSize_t GetGlobalIndex(const RClusterIndex &clusterIndex) {
      if (!fCurrentPage.Contains(clusterIndex)) {
         MapPage(clusterIndex);
      }
      return fCurrentPage.GetClusterInfo().GetIndexOffset() + clusterIndex.GetIndex();
   }

   RClusterIndex GetClusterIndex(NTupleSize_t globalIndex) {
      if (!fCurrentPage.Contains(globalIndex)) {
         MapPage(globalIndex);
      }
      return RClusterIndex(fCurrentPage.GetClusterInfo().GetId(),
                           globalIndex - fCurrentPage.GetClusterInfo().GetIndexOffset());
   }

   /// For offset columns only, look at the two adjacent values that define a collection's coordinates
   void GetCollectionInfo(const NTupleSize_t globalIndex, RClusterIndex *collectionStart, ClusterSize_t *collectionSize)
   {
      auto idxStart = (globalIndex == 0) ? 0 : *Map<ClusterSize_t>(globalIndex - 1);
      auto idxEnd = *Map<ClusterSize_t>(globalIndex);
      auto selfOffset = fCurrentPage.GetClusterInfo().GetIndexOffset();
      if (globalIndex == selfOffset) {
         // Passed cluster boundary
         idxStart = 0;
      }
      *collectionSize = idxEnd - idxStart;
      *collectionStart = RClusterIndex(fCurrentPage.GetClusterInfo().GetId(), idxStart);
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
      *varIndex = RClusterIndex(fCurrentPage.GetClusterInfo().GetId(), varSwitch->GetIndex());
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
   RPageSource *GetPageSource() const { return fPageSource; }
   RPageStorage::ColumnHandle_t GetHandleSource() const { return fHandleSource; }
   RPageStorage::ColumnHandle_t GetHandleSink() const { return fHandleSink; }
   RNTupleVersion GetVersion() const { return RNTupleVersion(); }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
