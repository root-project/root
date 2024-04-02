/// \file ROOT/RPageNullSink.hxx
/// \ingroup NTuple ROOT7
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-01-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageNullSink
#define ROOT7_RPageNullSink

#include <ROOT/RColumn.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RPageStorage.hxx>

namespace ROOT {
namespace Experimental {
namespace Internal {

/**
\class ROOT::Experimental::Internal::RPageNullSink
\ingroup NTuple
\brief Dummy sink that discards all pages

The RPageNullSink class is for internal testing only and can be used to measure the software overhead of serializing
elements into pages, without actually writing them onto disk or even serializing the RNTuple headers and footers.
*/
class RPageNullSink : public RPageSink {
   RPageAllocatorHeap fPageAllocator{};
   DescriptorId_t fNColumns = 0;
   std::uint64_t fNBytesCurrentCluster = 0;

public:
   RPageNullSink(std::string_view ntupleName, const RNTupleWriteOptions &options) : RPageSink(ntupleName, options) {}

   ColumnHandle_t AddColumn(DescriptorId_t, const RColumn &column) final { return {fNColumns++, &column}; }

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements) final
   {
      auto elementSize = columnHandle.fColumn->GetElement()->GetSize();
      return fPageAllocator.NewPage(columnHandle.fPhysicalId, elementSize, nElements);
   }
   void ReleasePage(RPage &page) final { fPageAllocator.DeletePage(page); }

   const RNTupleDescriptor &GetDescriptor() const final
   {
      static RNTupleDescriptor descriptor;
      return descriptor;
   }

   void ConnectFields(const std::vector<RFieldBase *> &fields, NTupleSize_t firstEntry)
   {
      auto connectField = [&](RFieldBase &f) { CallConnectPageSinkOnField(f, *this, firstEntry); };
      for (auto *f : fields) {
         connectField(*f);
         for (auto &descendant : *f) {
            connectField(descendant);
         }
      }
   }
   void InitImpl(RNTupleModel &model) final { ConnectFields(model.GetFieldZero().GetSubFields(), 0); }
   void UpdateSchema(const RNTupleModelChangeset &changeset, NTupleSize_t firstEntry) final
   {
      ConnectFields(changeset.fAddedFields, firstEntry);
   }

   void CommitPage(ColumnHandle_t, const RPage &page) final { fNBytesCurrentCluster += page.GetNBytes(); }
   void CommitSealedPage(DescriptorId_t, const RSealedPage &page) final { fNBytesCurrentCluster += page.fSize; }
   void CommitSealedPageV(std::span<RSealedPageGroup> ranges) final
   {
      for (auto &range : ranges) {
         for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt) {
            fNBytesCurrentCluster += sealedPageIt->fSize;
         }
      }
   }

   std::uint64_t CommitCluster(NTupleSize_t) final
   {
      std::uint64_t bytes = fNBytesCurrentCluster;
      fNBytesCurrentCluster = 0;
      return bytes;
   }
   void CommitClusterGroup() final {}
   void CommitDataset() final {}
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
