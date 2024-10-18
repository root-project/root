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
#include <ROOT/RFieldBase.hxx>
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
   DescriptorId_t fNColumns = 0;
   std::uint64_t fNBytesCurrentCluster = 0;

public:
   RPageNullSink(std::string_view ntupleName, const RNTupleWriteOptions &options) : RPageSink(ntupleName, options) {}

   ColumnHandle_t AddColumn(DescriptorId_t, RColumn &column) final { return {fNColumns++, &column}; }

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
   void InitImpl(RNTupleModel &model) final
   {
      auto &fieldZero = GetFieldZeroOfModel(model);
      ConnectFields(fieldZero.GetSubFields(), 0);
   }
   void UpdateSchema(const RNTupleModelChangeset &changeset, NTupleSize_t firstEntry) final
   {
      ConnectFields(changeset.fAddedFields, firstEntry);
   }
   void UpdateExtraTypeInfo(const RExtraTypeInfoDescriptor &) final {}

   void CommitSuppressedColumn(ColumnHandle_t) final {}
   void CommitPage(ColumnHandle_t, const RPage &page) final { fNBytesCurrentCluster += page.GetNBytes(); }
   void CommitSealedPage(DescriptorId_t, const RSealedPage &page) final
   {
      fNBytesCurrentCluster += page.GetBufferSize();
   }
   void CommitSealedPageV(std::span<RSealedPageGroup> ranges) final
   {
      for (auto &range : ranges) {
         for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt) {
            fNBytesCurrentCluster += sealedPageIt->GetBufferSize();
         }
      }
   }

   RStagedCluster StageCluster(NTupleSize_t) final
   {
      RStagedCluster stagedCluster;
      stagedCluster.fNBytesWritten = fNBytesCurrentCluster;
      fNBytesCurrentCluster = 0;
      return stagedCluster;
   }
   void CommitStagedClusters(std::span<RStagedCluster>) final {}
   void CommitClusterGroup() final {}
   void CommitDatasetImpl() final {}
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
