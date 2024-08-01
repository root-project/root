/// \file RNTupleParallelWriter.cxx
/// \ingroup NTuple ROOT7
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-02-01
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleParallelWriter.hxx>

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RPageSinkBuf.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>

namespace {

using ROOT::Experimental::DescriptorId_t;
using ROOT::Experimental::NTupleSize_t;
using ROOT::Experimental::RException;
using ROOT::Experimental::RExtraTypeInfoDescriptor;
using ROOT::Experimental::RNTupleDescriptor;
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::Internal::RColumn;
using ROOT::Experimental::Internal::RNTupleModelChangeset;
using ROOT::Experimental::Internal::RPage;
using ROOT::Experimental::Internal::RPageSink;

/// An internal RPageSink that enables multiple RNTupleFillContext to write into a single common RPageSink.
///
/// The setup with two contexts looks as follows:
///
///      +------ owned by RNTupleFillContext ------+
///      |                                         |
/// RPageSinkBuf --- forwards to ---> RPageSynchronizingSink ---+
///                  (and owns)                                 |
///                                    (via raw fInnerSink ptr) +-- RPageSink (usually a persistent sink)
///                                                             |
/// RPageSinkBuf --- forwards to ---> RPageSynchronizingSink ---+
///      |           (and owns)                    |
///      |                                         |
///      +------ owned by RNTupleFillContext ------+
///
/// The mutex used by the synchronizing sinks is owned by the RNTupleParallelWriter that also owns the original model,
/// the "final" sink (usually a persistent sink) and keeps weak_ptr's of the contexts (to make sure they are destroyed
/// before the writer is destructed).
class RPageSynchronizingSink : public RPageSink {
private:
   /// The wrapped inner sink, not owned by this class.
   RPageSink *fInnerSink;
   std::mutex *fMutex;

public:
   explicit RPageSynchronizingSink(RPageSink &inner, std::mutex &mutex)
      : RPageSink(inner.GetNTupleName(), inner.GetWriteOptions()), fInnerSink(&inner), fMutex(&mutex)
   {
      // Do not observe the sink's metrics: It will contain some counters for all threads, which is misleading for the
      // users.
      // fMetrics.ObserveMetrics(fSink->GetMetrics());
   }
   RPageSynchronizingSink(const RPageSynchronizingSink &) = delete;
   RPageSynchronizingSink &operator=(const RPageSynchronizingSink &) = delete;

   const RNTupleDescriptor &GetDescriptor() const final { return fInnerSink->GetDescriptor(); }

   ColumnHandle_t AddColumn(DescriptorId_t, const RColumn &) final { return {}; }
   void InitImpl(RNTupleModel &) final {}
   void UpdateSchema(const RNTupleModelChangeset &, NTupleSize_t) final
   {
      throw RException(R__FAIL("UpdateSchema not supported via RPageSynchronizingSink"));
   }
   void UpdateExtraTypeInfo(const RExtraTypeInfoDescriptor &) final
   {
      throw RException(R__FAIL("UpdateExtraTypeInfo not supported via RPageSynchronizingSink"));
   }

   void CommitSuppressedColumn(ColumnHandle_t handle) final { fInnerSink->CommitSuppressedColumn(handle); }
   void CommitPage(ColumnHandle_t, const RPage &) final
   {
      throw RException(R__FAIL("should never commit single pages via RPageSynchronizingSink"));
   }
   void CommitSealedPage(DescriptorId_t, const RSealedPage &) final
   {
      throw RException(R__FAIL("should never commit sealed pages via RPageSynchronizingSink"));
   }
   void CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup> ranges) final
   {
      fInnerSink->CommitSealedPageV(ranges);
   }
   std::uint64_t CommitCluster(NTupleSize_t nNewEntries) final { return fInnerSink->CommitCluster(nNewEntries); }
   void CommitClusterGroup() final
   {
      throw RException(R__FAIL("should never commit cluster group via RPageSynchronizingSink"));
   }
   void CommitDatasetImpl() final
   {
      throw RException(R__FAIL("should never commit dataset via RPageSynchronizingSink"));
   }

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements) final
   {
      return fInnerSink->ReservePage(columnHandle, nElements);
   }

   RSinkGuard GetSinkGuard() final { return RSinkGuard(fMutex); }
};

} // namespace

ROOT::Experimental::RNTupleParallelWriter::RNTupleParallelWriter(std::unique_ptr<RNTupleModel> model,
                                                                 std::unique_ptr<Internal::RPageSink> sink)
   : fSink(std::move(sink)), fModel(std::move(model)), fMetrics("RNTupleParallelWriter")
{
   fModel->Freeze();
   fSink->Init(*fModel.get());
   fMetrics.ObserveMetrics(fSink->GetMetrics());
}

ROOT::Experimental::RNTupleParallelWriter::~RNTupleParallelWriter()
{
   for (const auto &context : fFillContexts) {
      if (!context.expired()) {
         R__LOG_ERROR(NTupleLog()) << "RNTupleFillContext has not been destructed";
         return;
      }
   }

   // Now commit all clusters as a cluster group and then the dataset.
   try {
      fSink->CommitClusterGroup();
      fSink->CommitDataset();
   } catch (const RException &err) {
      R__LOG_ERROR(NTupleLog()) << "failure committing ntuple: " << err.GetError().GetReport();
   }
}

std::unique_ptr<ROOT::Experimental::RNTupleParallelWriter>
ROOT::Experimental::RNTupleParallelWriter::Recreate(std::unique_ptr<RNTupleModel> model, std::string_view ntupleName,
                                                    std::string_view storage, const RNTupleWriteOptions &options)
{
   if (!options.GetUseBufferedWrite()) {
      throw RException(R__FAIL("parallel writing requires buffering"));
   }

   auto sink = Internal::RPagePersistentSink::Create(ntupleName, storage, options);
   // Cannot use std::make_unique because the constructor of RNTupleParallelWriter is private.
   return std::unique_ptr<RNTupleParallelWriter>(new RNTupleParallelWriter(std::move(model), std::move(sink)));
}

std::unique_ptr<ROOT::Experimental::RNTupleParallelWriter>
ROOT::Experimental::RNTupleParallelWriter::Append(std::unique_ptr<RNTupleModel> model, std::string_view ntupleName,
                                                  TFile &file, const RNTupleWriteOptions &options)
{
   if (!options.GetUseBufferedWrite()) {
      throw RException(R__FAIL("parallel writing requires buffering"));
   }

   auto sink = std::make_unique<Internal::RPageSinkFile>(ntupleName, file, options);
   // Cannot use std::make_unique because the constructor of RNTupleParallelWriter is private.
   return std::unique_ptr<RNTupleParallelWriter>(new RNTupleParallelWriter(std::move(model), std::move(sink)));
}

std::shared_ptr<ROOT::Experimental::RNTupleFillContext> ROOT::Experimental::RNTupleParallelWriter::CreateFillContext()
{
   std::lock_guard g(fMutex);

   auto model = fModel->Clone();

   // TODO: Think about honoring RNTupleWriteOptions::SetUseBufferedWrite(false); this requires synchronization on every
   // call to CommitPage() *and* preparing multiple cluster descriptors in parallel!
   auto sink = std::make_unique<Internal::RPageSinkBuf>(std::make_unique<RPageSynchronizingSink>(*fSink, fSinkMutex));

   // Cannot use std::make_shared because the constructor of RNTupleFillContext is private. Also it would mean that the
   // (direct) memory of all contexts stays around until the vector of weak_ptr's is cleared.
   std::shared_ptr<RNTupleFillContext> context(new RNTupleFillContext(std::move(model), std::move(sink)));
   fFillContexts.push_back(context);
   return context;
}
