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
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPageSinkBuf.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>

namespace {

using ROOT::Experimental::DescriptorId_t;
using ROOT::Experimental::NTupleSize_t;
using ROOT::Experimental::RException;
using ROOT::Experimental::RFieldBase;
using ROOT::Experimental::RNTupleDescriptor;
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::Detail::RNTupleMetrics;
using ROOT::Experimental::Detail::RNTuplePlainCounter;
using ROOT::Experimental::Detail::RNTuplePlainTimer;
using ROOT::Experimental::Detail::RNTupleTickCounter;
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

   void CommitPage(ColumnHandle_t, const RPage &) final
   {
      throw RException(R__FAIL("should never commit single pages via RPageSynchronizingSink"));
   }
   RWrittenPage WriteSealedPage(DescriptorId_t, const RSealedPage &) final
   {
      throw RException(R__FAIL("should never commit sealed pages via RPageSynchronizingSink"));
   }
   void CommitWrittenPage(DescriptorId_t, const RWrittenPage &) final
   {
      throw RException(R__FAIL("should never commit written pages via RPageSynchronizingSink"));
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
   void CommitDataset() final { throw RException(R__FAIL("should never commit dataset via RPageSynchronizingSink")); }

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements) final
   {
      return fInnerSink->ReservePage(columnHandle, nElements);
   }
   void ReleasePage(RPage &page) final { fInnerSink->ReleasePage(page); }

   RSinkGuard GetSinkGuard() final { return RSinkGuard(fMutex); }
};

/// An internal RPageSink that enables multiple RNTupleFillContext to write into a single common RPageSink, without the
/// use of RPageSinkBufs.
///
/// The setup with two contexts looks as follows:
///
/// owned by RNTupleFillContext
///           |
/// RPageUnbufferedSyncSink ---+
///                            |
///   (via raw fInnerSink ptr) +-- RPageSink (usually a persistent sink)
///                            |
/// RPageUnbufferedSyncSink ---+
///           |
/// owned by RNTupleFillContext
///
/// The mutex used by the synchronizing sinks is owned by the RNTupleParallelWriter that also owns the original model,
/// the "final" sink (usually a persistent sink) and keeps weak_ptr's of the contexts (to make sure they are destroyed
/// before the writer is destructed).
class RPageUnbufferedSyncSink : public RPageSink {
private:
   struct RColumnBuf {
      std::vector<RWrittenPage> fWrittenPages;
   };

   std::vector<RColumnBuf> fBufferedColumns;

   struct RCounters {
      RNTuplePlainCounter &fTimeWallCriticalSection;
      RNTupleTickCounter<RNTuplePlainCounter> &fTimeCpuCriticalSection;
   };
   std::unique_ptr<RCounters> fCounters;

   /// The wrapped inner sink, not owned by this class.
   RPageSink *fInnerSink;
   std::mutex *fMutex;
   DescriptorId_t fNColumns = 0;

public:
   explicit RPageUnbufferedSyncSink(RPageSink &inner, std::mutex &mutex)
      : RPageSink(inner.GetNTupleName(), inner.GetWriteOptions()), fInnerSink(&inner), fMutex(&mutex)
   {
      fCompressor = std::make_unique<ROOT::Experimental::Internal::RNTupleCompressor>();

      fMetrics = RNTupleMetrics("RPageUnbufferedSyncSink");
      fCounters = std::make_unique<RCounters>(
         RCounters{*fMetrics.MakeCounter<RNTuplePlainCounter *>("timeWallCriticalSection", "ns",
                                                                "wall clock time spent in critical sections"),
                   *fMetrics.MakeCounter<RNTupleTickCounter<RNTuplePlainCounter> *>(
                      "timeCpuCriticalSection", "ns", "CPU time spent in critical section")});
      // Do not observe the sink's metrics: It will contain some counters for all threads, which is misleading for the
      // users.
      // fMetrics.ObserveMetrics(fSink->GetMetrics());
   }
   RPageUnbufferedSyncSink(const RPageUnbufferedSyncSink &) = delete;
   RPageUnbufferedSyncSink &operator=(const RPageUnbufferedSyncSink &) = delete;

   const RNTupleDescriptor &GetDescriptor() const final { return fInnerSink->GetDescriptor(); }

   ColumnHandle_t AddColumn(DescriptorId_t, const RColumn &column) final { return {fNColumns++, &column}; }
   void InitImpl(RNTupleModel &model) final
   {
      for (auto &f : model.GetFieldZero()) {
         CallConnectPageSinkOnField(f, *this);
      }
      fBufferedColumns.resize(fNColumns);
   }
   void UpdateSchema(const RNTupleModelChangeset &, NTupleSize_t) final
   {
      throw RException(R__FAIL("UpdateSchema not supported via RPageUnbufferedSyncSink"));
   }

   void CommitPage(ColumnHandle_t columnHandle, const RPage &page) final
   {
      // Compress outside the critical section.
      auto element = columnHandle.fColumn->GetElement();
      RSealedPage sealedPage = SealPage(page, *element, GetWriteOptions().GetCompression());

      auto colId = columnHandle.fPhysicalId;
      {
         RSinkGuard guard(GetSinkGuard());
         RNTuplePlainTimer timer(fCounters->fTimeWallCriticalSection, fCounters->fTimeCpuCriticalSection);
         fBufferedColumns[colId].fWrittenPages.push_back(fInnerSink->WriteSealedPage(colId, sealedPage));
      }
   }
   RWrittenPage WriteSealedPage(DescriptorId_t, const RSealedPage &) final
   {
      throw RException(R__FAIL("should never commit sealed pages via RPageUnbufferedSyncSink"));
   }
   void CommitWrittenPage(DescriptorId_t, const RWrittenPage &) final
   {
      throw RException(R__FAIL("should never commit written pages via RPageUnbufferedSyncSink"));
   }
   void CommitSealedPage(DescriptorId_t, const RSealedPage &) final
   {
      throw RException(R__FAIL("should never commit sealed pages via RPageUnbufferedSyncSink"));
   }
   void CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup>) final
   {
      throw RException(R__FAIL("should never commit sealed pages via RPageUnbufferedSyncSink"));
   }
   std::uint64_t CommitCluster(NTupleSize_t nNewEntries) final
   {
      RSinkGuard guard(GetSinkGuard());
      RNTuplePlainTimer timer(fCounters->fTimeWallCriticalSection, fCounters->fTimeCpuCriticalSection);
      for (DescriptorId_t colId = 0; colId < fNColumns; colId++) {
         auto &col = fBufferedColumns[colId];
         for (const auto &writtenPage : col.fWrittenPages) {
            fInnerSink->CommitWrittenPage(colId, writtenPage);
         }
         col.fWrittenPages.clear();
      }
      return fInnerSink->CommitCluster(nNewEntries);
   }
   void CommitClusterGroup() final
   {
      throw RException(R__FAIL("should never commit cluster group via RPageUnbufferedSyncSink"));
   }
   void CommitDataset() final { throw RException(R__FAIL("should never commit dataset via RPageUnbufferedSyncSink")); }

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements) final
   {
      return fInnerSink->ReservePage(columnHandle, nElements);
   }
   void ReleasePage(RPage &page) final { fInnerSink->ReleasePage(page); }

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
   auto sink = Internal::RPagePersistentSink::Create(ntupleName, storage, options);
   // Cannot use std::make_unique because the constructor of RNTupleParallelWriter is private.
   return std::unique_ptr<RNTupleParallelWriter>(new RNTupleParallelWriter(std::move(model), std::move(sink)));
}

std::unique_ptr<ROOT::Experimental::RNTupleParallelWriter>
ROOT::Experimental::RNTupleParallelWriter::Append(std::unique_ptr<RNTupleModel> model, std::string_view ntupleName,
                                                  TFile &file, const RNTupleWriteOptions &options)
{
   auto sink = std::make_unique<Internal::RPageSinkFile>(ntupleName, file, options);
   // Cannot use std::make_unique because the constructor of RNTupleParallelWriter is private.
   return std::unique_ptr<RNTupleParallelWriter>(new RNTupleParallelWriter(std::move(model), std::move(sink)));
}

std::shared_ptr<ROOT::Experimental::RNTupleFillContext> ROOT::Experimental::RNTupleParallelWriter::CreateFillContext()
{
   std::lock_guard g(fMutex);

   auto model = fModel->Clone();

   std::unique_ptr<RPageSink> sink;
   if (fSink->GetWriteOptions().GetUseBufferedWrite()) {
      sink = std::make_unique<Internal::RPageSinkBuf>(std::make_unique<RPageSynchronizingSink>(*fSink, fSinkMutex));
   } else {
      sink = std::make_unique<RPageUnbufferedSyncSink>(*fSink, fSinkMutex);
   }

   // Cannot use std::make_shared because the constructor of RNTupleFillContext is private. Also it would mean that the
   // (direct) memory of all contexts stays around until the vector of weak_ptr's is cleared.
   std::shared_ptr<RNTupleFillContext> context(new RNTupleFillContext(std::move(model), std::move(sink)));
   fFillContexts.push_back(context);
   return context;
}
