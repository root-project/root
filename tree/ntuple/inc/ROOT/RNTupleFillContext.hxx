/// \file ROOT/RNTupleFillContext.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleFillContext
#define ROOT_RNTupleFillContext

#include <ROOT/RConfig.hxx> // for R__unlikely
#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RRawPtrWriteEntry.hxx>
#include <ROOT/RNTupleFillStatus.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleTypes.hxx>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleFillContext
\ingroup NTuple
\brief A context for filling entries (data) into clusters of an RNTuple

An output cluster can be filled with entries. The caller has to make sure that the data that gets filled into a cluster
is not modified for the time of the Fill() call. The fill call serializes the C++ object into the column format and
writes data into the corresponding column page buffers.  Writing of the buffers to storage is deferred and can be
triggered by FlushCluster() or by destructing the context.  On I/O errors, an exception is thrown.

Instances of this class are not meant to be used in isolation and can be created from an RNTupleParallelWriter. For
sequential writing, please refer to RNTupleWriter.
*/
// clang-format on
class RNTupleFillContext {
   friend class ROOT::RNTupleWriter;
   friend class RNTupleParallelWriter;

private:
   /// The page sink's parallel page compression scheduler if IMT is on.
   /// Needs to be destructed after the page sink is destructed and so declared before.
   std::unique_ptr<ROOT::Internal::RPageStorage::RTaskScheduler> fZipTasks;
   std::unique_ptr<ROOT::Internal::RPageSink> fSink;
   /// Needs to be destructed before fSink
   std::unique_ptr<ROOT::RNTupleModel> fModel;

   Detail::RNTupleMetrics fMetrics;

   ROOT::NTupleSize_t fLastFlushed = 0;
   ROOT::NTupleSize_t fNEntries = 0;
   /// Keeps track of the number of bytes written into the current cluster
   std::size_t fUnzippedClusterSize = 0;
   /// The total number of bytes written to storage (i.e., after compression)
   std::uint64_t fNBytesFlushed = 0;
   /// The total number of bytes filled into all the so far committed clusters,
   /// i.e. the uncompressed size of the written clusters
   std::uint64_t fNBytesFilled = 0;
   /// Limit for committing cluster no matter the other tunables
   std::size_t fMaxUnzippedClusterSize;
   /// Estimator of uncompressed cluster size, taking into account the estimated compression ratio
   std::size_t fUnzippedClusterSizeEst;

   /// Whether to enable staged cluster committing, where only an explicit call to CommitStagedClusters() will logically
   /// append the clusters to the RNTuple.
   bool fStagedClusterCommitting = false;
   /// Vector of currently staged clusters.
   std::vector<ROOT::Internal::RPageSink::RStagedCluster> fStagedClusters;

   template <typename Entry>
   void FillNoFlushImpl(Entry &entry, ROOT::RNTupleFillStatus &status)
   {
      if (R__unlikely(entry.GetModelId() != fModel->GetModelId()))
         throw RException(R__FAIL("mismatch between entry and model"));

      const std::size_t bytesWritten = entry.Append();
      fUnzippedClusterSize += bytesWritten;
      fNEntries++;

      status.fNEntriesSinceLastFlush = fNEntries - fLastFlushed;
      status.fUnzippedClusterSize = fUnzippedClusterSize;
      status.fLastEntrySize = bytesWritten;
      status.fShouldFlushCluster =
         (fUnzippedClusterSize >= fMaxUnzippedClusterSize) || (fUnzippedClusterSize >= fUnzippedClusterSizeEst);
   }
   template <typename Entry>
   std::size_t FillImpl(Entry &entry)
   {
      ROOT::RNTupleFillStatus status;
      FillNoFlush(entry, status);
      if (status.ShouldFlushCluster())
         FlushCluster();
      return status.GetLastEntrySize();
   }

   RNTupleFillContext(std::unique_ptr<ROOT::RNTupleModel> model, std::unique_ptr<ROOT::Internal::RPageSink> sink);
   RNTupleFillContext(const RNTupleFillContext &) = delete;
   RNTupleFillContext &operator=(const RNTupleFillContext &) = delete;

public:
   ~RNTupleFillContext();

   /// Fill an entry into this context, but don't commit the cluster. The calling code must pass an RNTupleFillStatus
   /// and check RNTupleFillStatus::ShouldFlushCluster.
   ///
   /// This method will check the entry's model ID to ensure it comes from the context's own model or throw an exception
   /// otherwise.
   void FillNoFlush(ROOT::REntry &entry, ROOT::RNTupleFillStatus &status) { FillNoFlushImpl(entry, status); }
   /// Fill an entry into this context.  This method will check the entry's model ID to ensure it comes from the
   /// context's own model or throw an exception otherwise.
   /// \return The number of uncompressed bytes written.
   std::size_t Fill(ROOT::REntry &entry) { return FillImpl(entry); }

   /// Fill an RRawPtrWriteEntry into this context, but don't commit the cluster. The calling code must pass an
   /// RNTupleFillStatus and check RNTupleFillStatus::ShouldFlushCluster.
   ///
   /// This method will check the entry's model ID to ensure it comes from the context's own model or throw an exception
   /// otherwise.
   void FillNoFlush(Detail::RRawPtrWriteEntry &entry, ROOT::RNTupleFillStatus &status)
   {
      FillNoFlushImpl(entry, status);
   }
   /// Fill an RRawPtrWriteEntry into this context.  This method will check the entry's model ID to ensure it comes
   /// from the context's own model or throw an exception otherwise.
   /// \return The number of uncompressed bytes written.
   std::size_t Fill(Detail::RRawPtrWriteEntry &entry) { return FillImpl(entry); }

   /// Flush column data, preparing for CommitCluster or to reduce memory usage. This will trigger compression of pages,
   /// but not actually write to storage.
   void FlushColumns();
   /// Flush so far filled entries to storage
   void FlushCluster();
   /// Logically append staged clusters to the RNTuple.
   void CommitStagedClusters();

   const ROOT::RNTupleModel &GetModel() const { return *fModel; }
   std::unique_ptr<ROOT::REntry> CreateEntry() const { return fModel->CreateEntry(); }
   std::unique_ptr<Detail::RRawPtrWriteEntry> CreateRawPtrWriteEntry() const
   {
      return fModel->CreateRawPtrWriteEntry();
   }

   /// Return the entry number that was last flushed in a cluster.
   ROOT::NTupleSize_t GetLastFlushed() const { return fLastFlushed; }
   /// Return the number of entries filled so far.
   ROOT::NTupleSize_t GetNEntries() const { return fNEntries; }

   void EnableStagedClusterCommitting(bool val = true)
   {
      if (!val && !fStagedClusters.empty()) {
         throw RException(R__FAIL("cannot disable staged committing with pending clusters"));
      }
      fStagedClusterCommitting = val;
   }
   bool IsStagedClusterCommittingEnabled() const { return fStagedClusterCommitting; }

   void EnableMetrics() { fMetrics.Enable(); }
   const Detail::RNTupleMetrics &GetMetrics() const { return fMetrics; }
}; // class RNTupleFillContext

} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleFillContext
