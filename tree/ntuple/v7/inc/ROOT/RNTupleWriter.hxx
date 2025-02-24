/// \file ROOT/RNTupleWriter.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-20
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleWriter
#define ROOT7_RNTupleWriter

#include <ROOT/RConfig.hxx> // for R__unlikely
#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTupleFillContext.hxx>
#include <ROOT/RNTupleFillStatus.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPageStorage.hxx>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

class TDirectory;

namespace ROOT {

class RNTupleWriteOptions;

namespace Experimental {
namespace Internal {
// Non-public factory method for an RNTuple writer that uses an already constructed page sink
std::unique_ptr<RNTupleWriter>
CreateRNTupleWriter(std::unique_ptr<RNTupleModel> model, std::unique_ptr<Internal::RPageSink> sink);
} // namespace Internal

// clang-format off
/**
\class ROOT::Experimental::RNTupleWriter
\ingroup NTuple
\brief An RNTuple that gets filled with entries (data) and writes them to storage

An output ntuple can be filled with entries. The caller has to make sure that the data that gets filled into an ntuple
is not modified for the time of the Fill() call. The fill call serializes the C++ object into the column format and
writes data into the corresponding column page buffers.  Writing of the buffers to storage is deferred and can be
triggered by FlushCluster() or by destructing the writer.  On I/O errors, an exception is thrown.
*/
// clang-format on
class RNTupleWriter {
   friend RNTupleModel::RUpdater;
   friend std::unique_ptr<RNTupleWriter>
      Internal::CreateRNTupleWriter(std::unique_ptr<RNTupleModel>, std::unique_ptr<Internal::RPageSink>);

private:
   /// The page sink's parallel page compression scheduler if IMT is on.
   /// Needs to be destructed after the page sink (in the fill context) is destructed and so declared before.
   std::unique_ptr<Internal::RPageStorage::RTaskScheduler> fZipTasks;
   RNTupleFillContext fFillContext;
   Detail::RNTupleMetrics fMetrics;

   ROOT::NTupleSize_t fLastCommittedClusterGroup = 0;

   RNTupleWriter(std::unique_ptr<RNTupleModel> model, std::unique_ptr<Internal::RPageSink> sink);

   RNTupleModel &GetUpdatableModel();
   Internal::RPageSink &GetSink() { return *fFillContext.fSink; }

   // Helper function that is called from CommitCluster() when necessary
   void CommitClusterGroup();

   /// Create a writer, potentially wrapping the sink in a RPageSinkBuf.
   static std::unique_ptr<RNTupleWriter> Create(std::unique_ptr<RNTupleModel> model,
                                                std::unique_ptr<Internal::RPageSink> sink,
                                                const ROOT::RNTupleWriteOptions &options);

public:
   /// Throws an exception if the model is null.
   static std::unique_ptr<RNTupleWriter>
   Recreate(std::unique_ptr<RNTupleModel> model, std::string_view ntupleName, std::string_view storage,
            const ROOT::RNTupleWriteOptions &options = ROOT::RNTupleWriteOptions());
   static std::unique_ptr<RNTupleWriter>
   Recreate(std::initializer_list<std::pair<std::string_view, std::string_view>> fields, std::string_view ntupleName,
            std::string_view storage, const ROOT::RNTupleWriteOptions &options = ROOT::RNTupleWriteOptions());
   /// Throws an exception if the model is null.
   static std::unique_ptr<RNTupleWriter> Append(std::unique_ptr<RNTupleModel> model, std::string_view ntupleName,
                                                TDirectory &fileOrDirectory,
                                                const ROOT::RNTupleWriteOptions &options = ROOT::RNTupleWriteOptions());
   RNTupleWriter(const RNTupleWriter &) = delete;
   RNTupleWriter &operator=(const RNTupleWriter &) = delete;
   ~RNTupleWriter();

   /// The simplest user interface if the default entry that comes with the ntuple model is used.
   /// \return The number of uncompressed bytes written.
   std::size_t Fill() { return fFillContext.Fill(fFillContext.fModel->GetDefaultEntry()); }
   /// Multiple entries can have been instantiated from the ntuple model.  This method will perform
   /// a light check whether the entry comes from the ntuple's own model.
   /// \return The number of uncompressed bytes written.
   std::size_t Fill(REntry &entry) { return fFillContext.Fill(entry); }
   /// Fill an entry into this ntuple, but don't commit the cluster. The calling code must pass an RNTupleFillStatus
   /// and check RNTupleFillStatus::ShouldFlushCluster.
   void FillNoFlush(REntry &entry, RNTupleFillStatus &status) { fFillContext.FillNoFlush(entry, status); }
   /// Flush column data, preparing for CommitCluster or to reduce memory usage. This will trigger compression of pages,
   /// but not actually write to storage (unless buffered writing is turned off).
   void FlushColumns() { fFillContext.FlushColumns(); }
   /// Flush so far filled entries to storage
   void FlushCluster() { fFillContext.FlushCluster(); }
   /// Ensure that the data from the so far seen Fill calls has been written to storage
   void CommitCluster(bool commitClusterGroup = false)
   {
      fFillContext.FlushCluster();
      if (commitClusterGroup)
         CommitClusterGroup();
   }
   /// Closes the underlying file (page sink) and expires the model. Automatically called on destruct.
   /// Once the dataset is committed, calls to Fill(), [Commit|Flush]Cluster(), FlushColumns(), CreateEntry(),
   /// and model updating fail.
   void CommitDataset();

   std::unique_ptr<REntry> CreateEntry() { return fFillContext.CreateEntry(); }

   /// Return the entry number that was last flushed in a cluster.
   ROOT::NTupleSize_t GetLastFlushed() const { return fFillContext.GetLastFlushed(); }
   /// Return the entry number that was last committed in a cluster.
   ROOT::NTupleSize_t GetLastCommitted() const { return fFillContext.GetLastFlushed(); }
   /// Return the entry number that was last committed in a cluster group.
   ROOT::NTupleSize_t GetLastCommittedClusterGroup() const { return fLastCommittedClusterGroup; }
   /// Return the number of entries filled so far.
   ROOT::NTupleSize_t GetNEntries() const { return fFillContext.GetNEntries(); }

   void EnableMetrics() { fMetrics.Enable(); }
   const Detail::RNTupleMetrics &GetMetrics() const { return fMetrics; }

   const RNTupleModel &GetModel() const { return *fFillContext.fModel; }

   /// Get a `RNTupleModel::RUpdater` that provides limited support for incremental updates to the underlying
   /// model, e.g. addition of new fields.
   ///
   /// **Example: add a new field after the model has been used to construct a `RNTupleWriter` object**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTuple.hxx>
   /// using ROOT::Experimental::RNTupleModel;
   /// using ROOT::Experimental::RNTupleWriter;
   ///
   /// auto model = RNTupleModel::Create();
   /// auto fldFloat = model->MakeField<float>("fldFloat");
   /// auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", "some/file.root");
   /// auto updater = writer->CreateModelUpdater();
   /// updater->BeginUpdate();
   /// updater->AddField(std::make_unique<RField<float>>("pt"));
   /// updater->CommitUpdate();
   ///
   /// // ...
   /// ~~~
   std::unique_ptr<RNTupleModel::RUpdater> CreateModelUpdater()
   {
      return std::make_unique<RNTupleModel::RUpdater>(*this);
   }
}; // class RNTupleWriter

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleWriter
