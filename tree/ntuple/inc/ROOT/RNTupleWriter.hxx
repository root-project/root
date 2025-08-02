/// \file ROOT/RNTupleWriter.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-20

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleWriter
#define ROOT_RNTupleWriter

#include <ROOT/RConfig.hxx> // for R__unlikely
#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTupleFillContext.hxx>
#include <ROOT/RNTupleFillStatus.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RRawPtrWriteEntry.hxx>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

class TDirectory;

namespace ROOT {

class RNTupleWriteOptions;

namespace Internal {
// Non-public factory method for an RNTuple writer that uses an already constructed page sink
std::unique_ptr<RNTupleWriter>
CreateRNTupleWriter(std::unique_ptr<ROOT::RNTupleModel> model, std::unique_ptr<Internal::RPageSink> sink);
} // namespace Internal

// clang-format off
/**
\class ROOT::RNTupleWriter
\ingroup NTuple
\brief An RNTuple that gets filled with entries (data) and writes them to storage

RNTupleWriter is an interface for writing RNTuples to storage. It can be instantiated using the static functions
Append() and Recreate(), providing an RNTupleModel that defines the schema of the data to be written.

An RNTuple can be thought of as a table, whose columns are defined by its schema (i.e. by its associated RNTupleModel,
whose Fields map to 0 or more columns).
Writing into an RNTuple happens by filling *entries* into the RNTupleWriter, which make up the rows of the table.
The simplest way to do so is by:

- retrieving a (shared) pointer to each Field's value;
- writing a value into each pointer;
- calling `writer->Fill()` to commit the entry with all the current pointer values.

~~~ {.cpp}
#include <ROOT/RNTupleWriter.hxx>

/// 1. Create the model.
auto model = ROOT::RNTupleModel::Create();
// Define the schema by adding Fields to the model.
// MakeField returns a shared_ptr to the value to be written (in this case, a shared_ptr<int>)
auto pFoo = model->MakeField<int>("foo");

/// 2. Create writer from the model.
auto writer = ROOT::RNTupleReader::Recreate(std::move(model), "myNTuple", "some/file.root");

/// 3. Write into it.
for (int i = 0; i < 10; ++i) {
   // Assign the value you want to each RNTuple Field (in this case there is only one Field "foo").
   *pFoo = i;

   // Fill() writes the entire entry to the RNTuple.
   // After calling Fill() you can safely write another value into `pFoo` knowing that the previous one was
   // already saved.
   writer->Fill();
}

// On destruction, the writer will flush the written data to disk.
~~~

The caller has to make sure that the data that gets filled into an RNTuple is not modified for the time of the
Fill() call. The Fill call serializes the C++ object into the column format and
writes data into the corresponding column page buffers.

The actual writing of the buffers to storage is deferred and can be triggered by FlushCluster() or by
destructing the writer.

On I/O errors, a ROOT::RException is thrown.

*/
// clang-format on
class RNTupleWriter {
   friend ROOT::RNTupleModel::RUpdater;
   friend std::unique_ptr<RNTupleWriter>
      Internal::CreateRNTupleWriter(std::unique_ptr<ROOT::RNTupleModel>, std::unique_ptr<Internal::RPageSink>);

private:
   /// The page sink's parallel page compression scheduler if IMT is on.
   /// Needs to be destructed after the page sink (in the fill context) is destructed and so declared before.
   std::unique_ptr<Internal::RPageStorage::RTaskScheduler> fZipTasks;
   Experimental::RNTupleFillContext fFillContext;
   Experimental::Detail::RNTupleMetrics fMetrics;

   ROOT::NTupleSize_t fLastCommittedClusterGroup = 0;

   RNTupleWriter(std::unique_ptr<ROOT::RNTupleModel> model, std::unique_ptr<Internal::RPageSink> sink);

   ROOT::RNTupleModel &GetUpdatableModel();
   Internal::RPageSink &GetSink() { return *fFillContext.fSink; }

   // Helper function that is called from CommitCluster() when necessary
   void CommitClusterGroup();

   /// Create a writer, potentially wrapping the sink in a RPageSinkBuf.
   static std::unique_ptr<RNTupleWriter> Create(std::unique_ptr<ROOT::RNTupleModel> model,
                                                std::unique_ptr<Internal::RPageSink> sink,
                                                const ROOT::RNTupleWriteOptions &options);

public:
   /// Creates an RNTupleWriter backed by `storage`, overwriting it if one with the same URI exists.
   /// The format of the backing storage is determined by `storage`: in the simplest case it will be a local file, but
   /// a different backend may be selected via the URI prefix.
   ///
   /// The RNTupleWriter will create an RNTuple with the schema determined by `model` (which must not be null) and
   /// with name `ntupleName`. This same name can later be used to read back the RNTuple via RNTupleReader.
   ///
   /// \param model The RNTupleModel describing the schema of the RNTuple written by this writer
   /// \param ntupleName The name of the RNTuple to be written
   /// \param storage The URI where the RNTuple will be stored (usually just a file name or path)
   /// \param options May be passed to customize the behavior of the RNTupleWriter (see also RNTupleWriteOptions).
   ///
   /// Throws a ROOT::RException if the model is null.
   static std::unique_ptr<RNTupleWriter>
   Recreate(std::unique_ptr<ROOT::RNTupleModel> model, std::string_view ntupleName, std::string_view storage,
            const ROOT::RNTupleWriteOptions &options = ROOT::RNTupleWriteOptions());

   /// Convenience function allowing to call Recreate() with an inline-defined model.
   static std::unique_ptr<RNTupleWriter>
   Recreate(std::initializer_list<std::pair<std::string_view, std::string_view>> fields, std::string_view ntupleName,
            std::string_view storage, const ROOT::RNTupleWriteOptions &options = ROOT::RNTupleWriteOptions());

   /// Creates an RNTupleWriter that writes into an existing TFile or TDirectory, without overwriting its content.
   /// `fileOrDirectory` may be an empty TFile.
   /// \see Recreate()
   static std::unique_ptr<RNTupleWriter> Append(std::unique_ptr<ROOT::RNTupleModel> model, std::string_view ntupleName,
                                                TDirectory &fileOrDirectory,
                                                const ROOT::RNTupleWriteOptions &options = ROOT::RNTupleWriteOptions());
   RNTupleWriter(const RNTupleWriter &) = delete;
   RNTupleWriter &operator=(const RNTupleWriter &) = delete;
   ~RNTupleWriter();

   /// The simplest user interface if the default entry that comes with the ntuple model is used.
   /// \return The number of uncompressed bytes written.
   std::size_t Fill() { return fFillContext.Fill(fFillContext.fModel->GetDefaultEntry()); }
   /// Multiple entries can have been instantiated from the ntuple model.  This method will check the entry's model ID
   /// to ensure it comes from the writer's own model or throw an exception otherwise.
   /// \return The number of uncompressed bytes written.
   std::size_t Fill(ROOT::REntry &entry) { return fFillContext.Fill(entry); }
   /// Fill an entry into this ntuple, but don't commit the cluster. The calling code must pass an RNTupleFillStatus
   /// and check RNTupleFillStatus::ShouldFlushCluster.
   void FillNoFlush(ROOT::REntry &entry, RNTupleFillStatus &status) { fFillContext.FillNoFlush(entry, status); }

   /// Fill an RRawPtrWriteEntry into this ntuple.  This method will check the entry's model ID to ensure it comes from
   /// the writer's own model or throw an exception otherwise.
   /// \return The number of uncompressed bytes written.
   std::size_t Fill(Experimental::Detail::RRawPtrWriteEntry &entry) { return fFillContext.Fill(entry); }
   /// Fill an RRawPtrWriteEntry into this ntuple, but don't commit the cluster. The calling code must pass an
   /// RNTupleFillStatus and check RNTupleFillStatus::ShouldFlushCluster.
   void FillNoFlush(Experimental::Detail::RRawPtrWriteEntry &entry, RNTupleFillStatus &status)
   {
      fFillContext.FillNoFlush(entry, status);
   }

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

   std::unique_ptr<ROOT::REntry> CreateEntry() const { return fFillContext.CreateEntry(); }
   std::unique_ptr<Experimental::Detail::RRawPtrWriteEntry> CreateRawPtrWriteEntry() const
   {
      return fFillContext.CreateRawPtrWriteEntry();
   }

   /// Return the entry number that was last flushed in a cluster.
   ROOT::NTupleSize_t GetLastFlushed() const { return fFillContext.GetLastFlushed(); }
   /// Return the entry number that was last committed in a cluster.
   ROOT::NTupleSize_t GetLastCommitted() const { return fFillContext.GetLastFlushed(); }
   /// Return the entry number that was last committed in a cluster group.
   ROOT::NTupleSize_t GetLastCommittedClusterGroup() const { return fLastCommittedClusterGroup; }
   /// Return the number of entries filled so far.
   ROOT::NTupleSize_t GetNEntries() const { return fFillContext.GetNEntries(); }

   void EnableMetrics() { fMetrics.Enable(); }
   const Experimental::Detail::RNTupleMetrics &GetMetrics() const { return fMetrics; }

   const ROOT::RNTupleModel &GetModel() const { return *fFillContext.fModel; }

   /// Get a RNTupleModel::RUpdater that provides limited support for incremental updates to the underlying
   /// model, e.g. addition of new fields.
   ///
   /// **Example: add a new field after the model has been used to construct a `RNTupleWriter` object**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTuple.hxx>
   ///
   /// auto model = ROOT::RNTupleModel::Create();
   /// auto fldFloat = model->MakeField<float>("fldFloat");
   /// auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "myNTuple", "some/file.root");
   /// auto updater = writer->CreateModelUpdater();
   /// updater->BeginUpdate();
   /// updater->AddField(std::make_unique<RField<float>>("pt"));
   /// updater->CommitUpdate();
   ///
   /// // ...
   /// ~~~
   std::unique_ptr<ROOT::RNTupleModel::RUpdater> CreateModelUpdater()
   {
      return std::make_unique<ROOT::RNTupleModel::RUpdater>(*this);
   }
}; // class RNTupleWriter

} // namespace ROOT

#endif // ROOT_RNTupleWriter
