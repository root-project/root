/// \file ROOT/RNTuple.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTuple
#define ROOT7_RNTuple

#include <ROOT/RConfig.hxx> // for R__unlikely
#include <ROOT/RError.hxx>
#include <ROOT/RMiniFile.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RNTupleView.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RSpan.hxx>
#include <memory>
#include <string_view>

#include <TClassRef.h>

#include <iterator>
#include <memory>
#include <sstream>
#include <utility>

class TFile;

namespace ROOT {
namespace Experimental {

class RCollectionField;
class REntry;
class RNTuple;
class RNTupleModel;

namespace Internal {
struct RNTupleTester; // friend of RNTuple
} // namespace Internal

/**
 * Listing of the different options that can be printed by RNTupleReader::GetInfo()
 */
enum class ENTupleInfo {
   kSummary,  // The ntuple name, description, number of entries
   kStorageDetails, // size on storage, page sizes, compression factor, etc.
   kMetrics, // internals performance counters, requires that EnableMetrics() was called
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleReader
\ingroup NTuple
\brief An RNTuple that is used to read data from storage

An input ntuple provides data from storage as C++ objects. The ntuple model can be created from the data on storage
or it can be imposed by the user. The latter case allows users to read into a specialized ntuple model that covers
only a subset of the fields in the ntuple. The ntuple model is used when reading complete entries.
Individual fields can be read as well by instantiating a tree view.

~~~ {.cpp}
#include <ROOT/RNTuple.hxx>
using ROOT::Experimental::RNTupleReader;

#include <iostream>

auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
std::cout << "myNTuple has " << ntuple->GetNEntries() << " entries\n";
~~~
*/
// clang-format on
class RNTupleReader {
private:
   /// Set as the page source's scheduler for parallel page decompression if IMT is on
   /// Needs to be destructed after the pages source is destructed (an thus be declared before)
   std::unique_ptr<Internal::RPageStorage::RTaskScheduler> fUnzipTasks;

   std::unique_ptr<Internal::RPageSource> fSource;
   /// Needs to be destructed before fSource
   std::unique_ptr<RNTupleModel> fModel;
   /// We use a dedicated on-demand reader for Show() and Scan(). Printing data uses all the fields
   /// from the full model even if the analysis code uses only a subset of fields. The display reader
   /// is a clone of the original reader.
   std::unique_ptr<RNTupleReader> fDisplayReader;
   /// The ntuple descriptor in the page source is protected by a read-write lock. We don't expose that to the
   /// users of RNTupleReader::GetDescriptor().  Instead, if descriptor information is needed, we clone the
   /// descriptor.  Using the descriptor's generation number, we know if the cached descriptor is stale.
   /// Retrieving descriptor data from an RNTupleReader is supposed to be for testing and information purposes,
   /// not on a hot code path.
   std::unique_ptr<RNTupleDescriptor> fCachedDescriptor;
   Detail::RNTupleMetrics fMetrics;

   RNTupleReader(std::unique_ptr<RNTupleModel> model, std::unique_ptr<Internal::RPageSource> source);
   /// The model is generated from the ntuple metadata on storage.
   explicit RNTupleReader(std::unique_ptr<Internal::RPageSource> source);

   void ConnectModel(RNTupleModel &model);
   RNTupleReader *GetDisplayReader();
   void InitPageSource();

public:
   // Browse through the entries
   class RIterator {
   private:
      NTupleSize_t fIndex = kInvalidNTupleIndex;
   public:
      using iterator = RIterator;
      using iterator_category = std::forward_iterator_tag;
      using value_type = NTupleSize_t;
      using difference_type = NTupleSize_t;
      using pointer = NTupleSize_t*;
      using reference = NTupleSize_t&;

      RIterator() = default;
      explicit RIterator(NTupleSize_t index) : fIndex(index) {}
      ~RIterator() = default;

      iterator  operator++(int) /* postfix */        { auto r = *this; fIndex++; return r; }
      iterator& operator++()    /* prefix */         { ++fIndex; return *this; }
      reference operator* ()                         { return fIndex; }
      pointer   operator->()                         { return &fIndex; }
      bool      operator==(const iterator& rh) const { return fIndex == rh.fIndex; }
      bool      operator!=(const iterator& rh) const { return fIndex != rh.fIndex; }
   };

   /// Used to specify the underlying RNTuples in OpenFriends()
   struct ROpenSpec {
      std::string fNTupleName;
      std::string fStorage;
      RNTupleReadOptions fOptions;

      ROpenSpec() = default;
      ROpenSpec(std::string_view n, std::string_view s) : fNTupleName(n), fStorage(s) {}
   };

   /// Open an RNTuple for reading.
   ///
   /// Throws an RException if there is no RNTuple with the given name.
   ///
   /// **Example: open an RNTuple and print the number of entries**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTuple.hxx>
   /// using ROOT::Experimental::RNTupleReader;
   ///
   /// #include <iostream>
   ///
   /// auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
   /// std::cout << "myNTuple has " << ntuple->GetNEntries() << " entries\n";
   /// ~~~
   static std::unique_ptr<RNTupleReader> Open(std::string_view ntupleName,
                                              std::string_view storage,
                                              const RNTupleReadOptions &options = RNTupleReadOptions());
   static std::unique_ptr<RNTupleReader>
   Open(RNTuple *ntuple, const RNTupleReadOptions &options = RNTupleReadOptions());
   /// The caller imposes a model, which must be compatible with the model found in the data on storage.
   static std::unique_ptr<RNTupleReader> Open(std::unique_ptr<RNTupleModel> model, std::string_view ntupleName,
                                              std::string_view storage,
                                              const RNTupleReadOptions &options = RNTupleReadOptions());
   static std::unique_ptr<RNTupleReader>
   Open(std::unique_ptr<RNTupleModel> model, RNTuple *ntuple, const RNTupleReadOptions &options = RNTupleReadOptions());
   /// Open RNTuples as one virtual, horizontally combined ntuple.  The underlying RNTuples must
   /// have an identical number of entries.  Fields in the combined RNTuple are named with the ntuple name
   /// as a prefix, e.g. myNTuple1.px and myNTuple2.pt (see tutorial ntpl006_friends)
   static std::unique_ptr<RNTupleReader> OpenFriends(std::span<ROpenSpec> ntuples);
   std::unique_ptr<RNTupleReader> Clone()
   {
      return std::unique_ptr<RNTupleReader>(new RNTupleReader(fSource->Clone()));
   }
   ~RNTupleReader();

   NTupleSize_t GetNEntries() const { return fSource->GetNEntries(); }
   const RNTupleModel &GetModel();

   /// Returns a cached copy of the page source descriptor. The returned pointer remains valid until the next call
   /// to LoadEntry or to any of the views returned from the reader.
   const RNTupleDescriptor &GetDescriptor();

   /// Prints a detailed summary of the ntuple, including a list of fields.
   ///
   /// **Example: print summary information to stdout**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTuple.hxx>
   /// using ROOT::Experimental::ENTupleInfo;
   /// using ROOT::Experimental::RNTupleReader;
   ///
   /// #include <iostream>
   ///
   /// auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
   /// ntuple->PrintInfo();
   /// // or, equivalently:
   /// ntuple->PrintInfo(ENTupleInfo::kSummary, std::cout);
   /// ~~~
   /// **Example: print detailed column storage data to stderr**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTuple.hxx>
   /// using ROOT::Experimental::ENTupleInfo;
   /// using ROOT::Experimental::RNTupleReader;
   ///
   /// #include <iostream>
   ///
   /// auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
   /// ntuple->PrintInfo(ENTupleInfo::kStorageDetails, std::cerr);
   /// ~~~
   ///
   /// For use of ENTupleInfo::kMetrics, see #EnableMetrics.
   void PrintInfo(const ENTupleInfo what = ENTupleInfo::kSummary, std::ostream &output = std::cout);

   /// Shows the values of the i-th entry/row, starting with 0 for the first entry. By default,
   /// prints the output in JSON format.
   /// Uses the visitor pattern to traverse through each field of the given entry.
   void Show(NTupleSize_t index, std::ostream &output = std::cout);

   /// Analogous to Fill(), fills the default entry of the model. Returns false at the end of the ntuple.
   /// On I/O errors, raises an exception.
   void LoadEntry(NTupleSize_t index) {
      // TODO(jblomer): can be templated depending on the factory method / constructor
      if (R__unlikely(!fModel)) {
         fModel = fSource->GetSharedDescriptorGuard()->CreateModel();
         ConnectModel(*fModel);
      }
      LoadEntry(index, fModel->GetDefaultEntry());
   }
   /// Fills a user provided entry after checking that the entry has been instantiated from the ntuple model
   void LoadEntry(NTupleSize_t index, REntry &entry) { entry.Read(index); }

   /// Returns an iterator over the entry indices of the RNTuple.
   ///
   /// **Example: iterate over all entries and print each entry in JSON format**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTuple.hxx>
   /// using ROOT::Experimental::ENTupleShowFormat;
   /// using ROOT::Experimental::RNTupleReader;
   ///
   /// #include <iostream>
   ///
   /// auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
   /// for (auto i : ntuple->GetEntryRange()) {
   ///    ntuple->Show(i);
   /// }
   /// ~~~
   RNTupleGlobalRange GetEntryRange() { return RNTupleGlobalRange(0, GetNEntries()); }

   /// Provides access to an individual field that can contain either a scalar value or a collection, e.g.
   /// GetView<double>("particles.pt") or GetView<std::vector<double>>("particle").  It can as well be the index
   /// field of a collection itself, like GetView<NTupleSize_t>("particle").
   ///
   /// Raises an exception if there is no field with the given name.
   ///
   /// **Example: iterate over a field named "pt" of type `float`**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTuple.hxx>
   /// using ROOT::Experimental::RNTupleReader;
   ///
   /// #include <iostream>
   ///
   /// auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
   /// auto pt = ntuple->GetView<float>("pt");
   ///
   /// for (auto i : ntuple->GetEntryRange()) {
   ///    std::cout << i << ": " << pt(i) << "\n";
   /// }
   /// ~~~
   template <typename T>
   RNTupleView<T> GetView(std::string_view fieldName) {
      auto fieldId = fSource->GetSharedDescriptorGuard()->FindFieldId(fieldName);
      if (fieldId == kInvalidDescriptorId) {
         throw RException(R__FAIL("no field named '" + std::string(fieldName) + "' in RNTuple '" +
                                  fSource->GetSharedDescriptorGuard()->GetName() + "'"));
      }
      return GetView<T>(fieldId);
   }

   template <typename T>
   RNTupleView<T> GetView(DescriptorId_t fieldId)
   {
      return RNTupleView<T>(fieldId, fSource.get());
   }

   /// Raises an exception if:
   /// * there is no field with the given name or,
   /// * the field is not a collection
   RNTupleCollectionView GetCollectionView(std::string_view fieldName)
   {
      auto fieldId = fSource->GetSharedDescriptorGuard()->FindFieldId(fieldName);
      if (fieldId == kInvalidDescriptorId) {
         throw RException(R__FAIL("no field named '" + std::string(fieldName) + "' in RNTuple '" +
                                  fSource->GetSharedDescriptorGuard()->GetName() + "'"));
      }
      return GetCollectionView(fieldId);
   }

   RNTupleCollectionView GetCollectionView(DescriptorId_t fieldId)
   {
      return RNTupleCollectionView(fieldId, fSource.get());
   }

   RIterator begin() { return RIterator(0); }
   RIterator end() { return RIterator(GetNEntries()); }

   /// Enable performance measurements (decompression time, bytes read from storage, etc.)
   ///
   /// **Example: inspect the reader metrics after loading every entry**
   /// ~~~ {.cpp}
   /// #include <ROOT/RNTuple.hxx>
   /// using ROOT::Experimental::ENTupleInfo;
   /// using ROOT::Experimental::RNTupleReader;
   ///
   /// #include <iostream>
   ///
   /// auto ntuple = RNTupleReader::Open("myNTuple", "some/file.root");
   /// // metrics must be turned on beforehand
   /// ntuple->EnableMetrics();
   ///
   /// for (auto i : ntuple->GetEntryRange()) {
   ///    ntuple->LoadEntry(i);
   /// }
   /// ntuple->PrintInfo(ENTupleInfo::kMetrics);
   /// ~~~
   void EnableMetrics() { fMetrics.Enable(); }
   const Detail::RNTupleMetrics &GetMetrics() const { return fMetrics; }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleFillContext
\ingroup NTuple
\brief A context for filling entries (data) into clusters of an RNTuple

An output cluster can be filled with entries. The caller has to make sure that the data that gets filled into a cluster
is not modified for the time of the Fill() call. The fill call serializes the C++ object into the column format and
writes data into the corresponding column page buffers.  Writing of the buffers to storage is deferred and can be
triggered by CommitCluster() or by destructing the context.  On I/O errors, an exception is thrown.

Instances of this class are not meant to be used in isolation and can be created from an RNTupleParallelWriter. For
sequential writing, please refer to RNTupleWriter.
*/
// clang-format on
class RNTupleFillContext {
   friend class RNTupleWriter;
   friend class RNTupleParallelWriter;

private:
   std::unique_ptr<Internal::RPageSink> fSink;
   /// Needs to be destructed before fSink
   std::unique_ptr<RNTupleModel> fModel;

   Detail::RNTupleMetrics fMetrics;

   NTupleSize_t fLastCommitted = 0;
   NTupleSize_t fNEntries = 0;
   /// Keeps track of the number of bytes written into the current cluster
   std::size_t fUnzippedClusterSize = 0;
   /// The total number of bytes written to storage (i.e., after compression)
   std::uint64_t fNBytesCommitted = 0;
   /// The total number of bytes filled into all the so far committed clusters,
   /// i.e. the uncompressed size of the written clusters
   std::uint64_t fNBytesFilled = 0;
   /// Limit for committing cluster no matter the other tunables
   std::size_t fMaxUnzippedClusterSize;
   /// Estimator of uncompressed cluster size, taking into account the estimated compression ratio
   std::size_t fUnzippedClusterSizeEst;

   RNTupleFillContext(std::unique_ptr<RNTupleModel> model, std::unique_ptr<Internal::RPageSink> sink);
   RNTupleFillContext(const RNTupleFillContext &) = delete;
   RNTupleFillContext &operator=(const RNTupleFillContext &) = delete;

public:
   ~RNTupleFillContext();

   /// Fill an entry into this context.  This method will perform a light check whether the entry comes from the
   /// context's own model.
   /// \return The number of uncompressed bytes written.
   std::size_t Fill(REntry &entry)
   {
      if (R__unlikely(entry.GetModelId() != fModel->GetModelId()))
         throw RException(R__FAIL("mismatch between entry and model"));

      const std::size_t bytesWritten = entry.Append();
      fUnzippedClusterSize += bytesWritten;
      fNEntries++;
      if ((fUnzippedClusterSize >= fMaxUnzippedClusterSize) || (fUnzippedClusterSize >= fUnzippedClusterSizeEst))
         CommitCluster();
      return bytesWritten;
   }
   /// Ensure that the data from the so far seen Fill calls has been written to storage
   void CommitCluster();

   std::unique_ptr<REntry> CreateEntry() { return fModel->CreateEntry(); }

   /// Return the entry number that was last committed in a cluster.
   NTupleSize_t GetLastCommitted() const { return fLastCommitted; }
   /// Return the number of entries filled so far.
   NTupleSize_t GetNEntries() const { return fNEntries; }

   void EnableMetrics() { fMetrics.Enable(); }
   const Detail::RNTupleMetrics &GetMetrics() const { return fMetrics; }
};

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
triggered by CommitCluster() or by destructing the writer.  On I/O errors, an exception is thrown.
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

   NTupleSize_t fLastCommittedClusterGroup = 0;

   RNTupleWriter(std::unique_ptr<RNTupleModel> model, std::unique_ptr<Internal::RPageSink> sink);

   RNTupleModel &GetUpdatableModel() { return *fFillContext.fModel; }
   Internal::RPageSink &GetSink() { return *fFillContext.fSink; }

   // Helper function that is called from CommitCluster() when necessary
   void CommitClusterGroup();

   /// Create a writer, potentially wrapping the sink in a RPageSinkBuf.
   static std::unique_ptr<RNTupleWriter> Create(std::unique_ptr<RNTupleModel> model,
                                                std::unique_ptr<Internal::RPageSink> sink,
                                                const RNTupleWriteOptions &options);

public:
   /// Throws an exception if the model is null.
   static std::unique_ptr<RNTupleWriter> Recreate(std::unique_ptr<RNTupleModel> model,
                                                  std::string_view ntupleName,
                                                  std::string_view storage,
                                                  const RNTupleWriteOptions &options = RNTupleWriteOptions());
   static std::unique_ptr<RNTupleWriter>
   Recreate(std::initializer_list<std::pair<std::string_view, std::string_view>> fields, std::string_view ntupleName,
            std::string_view storage, const RNTupleWriteOptions &options = RNTupleWriteOptions());
   /// Throws an exception if the model is null.
   static std::unique_ptr<RNTupleWriter> Append(std::unique_ptr<RNTupleModel> model, std::string_view ntupleName,
                                                TFile &file,
                                                const RNTupleWriteOptions &options = RNTupleWriteOptions());
   RNTupleWriter(const RNTupleWriter&) = delete;
   RNTupleWriter& operator=(const RNTupleWriter&) = delete;
   ~RNTupleWriter();

   /// The simplest user interface if the default entry that comes with the ntuple model is used.
   /// \return The number of uncompressed bytes written.
   std::size_t Fill() { return fFillContext.Fill(fFillContext.fModel->GetDefaultEntry()); }
   /// Multiple entries can have been instantiated from the ntuple model.  This method will perform
   /// a light check whether the entry comes from the ntuple's own model.
   /// \return The number of uncompressed bytes written.
   std::size_t Fill(REntry &entry) { return fFillContext.Fill(entry); }
   /// Ensure that the data from the so far seen Fill calls has been written to storage
   void CommitCluster(bool commitClusterGroup = false)
   {
      fFillContext.CommitCluster();
      if (commitClusterGroup)
         CommitClusterGroup();
   }

   std::unique_ptr<REntry> CreateEntry() { return fFillContext.CreateEntry(); }

   /// Return the entry number that was last committed in a cluster.
   NTupleSize_t GetLastCommitted() const { return fFillContext.GetLastCommitted(); }
   /// Return the entry number that was last committed in a cluster group.
   NTupleSize_t GetLastCommittedClusterGroup() const { return fLastCommittedClusterGroup; }
   /// Return the number of entries filled so far.
   NTupleSize_t GetNEntries() const { return fFillContext.GetNEntries(); }

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

// clang-format off
/**
\class ROOT::Experimental::RCollectionNTuple
\ingroup NTuple
\brief A virtual ntuple used for writing untyped collections that can be used to some extent like an RNTupleWriter
*
* This class is between a field and a ntuple.  It carries the offset column for the collection and the default entry
* taken from the collection model.  It does not, however, own an ntuple model because the collection model has been
* merged into the larger ntuple model.
*/
// clang-format on
class RCollectionNTupleWriter {
   friend class RCollectionField;

private:
   std::size_t fBytesWritten = 0;
   ClusterSize_t fOffset;
   std::unique_ptr<REntry> fDefaultEntry;
public:
   explicit RCollectionNTupleWriter(std::unique_ptr<REntry> defaultEntry);
   RCollectionNTupleWriter(const RCollectionNTupleWriter&) = delete;
   RCollectionNTupleWriter& operator=(const RCollectionNTupleWriter&) = delete;
   ~RCollectionNTupleWriter() = default;

   std::size_t Fill() { return Fill(*fDefaultEntry); }
   std::size_t Fill(REntry &entry)
   {
      const std::size_t bytesWritten = entry.Append();
      fBytesWritten += bytesWritten;
      fOffset++;
      return bytesWritten;
   }

   ClusterSize_t *GetOffsetPtr() { return &fOffset; }
};

} // namespace Experimental
} // namespace ROOT

#endif
