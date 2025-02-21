/// \file ROOT/RPageStorageFile.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-11-21
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageStorageFile
#define ROOT7_RPageStorageFile

#include <ROOT/RMiniFile.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RRawFile.hxx>
#include <string_view>

#include <array>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <utility>

class TDirectory;

namespace ROOT {
class RNTuple; // for making RPageSourceFile a friend of RNTuple
class RNTupleLocator;

namespace Internal {
class RRawFile;
}

namespace Experimental {

namespace Internal {
class RClusterPool;
class RPageAllocatorHeap;

// clang-format off
/**
\class ROOT::Experimental::Internal::RPageSinkFile
\ingroup NTuple
\brief Storage provider that write ntuple pages into a file

The written file can be either in ROOT format or in RNTuple bare format.
*/
// clang-format on
class RPageSinkFile : public RPagePersistentSink {
private:
   // A set of pages to be committed together in a vector write.
   // Currently we assume they're all sequential (although they may span multiple ranges).
   struct CommitBatch {
      /// The list of pages to commit
      std::vector<const RSealedPage *> fSealedPages;
      /// Total size in bytes of the batch
      size_t fSize;
      /// Total uncompressed size of the elements in the page batch
      size_t fBytesPacked;
   };

   std::unique_ptr<RNTupleFileWriter> fWriter;
   /// Number of bytes committed to storage in the current cluster
   std::uint64_t fNBytesCurrentCluster = 0;
   RPageSinkFile(std::string_view ntupleName, const ROOT::RNTupleWriteOptions &options);

   /// We pass bytesPacked so that TFile::ls() reports a reasonable value for the compression ratio of the corresponding
   /// key. It is not strictly necessary to write and read the sealed page.
   RNTupleLocator WriteSealedPage(const RPageStorage::RSealedPage &sealedPage, std::size_t bytesPacked);

   /// Subroutine of CommitSealedPageVImpl, used to perform a vector write of the (multi-)range of pages
   /// contained in `batch`. The locators for the written pages are appended to `locators`.
   /// This procedure also updates some internal metrics of the page sink, hence it's not const.
   /// `batch` gets reset to size 0 after the writing is done (but its begin and end are not updated).
   void CommitBatchOfPages(CommitBatch &batch, std::vector<RNTupleLocator> &locators);

protected:
   using RPagePersistentSink::InitImpl;
   void InitImpl(unsigned char *serializedHeader, std::uint32_t length) final;
   RNTupleLocator CommitPageImpl(ColumnHandle_t columnHandle, const ROOT::Internal::RPage &page) override;
   RNTupleLocator
   CommitSealedPageImpl(ROOT::DescriptorId_t physicalColumnId, const RPageStorage::RSealedPage &sealedPage) final;
   std::vector<RNTupleLocator>
   CommitSealedPageVImpl(std::span<RPageStorage::RSealedPageGroup> ranges, const std::vector<bool> &mask) final;
   std::uint64_t StageClusterImpl() final;
   RNTupleLocator CommitClusterGroupImpl(unsigned char *serializedPageList, std::uint32_t length) final;
   using RPagePersistentSink::CommitDatasetImpl;
   void CommitDatasetImpl(unsigned char *serializedFooter, std::uint32_t length) final;

public:
   RPageSinkFile(std::string_view ntupleName, std::string_view path, const ROOT::RNTupleWriteOptions &options);
   RPageSinkFile(std::string_view ntupleName, TDirectory &fileOrDirectory, const ROOT::RNTupleWriteOptions &options);
   RPageSinkFile(const RPageSinkFile &) = delete;
   RPageSinkFile &operator=(const RPageSinkFile &) = delete;
   RPageSinkFile(RPageSinkFile &&) = default;
   RPageSinkFile &operator=(RPageSinkFile &&) = default;
   ~RPageSinkFile() override;
}; // class RPageSinkFile

// clang-format off
/**
\class ROOT::Experimental::Internal::RPageSourceFile
\ingroup NTuple
\brief Storage provider that reads ntuple pages from a file
*/
// clang-format on
class RPageSourceFile : public RPageSource {
   friend class ROOT::RNTuple;

private:
   /// Holds the uncompressed header and footer
   struct RStructureBuffer {
      std::unique_ptr<unsigned char[]> fBuffer; ///< single buffer for both header and footer
      void *fPtrHeader = nullptr;               ///< either nullptr or points into fBuffer
      void *fPtrFooter = nullptr;               ///< either nullptr or points into fBuffer

      /// Called at the end of Attach(), i.e. when the header and footer are processed
      void Reset()
      {
         RStructureBuffer empty;
         std::swap(empty, *this);
      }
   };

   /// Either provided by CreateFromAnchor, or read from the ROOT file given the ntuple name
   std::optional<RNTuple> fAnchor;
   /// The last cluster from which a page got loaded.  Points into fClusterPool->fPool
   RCluster *fCurrentCluster = nullptr;
   /// An RRawFile is used to request the necessary byte ranges from a local or a remote file
   std::unique_ptr<ROOT::Internal::RRawFile> fFile;
   /// Takes the fFile to read ntuple blobs from it
   RMiniFileReader fReader;
   /// The descriptor is created from the header and footer either in AttachImpl or in CreateFromAnchor
   RNTupleDescriptorBuilder fDescriptorBuilder;
   /// The cluster pool asynchronously preloads the next few clusters
   std::unique_ptr<RClusterPool> fClusterPool;
   /// Populated by LoadStructureImpl(), reset at the end of Attach()
   RStructureBuffer fStructureBuffer;

   RPageSourceFile(std::string_view ntupleName, const ROOT::RNTupleReadOptions &options);

   /// Helper function for LoadClusters: it prepares the memory buffer (page map) and the
   /// read requests for a given cluster and columns.  The reead requests are appended to
   /// the provided vector.  This way, requests can be collected for multiple clusters before
   /// sending them to RRawFile::ReadV().
   std::unique_ptr<RCluster>
   PrepareSingleCluster(const RCluster::RKey &clusterKey, std::vector<ROOT::Internal::RRawFile::RIOVec> &readRequests);

protected:
   void LoadStructureImpl() final;
   RNTupleDescriptor AttachImpl(RNTupleSerializer::EDescriptorDeserializeMode mode) final;
   /// The cloned page source creates a new raw file and reader and opens its own file descriptor to the data.
   std::unique_ptr<RPageSource> CloneImpl() const final;

   ROOT::Internal::RPageRef
   LoadPageImpl(ColumnHandle_t columnHandle, const RClusterInfo &clusterInfo, ROOT::NTupleSize_t idxInCluster) final;

public:
   RPageSourceFile(std::string_view ntupleName, std::string_view path, const ROOT::RNTupleReadOptions &options);
   RPageSourceFile(std::string_view ntupleName, std::unique_ptr<ROOT::Internal::RRawFile> file,
                   const ROOT::RNTupleReadOptions &options);
   /// Used from the RNTuple class to build a datasource if the anchor is already available.
   /// Requires the RNTuple object to be streamed from a file.
   static std::unique_ptr<RPageSourceFile>
   CreateFromAnchor(const RNTuple &anchor, const ROOT::RNTupleReadOptions &options = ROOT::RNTupleReadOptions());

   RPageSourceFile(const RPageSourceFile &) = delete;
   RPageSourceFile &operator=(const RPageSourceFile &) = delete;
   RPageSourceFile(RPageSourceFile &&) = delete;
   RPageSourceFile &operator=(RPageSourceFile &&) = delete;
   ~RPageSourceFile() override;

   void
   LoadSealedPage(ROOT::DescriptorId_t physicalColumnId, RNTupleLocalIndex localIndex, RSealedPage &sealedPage) final;

   std::vector<std::unique_ptr<RCluster>> LoadClusters(std::span<RCluster::RKey> clusterKeys) final;
}; // class RPageSourceFile

} // namespace Internal

} // namespace Experimental
} // namespace ROOT

#endif
