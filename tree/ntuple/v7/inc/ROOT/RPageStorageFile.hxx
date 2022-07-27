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
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RRawFile.hxx>
#include <ROOT/RStringView.hxx>

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#define MY_CODE_RPAGE_STORAGE_FILE

class TFile;

namespace ROOT {

namespace Internal {
class RRawFile;
}

namespace Experimental {
class RNTuple; // for making RPageSourceFile a friend of RNTuple

namespace Detail {

class RClusterPool;
class RPageAllocatorHeap;
class RPagePool;


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSinkFile
\ingroup NTuple
\brief Storage provider that write ntuple pages into a file

The written file can be either in ROOT format or in RNTuple bare format.
*/
// clang-format on
class RPageSinkFile : public RPageSink {
private:
   std::unique_ptr<RPageAllocatorHeap> fPageAllocator;

   std::unique_ptr<Internal::RNTupleFileWriter> fWriter;
   /// Number of bytes committed to storage in the current cluster
   std::uint64_t fNBytesCurrentCluster = 0;
   RPageSinkFile(std::string_view ntupleName, const RNTupleWriteOptions &options);

   RNTupleLocator WriteSealedPage(const RPageStorage::RSealedPage &sealedPage,
                                                std::size_t bytesPacked);

protected:
   void CreateImpl(const RNTupleModel &model, unsigned char *serializedHeader, std::uint32_t length) final;
   RNTupleLocator CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page) final;
   RNTupleLocator CommitSealedPageImpl(DescriptorId_t columnId, const RPageStorage::RSealedPage &sealedPage) final;
   std::uint64_t CommitClusterImpl(NTupleSize_t nEntries) final;
   RNTupleLocator CommitClusterGroupImpl(unsigned char *serializedPageList, std::uint32_t length) final;
   void CommitDatasetImpl(unsigned char *serializedFooter, std::uint32_t length) final;

public:
   RPageSinkFile(std::string_view ntupleName, std::string_view path, const RNTupleWriteOptions &options);
   RPageSinkFile(std::string_view ntupleName, std::string_view path, const RNTupleWriteOptions &options,
                 std::unique_ptr<TFile> &file);
   RPageSinkFile(std::string_view ntupleName, TFile &file, const RNTupleWriteOptions &options);
   RPageSinkFile(const RPageSinkFile&) = delete;
   RPageSinkFile& operator=(const RPageSinkFile&) = delete;
   RPageSinkFile(RPageSinkFile&&) = default;
   RPageSinkFile& operator=(RPageSinkFile&&) = default;
   virtual ~RPageSinkFile();

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements) final;
   void ReleasePage(RPage &page) final;

#ifdef MY_CODE_RPAGE_STORAGE_FILE
   void ZeroCopy( std::string_view ntupleName, std::string_view location );
   void ZeroCopyMerge( std::string_view ntupleNameSrc1, std::string_view locationSrc1, std::string_view ntupleNameSrc2, std::string_view locationSrc2 );
#endif
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageAllocatorFile
\ingroup NTuple
\brief Manages pages read from a the file
*/
// clang-format on
class RPageAllocatorFile {
public:
   static RPage NewPage(ColumnId_t columnId, void *mem, std::size_t elementSize, std::size_t nElements);
   static void DeletePage(const RPage& page);
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSourceFile
\ingroup NTuple
\brief Storage provider that reads ntuple pages from a file
*/
// clang-format on
class RPageSourceFile : public RPageSource {
   friend class ROOT::Experimental::RNTuple;

private:
   /// Summarizes cluster-level information that are necessary to populate a certain page.
   /// Used by PopulatePageFromCluster().
   struct RClusterInfo {
      DescriptorId_t fClusterId = 0;
      /// Location of the page on disk
      RClusterDescriptor::RPageRange::RPageInfoExtended fPageInfo;
      /// The first element number of the page's column in the given cluster
      std::uint64_t fColumnOffset = 0;
   };

   /// Populated pages might be shared; there memory buffer is managed by the RPageAllocatorFile
   std::unique_ptr<RPageAllocatorFile> fPageAllocator;
   /// The page pool might, at some point, be used by multiple page sources
   std::shared_ptr<RPagePool> fPagePool;
   /// The last cluster from which a page got populated.  Points into fClusterPool->fPool
   RCluster *fCurrentCluster = nullptr;
   /// An RRawFile is used to request the necessary byte ranges from a local or a remote file
   std::unique_ptr<ROOT::Internal::RRawFile> fFile;
   /// Takes the fFile to read ntuple blobs from it
   Internal::RMiniFileReader fReader;
   /// The descriptor is created from the header and footer either in AttachImpl or in CreateFromAnchor
   RNTupleDescriptorBuilder fDescriptorBuilder;
   /// The cluster pool asynchronously preloads the next few clusters
   std::unique_ptr<RClusterPool> fClusterPool;

   /// Deserialized header and footer into a minimal descriptor held by fDescriptorBuilder
   void InitDescriptor(const Internal::RFileNTupleAnchor &anchor);

   RPageSourceFile(std::string_view ntupleName, const RNTupleReadOptions &options);
   /// Used from the RNTuple class to build a datasource if the anchor is already available
   static std::unique_ptr<RPageSourceFile> CreateFromAnchor(const Internal::RFileNTupleAnchor &anchor,
                                                            std::string_view path, const RNTupleReadOptions &options);
   RPage PopulatePageFromCluster(ColumnHandle_t columnHandle, const RClusterInfo &clusterInfo,
                                 ClusterSize_t::ValueType idxInCluster);

   /// Helper function for LoadClusters: it prepares the memory buffer (page map) and the
   /// read requests for a given cluster and columns.  The reead requests are appended to
   /// the provided vector.  This way, requests can be collected for multiple clusters before
   /// sending them to RRawFile::ReadV().
   std::unique_ptr<RCluster> PrepareSingleCluster(
      const RCluster::RKey &clusterKey,
      std::vector<ROOT::Internal::RRawFile::RIOVec> &readRequests);


protected:
   RNTupleDescriptor AttachImpl() final;
   void UnzipClusterImpl(RCluster *cluster) final;

public:

#ifdef MY_CODE_RPAGE_STORAGE_FILE
   ROOT::Experimental::Internal::RFileNTupleAnchor GetAnchor();
#endif

   RPageSourceFile(std::string_view ntupleName, std::string_view path, const RNTupleReadOptions &options);
   /// The cloned page source creates a new raw file and reader and opens its own file descriptor to the data.
   /// The meta-data (header and footer) is reread and parsed by the clone.
   std::unique_ptr<RPageSource> Clone() const final;

   RPageSourceFile(const RPageSourceFile&) = delete;
   RPageSourceFile& operator=(const RPageSourceFile&) = delete;
   RPageSourceFile(RPageSourceFile &&) = delete;
   RPageSourceFile &operator=(RPageSourceFile &&) = delete;
   virtual ~RPageSourceFile();

   RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex) final;
   RPage PopulatePage(ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex) final;
   void ReleasePage(RPage &page) final;

   void LoadSealedPage(DescriptorId_t columnId, const RClusterIndex &clusterIndex,
                       RSealedPage &sealedPage) final;

   std::vector<std::unique_ptr<RCluster>> LoadClusters(std::span<RCluster::RKey> clusterKeys) final;
};


} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
