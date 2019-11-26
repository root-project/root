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

#include <ROOT/RPageStorage.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RStringView.hxx>

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTuple
\ingroup NTuple
\brief Entry point for an RNTuple in a ROOT file

The class points to the header and footer keys, which in turn have the references to the pages.
Only the RNTuple key will be listed in the list of keys. Like TBaskets, the pages are "invisible" keys.
Byte offset references in the RNTuple header and footer reference directly the data part of page records,
skipping the TFile key part.
*/
// clang-format on
struct RNTuple {
   /// The file offset of the key containing the ntuple header
   std::uint64_t fSeekHeader = 0;
   /// The size of the ntuple header including the TKey
   std::uint32_t fNBytesHeader = 0;
   /// The file offset of the key containing the ntuple footer
   std::uint64_t fSeekFooter = 0;
   /// The size of the ntuple footer including the TKey
   std::uint32_t fNBytesFooter = 0;
   /// Currently unused, reserved for later use
   std::uint64_t fReserved = 0;
};


namespace Internal {
/// Holds references to an open ROOT file during writing
struct RTFileControlBlock;
} // namespace Internal


namespace Detail {

class RPageAllocatorHeap;
class RPagePool;
class RRawFile;


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSinkFile
\ingroup NTuple
\brief Storage provider that write ntuple pages into a file

The written file can be either in ROOT format or in raw format.
*/
// clang-format on
class RPageSinkFile : public RPageSink {
private:
   static constexpr std::size_t kDefaultElementsPerPage = 10000;
   /// Cannot process data blocks larger than 1MB
   static constexpr std::size_t kMaxRecordSize = 1024 * 1024;

   RNTupleMetrics fMetrics;
   std::unique_ptr<RPageAllocatorHeap> fPageAllocator;

   FILE *fFile = nullptr;
   /// Byte offset of the next write (current file size)
   std::uint64_t fFilePos = 0;
   /// Byte offset of the begining of the currently open cluster
   std::uint64_t fClusterStart = 0;
   /// The file name without the parent directory (e.g. "events.root")
   std::string fFileName;
   /// Scratch space for compression of keys / pages
   std::unique_ptr<std::array<char, kMaxRecordSize>> fZipBuffer;
   /// Keeps track of TFile control structures, which need to be updated on committing the data set
   std::unique_ptr<ROOT::Experimental::Internal::RTFileControlBlock> fControlBlock;

   /// Writes bytes in the open fFile, either at fFilePos or at the given offset
   void Write(void *from, size_t size, std::int64_t offset = -1);
   /// Writes a TKey including the data record, given by buffer, into fFile; returns the file offset to the payload
   std::uint64_t WriteKey(void *buffer, std::size_t nbytes, std::int64_t offset = -1,
                          std::uint64_t directoryOffset = 100, int compression = 0,
                          const std::string &className = "",
                          const std::string &objectName = "",
                          const std::string &title = "");

protected:
   void DoCreate(const RNTupleModel &model) final;
   RClusterDescriptor::RLocator DoCommitPage(ColumnHandle_t columnHandle, const RPage &page) final;
   RClusterDescriptor::RLocator DoCommitCluster(NTupleSize_t nEntries) final;
   void DoCommitDataset() final;

public:
   RPageSinkFile(std::string_view ntupleName, std::string_view path, const RNTupleWriteOptions &options);
   virtual ~RPageSinkFile();

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements = 0) final;
   void ReleasePage(RPage &page) final;

   RNTupleMetrics &GetMetrics() final { return fMetrics; }
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageAllocatorFile
\ingroup NTuple
\brief Manages pages read from a raw file
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
public:
   /// Cannot process pages larger than 1MB
   static constexpr std::size_t kMaxPageSize = 1024 * 1024;

private:
   RNTupleMetrics fMetrics;
   /// Populated pages might be shared; there memory buffer is managed by the RPageAllocatorFile
   std::unique_ptr<RPageAllocatorKey> fPageAllocator;
   /// The page pool migh, at some point, be used by multiple page sources
   std::shared_ptr<RPagePool> fPagePool;
   /// A scatch space to uncompress keys / buffers
   std::unique_ptr<std::array<unsigned char, kMaxPageSize>> fUnzipBuffer;
   /// An RRawFile is used to request the necessary byte ranges from a local or a remote file
   std::unique_ptr<RRawFile> fFile;

   RPageSourceFile(std::string_view ntupleName, const RNTupleReadOptions &options);
   void Read(void *buffer, std::size_t nbytes, std::uint64_t offset);
   RPage PopulatePageFromCluster(ColumnHandle_t columnHandle, const RClusterDescriptor &clusterDescriptor,
                                 ClusterSize_t::ValueType clusterIndex);

protected:
   RNTupleDescriptor DoAttach() final;

public:
   RPageSourceFile(std::string_view ntupleName, std::string_view path, const RNTupleReadOptions &options);
   std::unique_ptr<RPageSource> Clone() const final;
   virtual ~RPageSourceFile();

   RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex) final;
   RPage PopulatePage(ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex) final;
   void ReleasePage(RPage &page) final;

   RNTupleMetrics &GetMetrics() final { return fMetrics; }
};


} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
