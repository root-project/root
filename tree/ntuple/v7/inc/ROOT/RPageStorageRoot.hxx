/// \file ROOT/RPageStorage.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-07-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageStorageRoot
#define ROOT7_RPageStorageRoot

#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RColumnModel.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <TDirectory.h>
#include <TFile.h>

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
\brief Entry point for an RNTuple in a root file

The class points to the header and footer, which in turn have the references to the pages.
Only the RNTuple key will be listed in the list of keys. Like TBaskets, the pages are "invisible" keys.
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

struct RNTupleBlob {
   RNTupleBlob() {}
   RNTupleBlob(int size, unsigned char *content) : fSize(size), fContent(content) {}
   RNTupleBlob(const RNTupleBlob &other) = delete;
   RNTupleBlob &operator =(const RNTupleBlob &other) = delete;
   ~RNTupleBlob() = default;

   std::int32_t fVersion = 0;
   int fSize = 0;
   unsigned char* fContent = nullptr; //[fSize]
};

struct RTFileControlBlock;

} // namespace Internal


namespace Detail {

class RPagePool;

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSinkRoot
\ingroup NTuple
\brief Storage provider that write ntuple pages into a ROOT TFile
*/
// clang-format on
class RPageSinkRoot : public RPageSink {
private:
   static constexpr std::size_t kDefaultElementsPerPage = 10000;
   /// Cannot process data blocks larger than 1MB
   static constexpr std::size_t kMaxRecordSize = 1024 * 1024;

   RNTupleMetrics fMetrics;
   std::unique_ptr<RPageAllocatorHeap> fPageAllocator;

   FILE *fBinaryFile = nullptr;
   std::uint64_t fFilePos = 0;
   std::string fFileName;
   std::unique_ptr<std::array<char, kMaxRecordSize>> fZipBuffer;
   std::unique_ptr<ROOT::Experimental::Internal::RTFileControlBlock> fControlBlock;
   std::unique_ptr<TFile> fFile;
   TDirectory *fDirectory = nullptr;


   /// Instead of a physical file offset, pages in root are identified by an index which becomes part of the key
   DescriptorId_t fLastPageIdx = 0;

   void Write(void *from, size_t size, std::int64_t offset = -1);
   void WriteKey(void *buffer, std::size_t nbytes, std::int64_t offset = -1,
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
   RPageSinkRoot(std::string_view ntupleName, std::string_view path, const RNTupleWriteOptions &options);
   virtual ~RPageSinkRoot();

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements = 0) final;
   void ReleasePage(RPage &page) final;

   RNTupleMetrics &GetMetrics() final { return fMetrics; }
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageAllocatorKey
\ingroup NTuple
\brief Adopts the memory returned by TKey->ReadObject()
*/
// clang-format on
class RPageAllocatorKey {
public:
   static RPage NewPage(ColumnId_t columnId, void *mem, std::size_t elementSize, std::size_t nElements);
   static void DeletePage(const RPage& page, ROOT::Experimental::Internal::RNTupleBlob *payload);
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSourceRoot
\ingroup NTuple
\brief Storage provider that reads ntuple pages from a ROOT TFile
*/
// clang-format on
class RPageSourceRoot : public RPageSource {
private:
   RNTupleMetrics fMetrics;
   std::unique_ptr<RPageAllocatorKey> fPageAllocator;
   std::shared_ptr<RPagePool> fPagePool;

   /// Currently, an ntuple is stored as a directory in a TFile
   std::unique_ptr<TFile> fFile;
   TDirectory *fDirectory = nullptr;

   RPage PopulatePageFromCluster(ColumnHandle_t columnHandle, const RClusterDescriptor &clusterDescriptor,
                                 ClusterSize_t::ValueType clusterIndex);

protected:
   RNTupleDescriptor DoAttach() final;

public:
   RPageSourceRoot(std::string_view ntupleName, std::string_view path, const RNTupleReadOptions &options);
   std::unique_ptr<RPageSource> Clone() const final;
   virtual ~RPageSourceRoot();

   RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex) final;
   RPage PopulatePage(ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex) final;
   void ReleasePage(RPage &page) final;

   RNTupleMetrics &GetMetrics() final { return fMetrics; }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
