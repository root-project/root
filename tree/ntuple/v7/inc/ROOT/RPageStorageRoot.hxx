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

#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

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

   RNTupleMetrics fMetrics;
   std::unique_ptr<RPageAllocatorHeap> fPageAllocator;

   /// Currently, an ntuple is stored as a directory in a TFile
   FILE *fBinaryFile = nullptr;
   std::unique_ptr<TFile> fFile;
   TDirectory *fDirectory = nullptr;


   /// Instead of a physical file offset, pages in root are identified by an index which becomes part of the key
   DescriptorId_t fLastPageIdx = 0;

   std::uint64_t Write(void *from, size_t size, std::uint64_t offset);

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
