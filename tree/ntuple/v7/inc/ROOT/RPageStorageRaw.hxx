/// \file ROOT/RPageStorageRaw.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-08-23
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageStorageRaw
#define ROOT7_RPageStorageRaw

#include <ROOT/RPageStorage.hxx>
#include <ROOT/RStringView.hxx>

#include <cstdint>
#include <cstdio>
#include <memory>

namespace ROOT {
namespace Experimental {
namespace Detail {

class RPageAllocatorHeap;
class RPagePool;
class RRawFile;

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSinkRaw
\ingroup NTuple
\brief Storage provider that write ntuple pages into a raw binary file
*/
// clang-format on
class RPageSinkRaw : public RPageSink {
public:
   struct RSettings {
      FILE *fFile = nullptr;  // TODO(jblomer): add write support to RRawFile
      int fCompressionSettings = 0;
   };

private:
   static constexpr std::size_t kDefaultElementsPerPage = 10000;
   std::unique_ptr<RPageAllocatorHeap> fPageAllocator;
   RSettings fSettings;
   size_t fFilePos = 0;
   size_t fClusterStart = 0;

   void Write(const void *buffer, std::size_t nbytes);

protected:
   void DoCreate(const RNTupleModel &model) final;
   RClusterDescriptor::RLocator DoCommitPage(ColumnHandle_t columnHandle, const RPage &page) final;
   RClusterDescriptor::RLocator DoCommitCluster(NTupleSize_t nEntries) final;
   void DoCommitDataset() final;

public:
   RPageSinkRaw(std::string_view ntupleName, RSettings settings);
   RPageSinkRaw(std::string_view ntupleName, std::string_view path);
   virtual ~RPageSinkRaw();

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements = 0) final;
   void ReleasePage(RPage &page) final;
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
\class ROOT::Experimental::Detail::RPageSourceRaw
\ingroup NTuple
\brief Storage provider that reads ntuple pages from a raw file
*/
// clang-format on
class RPageSourceRaw : public RPageSource {
public:
   struct RSettings {
      std::unique_ptr<RRawFile> fFile;
   };

private:
   std::unique_ptr<RPageAllocatorFile> fPageAllocator;
   std::shared_ptr<RPagePool> fPagePool;
   RSettings fSettings;

   void Read(void *buffer, std::size_t nbytes, std::uint64_t offset);
   RPage PopulatePageFromCluster(ColumnHandle_t columnHandle,
                                 const RClusterDescriptor &clusterDescriptor,
                                 ClusterSize_t::ValueType clusterIndex);

protected:
   RNTupleDescriptor DoAttach() final;

public:
   RPageSourceRaw(std::string_view ntupleName, std::string_view path);
   std::unique_ptr<RPageSource> Clone() const final;
   virtual ~RPageSourceRaw();

   RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex) final;
   RPage PopulatePage(ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex) final;
   void ReleasePage(RPage &page) final;
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
