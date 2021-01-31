/// \file ROOT/RPageStorageDaos.hxx
/// \ingroup NTuple ROOT7
/// \author Javier Lopez-Gomez <j.lopez@cern.ch>
/// \date 2020-11-03
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageStorageDaos
#define ROOT7_RPageStorageDaos

#include <ROOT/RPageStorage.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RStringView.hxx>

#include <array>
#include <cstdio>
#include <memory>
#include <string>

namespace ROOT {

namespace Experimental {
namespace Detail {

class RCluster;
class RClusterPool;
class RPageAllocatorHeap;
class RPagePool;
class RDaosPool;
class RDaosContainer;


// clang-format off
/**
\class ROOT::Experimental::RDaosNTuple
\ingroup NTuple
\brief Entry point for an RNTuple in a DAOS container. It encodes essential
information to read the ntuple; currently, it contains (un)compressed size of
the header/footer blobs.
*/
// clang-format on
struct RDaosNTuple {
   /// Allows for evolving the struct in future versions
   std::uint32_t fVersion = 0;
   /// The size of the compressed ntuple header
   std::uint32_t fNBytesHeader = 0;
   /// The size of the uncompressed ntuple header
   std::uint32_t fLenHeader = 0;
   /// The size of the compressed ntuple footer
   std::uint32_t fNBytesFooter = 0;
   /// The size of the uncompressed ntuple footer
   std::uint32_t fLenFooter = 0;
   /// Currently unused, reserved for later use
   std::uint64_t fReserved = 0;

   bool operator ==(const RDaosNTuple &other) const {
      return fVersion == other.fVersion &&
         fNBytesHeader == other.fNBytesHeader &&
         fLenHeader == other.fLenHeader &&
         fNBytesFooter == other.fNBytesFooter &&
         fLenFooter == other.fLenFooter &&
         fReserved == other.fReserved;
   }
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSinkDaos
\ingroup NTuple
\brief Storage provider that writes ntuple pages to into a DAOS container

Currently, an object is allocated for each page + 3 additional objects (anchor/header/footer).
*/
// clang-format on
class RPageSinkDaos : public RPageSink {
private:
   static constexpr std::size_t kDefaultElementsPerPage = 10000;

   RNTupleMetrics fMetrics;
   std::unique_ptr<RPageAllocatorHeap> fPageAllocator;

   std::shared_ptr<RDaosPool> fDaosPool;
   std::unique_ptr<RDaosContainer> fDaosContainer;
   /// A URI to a DAOS pool of the form 'daos://pool-uuid:svc_replicas/container-uuid'
   std::string fLocator;

   RDaosNTuple fNTupleAnchor;
   // FIXME: do we really need these data members?
   /// Byte offset of the first page of the current cluster
   std::uint64_t fClusterMinOffset = std::uint64_t(-1);
   /// Byte offset of the end of the last page of the current cluster
   std::uint64_t fClusterMaxOffset = 0;
   /// Helper for zipping keys and header / footer; comprises a 16MB zip buffer
   RNTupleCompressor fCompressor;

protected:
   void CreateImpl(const RNTupleModel &model) final;
   RClusterDescriptor::RLocator CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page) final;
   RClusterDescriptor::RLocator CommitClusterImpl(NTupleSize_t nEntries) final;
   void CommitDatasetImpl() final;
   void WriteNTupleHeader(const void *data, size_t nbytes, size_t lenHeader);
   void WriteNTupleFooter(const void *data, size_t nbytes, size_t lenFooter);
   void WriteNTupleAnchor();

public:
   RPageSinkDaos(std::string_view ntupleName, std::string_view locator, const RNTupleWriteOptions &options);
   virtual ~RPageSinkDaos();

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements = 0) final;
   void ReleasePage(RPage &page) final;

   RNTupleMetrics &GetMetrics() final { return fMetrics; }
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageAllocatorDaos
\ingroup NTuple
\brief Manages pages read from a DAOS container
*/
// clang-format on
class RPageAllocatorDaos {
public:
   static RPage NewPage(ColumnId_t columnId, void *mem, std::size_t elementSize, std::size_t nElements);
   static void DeletePage(const RPage& page);
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSourceDaos
\ingroup NTuple
\brief Storage provider that reads ntuple pages from a DAOS container
*/
// clang-format on
class RPageSourceDaos : public RPageSource {
public:
   // FIXME: this value probably needs to match DAOS object size limit.
   /// Cannot process pages larger than 1MB
   static constexpr std::size_t kMaxPageSize = 1024 * 1024;

private:
   /// I/O performance counters that get registered in fMetrics
   struct RCounters {
      RNTupleAtomicCounter &fNReadV;
      RNTupleAtomicCounter &fNRead;
      RNTupleAtomicCounter &fSzReadPayload;
      RNTupleAtomicCounter &fSzUnzip;
      RNTupleAtomicCounter &fNClusterLoaded;
      RNTupleAtomicCounter &fNPageLoaded;
      RNTupleAtomicCounter &fNPagePopulated;
      RNTupleAtomicCounter &fTimeWallRead;
      RNTupleAtomicCounter &fTimeWallUnzip;
      RNTupleTickCounter<RNTupleAtomicCounter> &fTimeCpuRead;
      RNTupleTickCounter<RNTupleAtomicCounter> &fTimeCpuUnzip;
   };
   std::unique_ptr<RCounters> fCounters;
   /// Wraps the I/O counters and is observed by the RNTupleReader metrics
   RNTupleMetrics fMetrics;

   /// Populated pages might be shared; there memory buffer is managed by the RPageAllocatorDaos
   std::unique_ptr<RPageAllocatorDaos> fPageAllocator;
   // TODO: the page pool should probably be handled by the base class.
   /// The page pool might, at some point, be used by multiple page sources
   std::shared_ptr<RPagePool> fPagePool;
   /// The last cluster from which a page got populated.  Points into fClusterPool->fPool
   RCluster *fCurrentCluster = nullptr;
   /// Helper to unzip pages and header/footer; comprises a 16MB unzip buffer
   RNTupleDecompressor fDecompressor;
   /// A connection to a DAOS pool
   std::shared_ptr<RDaosPool> fDaosPool;
   /// A container that stores object data (header/footer, pages, etc.)
   std::unique_ptr<RDaosContainer> fDaosContainer;
   /// A URI to a DAOS pool of the form 'daos://pool-uuid:svc_replicas/container-uuid'
   std::string fLocator;
   /// The cluster pool asynchronously preloads the next few clusters
   std::unique_ptr<RClusterPool> fClusterPool;

   RPage PopulatePageFromCluster(ColumnHandle_t columnHandle, const RClusterDescriptor &clusterDescriptor,
                                 ClusterSize_t::ValueType clusterIndex);

protected:
   RNTupleDescriptor AttachImpl() final;

public:
   RPageSourceDaos(std::string_view ntupleName, std::string_view locator, const RNTupleReadOptions &options);
   /// The cloned page source creates a new connection to the pool/container.
   /// The meta-data (header and footer) is reread and parsed by the clone.
   std::unique_ptr<RPageSource> Clone() const final;
   virtual ~RPageSourceDaos();

   RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex) final;
   RPage PopulatePage(ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex) final;
   void ReleasePage(RPage &page) final;

   std::unique_ptr<RCluster> LoadCluster(DescriptorId_t clusterId, const ColumnSet_t &columns) final;

   RNTupleMetrics &GetMetrics() final { return fMetrics; }
};


} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
