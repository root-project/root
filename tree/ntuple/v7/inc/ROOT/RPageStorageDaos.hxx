/// \file ROOT/RPageStorageDaos.hxx
/// \ingroup NTuple ROOT7
/// \author Javier Lopez-Gomez <j.lopez@cern.ch>
/// \date 2020-11-03
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageStorageDaos
#define ROOT7_RPageStorageDaos

#include <ROOT/RPageStorage.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RStringView.hxx>

#include <array>
#include <atomic>
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
\class ROOT::Experimental::Detail::RDaosNTupleAnchor
\ingroup NTuple
\brief Entry point for an RNTuple in a DAOS container. It encodes essential
information to read the ntuple; currently, it contains (un)compressed size of
the header/footer blobs and the object class for user data OIDs.
The length of a serialized anchor cannot be greater than the value returned by the `GetSize` function.
*/
// clang-format on
struct RDaosNTupleAnchor {
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
   /// The object class for user data OIDs, e.g. `SX`
   std::string fObjClass{};

   bool operator ==(const RDaosNTupleAnchor &other) const {
      return fVersion == other.fVersion &&
         fNBytesHeader == other.fNBytesHeader &&
         fLenHeader == other.fLenHeader &&
         fNBytesFooter == other.fNBytesFooter &&
         fLenFooter == other.fLenFooter &&
         fObjClass == other.fObjClass;
   }

   std::uint32_t Serialize(void *buffer) const;
   std::uint32_t Deserialize(const void *buffer);

   static std::uint32_t GetSize();
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
   std::unique_ptr<RPageAllocatorHeap> fPageAllocator;

   /// \brief Underlying DAOS container. An internal `std::shared_ptr` keep the pool connection alive.
   /// ISO C++ ensures the correct destruction order, i.e., `~RDaosContainer` is invoked first
   /// (which calls `daos_cont_close()`; the destructor for the `std::shared_ptr<RDaosPool>` is invoked
   /// after (which calls `daos_pool_disconect()`).
   std::unique_ptr<RDaosContainer> fDaosContainer;
   /// OID for the next committed page; it is automatically incremented in `CommitSealedPageImpl()`
   std::atomic<std::uint64_t> fOid{0};
   /// \brief A URI to a DAOS pool of the form 'daos://pool-uuid:svc_replicas/container-uuid'
   std::string fURI;
   /// Tracks the number of bytes committed to the current cluster
   std::uint64_t fNBytesCurrentCluster{0};

   RDaosNTupleAnchor fNTupleAnchor;

protected:
   void CreateImpl(const RNTupleModel &model) final;
   RClusterDescriptor::RLocator CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page) final;
   RClusterDescriptor::RLocator CommitSealedPageImpl(DescriptorId_t columnId,
                                                     const RPageStorage::RSealedPage &sealedPage) final;
   std::uint64_t CommitClusterImpl(NTupleSize_t nEntries) final;
   void CommitDatasetImpl() final;
   void WriteNTupleHeader(const void *data, size_t nbytes, size_t lenHeader);
   void WriteNTupleFooter(const void *data, size_t nbytes, size_t lenFooter);
   void WriteNTupleAnchor();

public:
   RPageSinkDaos(std::string_view ntupleName, std::string_view uri, const RNTupleWriteOptions &options);
   virtual ~RPageSinkDaos();

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements) final;
   void ReleasePage(RPage &page) final;
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
private:
   /// Populated pages might be shared; the memory buffer is managed by the RPageAllocatorDaos
   std::unique_ptr<RPageAllocatorDaos> fPageAllocator;
   // TODO: the page pool should probably be handled by the base class.
   /// The page pool might, at some point, be used by multiple page sources
   std::shared_ptr<RPagePool> fPagePool;
   /// The last cluster from which a page got populated.  Points into fClusterPool->fPool
   RCluster *fCurrentCluster = nullptr;
   /// A container that stores object data (header/footer, pages, etc.)
   std::unique_ptr<RDaosContainer> fDaosContainer;
   /// A URI to a DAOS pool of the form 'daos://pool-uuid:svc_replicas/container-uuid'
   std::string fURI;
   /// The cluster pool asynchronously preloads the next few clusters
   std::unique_ptr<RClusterPool> fClusterPool;

   RPage PopulatePageFromCluster(ColumnHandle_t columnHandle, const RClusterDescriptor &clusterDescriptor,
                                 ClusterSize_t::ValueType idxInCluster);

protected:
   RNTupleDescriptor AttachImpl() final;
   void UnzipClusterImpl(RCluster *cluster) final;

public:
   RPageSourceDaos(std::string_view ntupleName, std::string_view uri, const RNTupleReadOptions &options);
   /// The cloned page source creates a new connection to the pool/container.
   /// The meta-data (header and footer) is reread and parsed by the clone.
   std::unique_ptr<RPageSource> Clone() const final;
   virtual ~RPageSourceDaos();

   RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex) final;
   RPage PopulatePage(ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex) final;
   void ReleasePage(RPage &page) final;

   void LoadSealedPage(DescriptorId_t columnId, const RClusterIndex &clusterIndex,
                       RSealedPage &sealedPage) final;

   std::unique_ptr<RCluster> LoadCluster(DescriptorId_t clusterId, const ColumnSet_t &columns) final;

   /// Return the object class used for user data OIDs in this ntuple.
   std::string GetObjectClass() const;
};


} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
