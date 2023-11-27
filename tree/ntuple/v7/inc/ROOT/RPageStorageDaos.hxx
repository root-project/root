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

#include <ROOT/RError.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <string_view>

#include <array>
#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <optional>

namespace ROOT {
namespace Experimental {

namespace Internal {
enum EDaosLocatorFlags {
   // Indicates that the referenced page is "caged", i.e. it is stored in a larger blob that contains multiple pages.
   kCagedPage = 0x01,
};
}

namespace Detail {

class RCluster;
class RClusterPool;
class RPageAllocatorHeap;
class RPagePool;
class RDaosPool;
class RDaosContainer;

using ntuple_index_t = std::uint32_t;

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
   RResult<std::uint32_t> Deserialize(const void *buffer, std::uint32_t bufSize);

   static std::uint32_t GetSize();
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RDaosContainerNTupleLocator
\ingroup NTuple
\brief Helper structure concentrating the functionality required to locate an ntuple within a DAOS container.
It includes a hashing function that converts the RNTuple's name into a 32-bit identifier; this value is used to index
the subspace for the ntuple among all objects in the container. A zero-value hash value is reserved for storing any
future metadata related to container-wide management; a zero-index ntuple is thus disallowed and remapped to "1".
Once the index is computed, `InitNTupleDescriptorBuilder()` can be called to return a partially-filled builder with
the ntuple's anchor, header and footer, lacking only pagelists. Upon that call, a copy of the anchor is stored in `fAnchor`.
*/
// clang-format on
struct RDaosContainerNTupleLocator {
   std::string fName{};
   ntuple_index_t fIndex{};
   std::optional<ROOT::Experimental::Detail::RDaosNTupleAnchor> fAnchor;
   static const ntuple_index_t kReservedIndex = 0;

   RDaosContainerNTupleLocator() = default;
   explicit RDaosContainerNTupleLocator(const std::string &ntupleName) : fName(ntupleName), fIndex(Hash(ntupleName)){};

   bool IsValid() { return fAnchor.has_value() && fAnchor->fNBytesHeader; }
   [[nodiscard]] ntuple_index_t GetIndex() const { return fIndex; };
   static ntuple_index_t Hash(const std::string &ntupleName)
   {
      // Convert string to numeric representation via `std::hash`.
      uint64_t h = std::hash<std::string>{}(ntupleName);
      // Fold the hash into 32-bit using `boost::hash_combine()` algorithm and magic number.
      auto seed = static_cast<uint32_t>(h >> 32);
      seed ^= static_cast<uint32_t>(h & 0xffffffff) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      auto hash = static_cast<ntuple_index_t>(seed);
      return (hash == kReservedIndex) ? kReservedIndex + 1 : hash;
   }

   int InitNTupleDescriptorBuilder(RDaosContainer &cont, RNTupleDecompressor &decompressor,
                                   RNTupleDescriptorBuilder &builder);

   static std::pair<RDaosContainerNTupleLocator, RNTupleDescriptorBuilder>
   LocateNTuple(RDaosContainer &cont, const std::string &ntupleName, RNTupleDecompressor &decompressor);
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSinkDaos
\ingroup NTuple
\brief Storage provider that writes ntuple pages to into a DAOS container

Currently, an object is allocated for ntuple metadata (anchor/header/footer).
Objects can correspond to pages or clusters of pages depending on the RNTuple-DAOS mapping strategy.
*/
// clang-format on
class RPageSinkDaos : public RPagePersistentSink {
private:
   std::unique_ptr<RPageAllocatorHeap> fPageAllocator;

   /// \brief Underlying DAOS container. An internal `std::shared_ptr` keep the pool connection alive.
   /// ISO C++ ensures the correct destruction order, i.e., `~RDaosContainer` is invoked first
   /// (which calls `daos_cont_close()`; the destructor for the `std::shared_ptr<RDaosPool>` is invoked
   /// after (which calls `daos_pool_disconect()`).
   std::unique_ptr<RDaosContainer> fDaosContainer;
   /// Page identifier for the next committed page; it is automatically incremented in `CommitSealedPageImpl()`
   std::atomic<std::uint64_t> fPageId{0};
   /// Cluster group counter for the next committed cluster pagelist; incremented in `CommitClusterGroupImpl()`
   std::atomic<std::uint64_t> fClusterGroupId{0};
   /// \brief A URI to a DAOS pool of the form 'daos://pool-label/container-label'
   std::string fURI;
   /// Tracks the number of bytes committed to the current cluster
   std::uint64_t fNBytesCurrentCluster{0};

   RDaosNTupleAnchor fNTupleAnchor;
   ntuple_index_t fNTupleIndex{0};
   uint32_t fCageSizeLimit{};

protected:
   void CreateImpl(const RNTupleModel &model, unsigned char *serializedHeader, std::uint32_t length) final;
   RNTupleLocator CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page) final;
   RNTupleLocator
   CommitSealedPageImpl(DescriptorId_t physicalColumnId, const RPageStorage::RSealedPage &sealedPage) final;
   std::vector<RNTupleLocator> CommitSealedPageVImpl(std::span<RPageStorage::RSealedPageGroup> ranges) final;
   std::uint64_t CommitClusterImpl(NTupleSize_t nEntries) final;
   RNTupleLocator CommitClusterGroupImpl(unsigned char *serializedPageList, std::uint32_t length) final;
   void CommitDatasetImpl(unsigned char *serializedFooter, std::uint32_t length) final;
   void WriteNTupleHeader(const void *data, size_t nbytes, size_t lenHeader);
   void WriteNTupleFooter(const void *data, size_t nbytes, size_t lenFooter);
   void WriteNTupleAnchor();

public:
   RPageSinkDaos(std::string_view ntupleName, std::string_view uri, const RNTupleWriteOptions &options);
   ~RPageSinkDaos() override;

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements) final;
   void ReleasePage(RPage &page) final;
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
   /// Summarizes cluster-level information that are necessary to populate a certain page.
   /// Used by PopulatePageFromCluster().
   struct RClusterInfo {
      DescriptorId_t fClusterId = 0;
      /// Location of the page on disk
      RClusterDescriptor::RPageRange::RPageInfoExtended fPageInfo;
      /// The first element number of the page's column in the given cluster
      std::uint64_t fColumnOffset = 0;
   };

   ntuple_index_t fNTupleIndex{0};

   /// Populated pages might be shared; the page pool might, at some point, be used by multiple page sources
   std::shared_ptr<RPagePool> fPagePool;
   /// The last cluster from which a page got populated.  Points into fClusterPool->fPool
   RCluster *fCurrentCluster = nullptr;
   /// A container that stores object data (header/footer, pages, etc.)
   std::unique_ptr<RDaosContainer> fDaosContainer;
   /// A URI to a DAOS pool of the form 'daos://pool-label/container-label'
   std::string fURI;
   /// The cluster pool asynchronously preloads the next few clusters
   std::unique_ptr<RClusterPool> fClusterPool;

   RNTupleDescriptorBuilder fDescriptorBuilder;

   RPage PopulatePageFromCluster(ColumnHandle_t columnHandle, const RClusterInfo &clusterInfo,
                                 ClusterSize_t::ValueType idxInCluster);

protected:
   RNTupleDescriptor AttachImpl() final;
   void UnzipClusterImpl(RCluster *cluster) final;

public:
   RPageSourceDaos(std::string_view ntupleName, std::string_view uri, const RNTupleReadOptions &options);
   /// The cloned page source creates a new connection to the pool/container.
   /// The meta-data (header and footer) is reread and parsed by the clone.
   std::unique_ptr<RPageSource> Clone() const final;
   ~RPageSourceDaos() override;

   RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex) final;
   RPage PopulatePage(ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex) final;
   void ReleasePage(RPage &page) final;

   void
   LoadSealedPage(DescriptorId_t physicalColumnId, const RClusterIndex &clusterIndex, RSealedPage &sealedPage) final;

   std::vector<std::unique_ptr<RCluster>> LoadClusters(std::span<RCluster::RKey> clusterKeys) final;

   /// Return the object class used for user data OIDs in this ntuple.
   std::string GetObjectClass() const;
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
