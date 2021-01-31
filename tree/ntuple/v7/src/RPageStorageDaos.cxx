/// \file RPageStorageDaos.cxx
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

#include <ROOT/RCluster.hxx>
#include <ROOT/RClusterPool.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPagePool.hxx>
#include <ROOT/RDaos.hxx>
#include <ROOT/RPageStorageDaos.hxx>

#include <RVersion.h>
#include <TError.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <regex>

namespace {
struct RDaosLocator {
   /// \brief UUID of the DAOS pool
   std::string fPoolUuid;
   /// \brief Ranks of the service replicas, separated by `_`
   std::string fSvcReplicas;
   /// \brief UUID of the container for this RNTuple
   std::string fContainerUuid;
};

/**
  \brief Parse a DAOS pool URI of the form 'daos://pool-uuid:svc_replicas/container_uuid'.
*/
RDaosLocator ParseDaosPoolURI(std::string_view uri)
{
   std::regex re("daos://([[:xdigit:]-]+):([[:digit:]_]+)/([[:xdigit:]-]+)");
   std::cmatch m;
   if (!std::regex_match(uri.data(), m, re))
      throw std::runtime_error("Invalid DAOS pool URI.");
   return { m[1], m[2], m[3] };
}

/// \brief Some random distribution/attribute key.  TODO: apply recommended schema, i.e.
/// an OID for each cluster + a dkey for each page. 
static constexpr std::uint64_t kDistributionKey = 0x5a3c69f0cafe4a11;
static constexpr std::uint64_t kAttributeKey = 0x4243544b5344422d;

static constexpr daos_obj_id_t kOidAnchor{std::uint64_t(-1), 0};
static constexpr daos_obj_id_t kOidHeader{std::uint64_t(-2), 0};
static constexpr daos_obj_id_t kOidFooter{std::uint64_t(-3), 0};
} // namespace


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPageSinkDaos::RPageSinkDaos(std::string_view ntupleName, std::string_view locator,
   const RNTupleWriteOptions &options)
   : RPageSink(ntupleName, options)
   , fMetrics("RPageSinkRoot")
   , fPageAllocator(std::make_unique<RPageAllocatorHeap>())
   , fLocator(locator)
{
}


ROOT::Experimental::Detail::RPageSinkDaos::~RPageSinkDaos()
{
}


void ROOT::Experimental::Detail::RPageSinkDaos::CreateImpl(const RNTupleModel & /* model */)
{
   auto args = ParseDaosPoolURI(fLocator);
   fDaosPool = std::make_shared<RDaosPool>(args.fPoolUuid, args.fSvcReplicas);
   fDaosContainer = std::make_unique<RDaosContainer>(fDaosPool, args.fContainerUuid, /*create =*/ true);

   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szHeader = descriptor.SerializeHeader(nullptr);
   auto buffer = std::unique_ptr<unsigned char[]>(new unsigned char[szHeader]);
   descriptor.SerializeHeader(buffer.get());

   auto zipBuffer = std::unique_ptr<unsigned char[]>(new unsigned char[szHeader]);
   auto szZipHeader = fCompressor(buffer.get(), szHeader, fOptions.GetCompression(),
      [&zipBuffer](const void *b, size_t n, size_t o){ memcpy(zipBuffer.get() + o, b, n); } );
   WriteNTupleHeader(zipBuffer.get(), szZipHeader, szHeader);
}


ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkDaos::CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page)
{
   // TODO: change this to use general-purpose SerializePage/DeserializePage after the required PR merge.
   unsigned char *buffer = reinterpret_cast<unsigned char *>(page.GetBuffer());
   bool isAdoptedBuffer = true;
   auto packedBytes = page.GetSize();
   auto element = columnHandle.fColumn->GetElement();
   const auto isMappable = element->IsMappable();

   if (!isMappable) {
      packedBytes = (page.GetNElements() * element->GetBitsOnStorage() + 7) / 8;
      buffer = new unsigned char[packedBytes];
      isAdoptedBuffer = false;
      element->Pack(buffer, page.GetBuffer(), page.GetNElements());
   }
   auto zippedBytes = packedBytes;

   if (fOptions.GetCompression() != 0) {
      zippedBytes = fCompressor(buffer, packedBytes, fOptions.GetCompression());
      if (!isAdoptedBuffer)
         delete[] buffer;
      buffer = const_cast<unsigned char *>(reinterpret_cast<const unsigned char *>(fCompressor.GetZipBuffer()));
      isAdoptedBuffer = true;
   }

   auto offsetData = std::get<0>(fDaosContainer->WriteObject(buffer, zippedBytes,
                                                         kDistributionKey, kAttributeKey)).lo;
   fClusterMinOffset = std::min(offsetData, fClusterMinOffset);
   fClusterMaxOffset = std::max(offsetData + zippedBytes, fClusterMaxOffset);

   if (!isAdoptedBuffer)
      delete[] buffer;

   RClusterDescriptor::RLocator result;
   result.fPosition = offsetData;
   result.fBytesOnStorage = zippedBytes;
   return result;
}


// TODO(jalopezg): the current byte range arithmetic makes little sense for the
// object store. We might find out, however, that there are native ways to group
// clusters in DAOS.
ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkDaos::CommitClusterImpl(ROOT::Experimental::NTupleSize_t /* nEntries */)
{
   RClusterDescriptor::RLocator result;
   result.fPosition = fClusterMinOffset;
   result.fBytesOnStorage = fClusterMaxOffset - fClusterMinOffset;
   fClusterMinOffset = std::uint64_t(-1);
   fClusterMaxOffset = 0;
   return result;
}


void ROOT::Experimental::Detail::RPageSinkDaos::CommitDatasetImpl()
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szFooter = descriptor.SerializeFooter(nullptr);
   auto buffer = std::unique_ptr<unsigned char []>(new unsigned char[szFooter]);
   descriptor.SerializeFooter(buffer.get());

   auto zipBuffer = std::unique_ptr<unsigned char[]>(new unsigned char[szFooter]);
   auto szZipFooter = fCompressor(buffer.get(), szFooter, fOptions.GetCompression(),
      [&zipBuffer](const void *b, size_t n, size_t o){ memcpy(zipBuffer.get() + o, b, n); } );
   WriteNTupleFooter(zipBuffer.get(), szZipFooter, szFooter);
   WriteNTupleAnchor();
}


void ROOT::Experimental::Detail::RPageSinkDaos::WriteNTupleHeader(
		const void *data, size_t nbytes, size_t lenHeader)
{
   fDaosContainer->WriteObject(kOidHeader, data, nbytes, kDistributionKey, kAttributeKey);
   fNTupleAnchor.fLenHeader = lenHeader;
   fNTupleAnchor.fNBytesHeader = nbytes;
}

void ROOT::Experimental::Detail::RPageSinkDaos::WriteNTupleFooter(
		const void *data, size_t nbytes, size_t lenFooter)
{
   fDaosContainer->WriteObject(kOidFooter, data, nbytes, kDistributionKey, kAttributeKey);
   fNTupleAnchor.fLenFooter = lenFooter;
   fNTupleAnchor.fNBytesFooter = nbytes;
}

void ROOT::Experimental::Detail::RPageSinkDaos::WriteNTupleAnchor() {
   fDaosContainer->WriteObject(kOidAnchor, &fNTupleAnchor, sizeof(fNTupleAnchor),
                           kDistributionKey, kAttributeKey);
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSinkDaos::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   if (nElements == 0)
      nElements = kDefaultElementsPerPage;
   auto elementSize = columnHandle.fColumn->GetElement()->GetSize();
   return fPageAllocator->NewPage(columnHandle.fId, elementSize, nElements);
}

void ROOT::Experimental::Detail::RPageSinkDaos::ReleasePage(RPage &page)
{
   fPageAllocator->DeletePage(page);
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageAllocatorDaos::NewPage(
   ColumnId_t columnId, void *mem, std::size_t elementSize, std::size_t nElements)
{
   RPage newPage(columnId, mem, elementSize * nElements, elementSize);
   newPage.TryGrow(nElements);
   return newPage;
}

void ROOT::Experimental::Detail::RPageAllocatorDaos::DeletePage(const RPage& page)
{
   if (page.IsNull())
      return;
   delete[] reinterpret_cast<unsigned char *>(page.GetBuffer());
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPageSourceDaos::RPageSourceDaos(std::string_view ntupleName, std::string_view locator,
   const RNTupleReadOptions &options)
   : RPageSource(ntupleName, options)
   , fMetrics("RPageSourceDaos")
   , fPageAllocator(std::make_unique<RPageAllocatorDaos>())
   , fPagePool(std::make_shared<RPagePool>())
   , fLocator(locator)
   , fClusterPool(std::make_unique<RClusterPool>(*this))
{
   fCounters = std::unique_ptr<RCounters>(new RCounters{
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("nReadV", "", "number of vector read requests"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("nRead", "", "number of byte ranges read"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("szReadPayload", "B", "volume read from container"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*> ("szUnzip", "B", "volume after unzipping"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("nClusterLoaded", "",
                                                   "number of partial clusters preloaded from storage"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*> ("nPageLoaded", "", "number of pages loaded from storage"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*> ("nPagePopulated", "", "number of populated pages"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("timeWallRead", "ns", "wall clock time spent reading"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*> ("timeWallUnzip", "ns", "wall clock time spent decompressing"),
      *fMetrics.MakeCounter<RNTupleTickCounter<RNTupleAtomicCounter>*>("timeCpuRead", "ns", "CPU time spent reading"),
      *fMetrics.MakeCounter<RNTupleTickCounter<RNTupleAtomicCounter>*> ("timeCpuUnzip", "ns",
                                                                       "CPU time spent decompressing")
   });

   auto args = ParseDaosPoolURI(locator);
   fDaosPool = std::make_shared<RDaosPool>(args.fPoolUuid, args.fSvcReplicas);
   fDaosContainer = std::make_unique<RDaosContainer>(fDaosPool, args.fContainerUuid);
}


ROOT::Experimental::Detail::RPageSourceDaos::~RPageSourceDaos() = default;


ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Detail::RPageSourceDaos::AttachImpl()
{
   RNTupleDescriptorBuilder descBuilder;
   RDaosNTuple ntpl;
   fDaosContainer->ReadObject(kOidAnchor, &ntpl, sizeof(ntpl), kDistributionKey, kAttributeKey);

   auto buffer = std::unique_ptr<unsigned char[]>(new unsigned char[ntpl.fLenHeader]);
   auto zipBuffer = std::unique_ptr<unsigned char[]>(new unsigned char[ntpl.fNBytesHeader]);
   fDaosContainer->ReadObject(kOidHeader, zipBuffer.get(), ntpl.fNBytesHeader, kDistributionKey, kAttributeKey);
   fDecompressor(zipBuffer.get(), ntpl.fNBytesHeader, ntpl.fLenHeader, buffer.get());
   descBuilder.SetFromHeader(buffer.get());

   buffer = std::unique_ptr<unsigned char[]>(new unsigned char[ntpl.fLenFooter]);
   zipBuffer = std::unique_ptr<unsigned char[]>(new unsigned char[ntpl.fNBytesFooter]);
   fDaosContainer->ReadObject(kOidFooter, zipBuffer.get(), ntpl.fNBytesFooter, kDistributionKey, kAttributeKey);
   fDecompressor(zipBuffer.get(), ntpl.fNBytesFooter, ntpl.fLenFooter, buffer.get());
   descBuilder.AddClustersFromFooter(buffer.get());

   return descBuilder.MoveDescriptor();
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceDaos::PopulatePageFromCluster(
   ColumnHandle_t columnHandle, const RClusterDescriptor &clusterDescriptor, ClusterSize_t::ValueType clusterIndex)
{
   const auto columnId = columnHandle.fId;
   const auto clusterId = clusterDescriptor.GetId();
   const auto &pageRange = clusterDescriptor.GetPageRange(columnId);

   fCounters->fNPagePopulated.Inc();

   // TODO(jblomer): binary search
   RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   decltype(clusterIndex) firstInPage = 0;
   NTupleSize_t pageNo = 0;
   for (const auto &pi : pageRange.fPageInfos) {
      if (firstInPage + pi.fNElements > clusterIndex) {
         pageInfo = pi;
         break;
      }
      firstInPage += pi.fNElements;
      ++pageNo;
   }
   R__ASSERT(firstInPage <= clusterIndex);
   R__ASSERT((firstInPage + pageInfo.fNElements) > clusterIndex);

   const auto element = columnHandle.fColumn->GetElement();
   const auto elementSize = element->GetSize();

   const auto bytesOnStorage = pageInfo.fLocator.fBytesOnStorage;
   const auto bytesPacked = (element->GetBitsOnStorage() * pageInfo.fNElements + 7) / 8;
   const auto pageSize = elementSize * pageInfo.fNElements;

   auto pageBuffer = new unsigned char[bytesPacked];
   if (fOptions.GetClusterCache() == RNTupleReadOptions::EClusterCache::kOff) {
      fDaosContainer->ReadObject({static_cast<decltype(daos_obj_id_t::lo)>(pageInfo.fLocator.fPosition), 0},
                             pageBuffer, bytesOnStorage, kDistributionKey, kAttributeKey);
      fCounters->fNPageLoaded.Inc();
   } else {
      if (!fCurrentCluster || (fCurrentCluster->GetId() != clusterId) || !fCurrentCluster->ContainsColumn(columnId))
         fCurrentCluster = fClusterPool->GetCluster(clusterId, fActiveColumns);
      R__ASSERT(fCurrentCluster->ContainsColumn(columnId));
      ROnDiskPage::Key key(columnId, pageNo);
      auto onDiskPage = fCurrentCluster->GetOnDiskPage(key);
      R__ASSERT(onDiskPage);
      R__ASSERT(bytesOnStorage == onDiskPage->GetSize());
      memcpy(pageBuffer, onDiskPage->GetAddress(), onDiskPage->GetSize());
   }

   if (bytesOnStorage != bytesPacked) {
      RNTupleAtomicTimer timer(fCounters->fTimeWallUnzip, fCounters->fTimeCpuUnzip);
      fDecompressor(pageBuffer, bytesOnStorage, bytesPacked);
      fCounters->fSzUnzip.Add(bytesPacked);
   }

   if (!element->IsMappable()) {
      auto unpackedBuffer = new unsigned char[pageSize];
      element->Unpack(unpackedBuffer, pageBuffer, pageInfo.fNElements);
      delete[] pageBuffer;
      pageBuffer = unpackedBuffer;
   }

   const auto indexOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   auto newPage = fPageAllocator->NewPage(columnId, pageBuffer, elementSize, pageInfo.fNElements);
   newPage.SetWindow(indexOffset + firstInPage, RPage::RClusterInfo(clusterId, indexOffset));
   fPagePool->RegisterPage(newPage,
      RPageDeleter([](const RPage &page, void * /*userData*/)
      {
         RPageAllocatorDaos::DeletePage(page);
      }, nullptr));
   return newPage;
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceDaos::PopulatePage(
   ColumnHandle_t columnHandle, NTupleSize_t globalIndex)
{
   const auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, globalIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   const auto clusterId = fDescriptor.FindClusterId(columnId, globalIndex);
   R__ASSERT(clusterId != kInvalidDescriptorId);
   const auto &clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
   const auto selfOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   R__ASSERT(selfOffset <= globalIndex);
   return PopulatePageFromCluster(columnHandle, clusterDescriptor, globalIndex - selfOffset);
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceDaos::PopulatePage(
   ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex)
{
   const auto clusterId = clusterIndex.GetClusterId();
   const auto index = clusterIndex.GetIndex();
   const auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, clusterIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   R__ASSERT(clusterId != kInvalidDescriptorId);
   const auto &clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
   return PopulatePageFromCluster(columnHandle, clusterDescriptor, index);
}

void ROOT::Experimental::Detail::RPageSourceDaos::ReleasePage(RPage &page)
{
   fPagePool->ReturnPage(page);
}

std::unique_ptr<ROOT::Experimental::Detail::RPageSource> ROOT::Experimental::Detail::RPageSourceDaos::Clone() const
{
   auto clone = new RPageSourceDaos(fNTupleName, fLocator, fOptions);
   return std::unique_ptr<RPageSourceDaos>(clone);
}

std::unique_ptr<ROOT::Experimental::Detail::RCluster>
ROOT::Experimental::Detail::RPageSourceDaos::LoadCluster(DescriptorId_t clusterId, const ColumnSet_t &columns)
{
   fCounters->fNClusterLoaded.Inc();

   const auto &clusterDesc = GetDescriptor().GetClusterDescriptor(clusterId);
   auto clusterLocator = clusterDesc.GetLocator();
   auto clusterSize = clusterLocator.fBytesOnStorage;
   R__ASSERT(clusterSize > 0);

   struct ROnDiskPageLocator {
      ROnDiskPageLocator() = default;
      ROnDiskPageLocator(DescriptorId_t c, NTupleSize_t p, std::uint64_t o, std::uint64_t s, std::size_t b)
         : fColumnId(c), fPageNo(p), fOffset(o), fSize(s), fBufPos(b) {}
      DescriptorId_t fColumnId = 0;
      NTupleSize_t fPageNo = 0;
      std::uint64_t fOffset = 0;
      std::uint64_t fSize = 0;
      std::size_t fBufPos = 0;
   };

   // Collect the page necessary page meta-data and sum up the total size of the compressed and packed pages
   std::vector<ROnDiskPageLocator> onDiskPages;
   std::size_t szPayload = 0;
   for (auto columnId : columns) {
      const auto &pageRange = clusterDesc.GetPageRange(columnId);
      NTupleSize_t pageNo = 0;
      for (const auto &pageInfo : pageRange.fPageInfos) {
         const auto &pageLocator = pageInfo.fLocator;
         onDiskPages.emplace_back(ROnDiskPageLocator(
            columnId, pageNo, pageLocator.fPosition, pageLocator.fBytesOnStorage, szPayload));
         szPayload += pageLocator.fBytesOnStorage;
         ++pageNo;
      }
   }

   // Prepare the input vector for the RDaosContainer::ReadV() call
   std::vector<RDaosContainer::RWOperation<std::uint64_t, std::uint64_t>> readRequests;
   auto buffer = new unsigned char[szPayload];
   for (auto &s : onDiskPages) {
      std::vector<d_iov_t> iovs(1);
      d_iov_set(&iovs[0], buffer + s.fBufPos, s.fSize);
      readRequests.emplace_back(daos_obj_id_t{s.fOffset, 0},
                                kDistributionKey, kAttributeKey, iovs);
   }
   fCounters->fSzReadPayload.Add(szPayload);

   // Register the on disk pages in a page map
   auto pageMap = std::make_unique<ROnDiskPageMapHeap>(std::unique_ptr<unsigned char []>(buffer));
   for (const auto &s : onDiskPages) {
      ROnDiskPage::Key key(s.fColumnId, s.fPageNo);
      pageMap->Register(key, ROnDiskPage(buffer + s.fBufPos, s.fSize));
   }
   fCounters->fNPageLoaded.Add(onDiskPages.size());

   {
      RNTupleAtomicTimer timer(fCounters->fTimeWallRead, fCounters->fTimeCpuRead);
      fDaosContainer->ReadV(readRequests);
   }
   fCounters->fNReadV.Inc();
   fCounters->fNRead.Add(readRequests.size());

   auto cluster = std::make_unique<RCluster>(clusterId);
   cluster->Adopt(std::move(pageMap));
   for (auto colId : columns)
      cluster->SetColumnAvailable(colId);
   return cluster;
}
