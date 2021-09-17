/// \file RPageStorageDaos.cxx
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

#include <ROOT/RCluster.hxx>
#include <ROOT/RClusterPool.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtil.hxx>
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
struct RDaosURI {
   /// \brief UUID of the DAOS pool
   std::string fPoolUuid;
   /// \brief Ranks of the service replicas, separated by `_`
   std::string fSvcReplicas;
   /// \brief UUID of the container for this RNTuple
   std::string fContainerUuid;
};

/**
  \brief Parse a DAOS RNTuple URI of the form 'daos://pool-uuid:svc_replicas/container_uuid'.
*/
RDaosURI ParseDaosURI(std::string_view uri)
{
   std::regex re("daos://([[:xdigit:]-]+):([[:digit:]_]+)/([[:xdigit:]-]+)");
   std::cmatch m;
   if (!std::regex_match(uri.data(), m, re))
      throw ROOT::Experimental::RException(R__FAIL("Invalid DAOS pool URI."));
   return { m[1], m[2], m[3] };
}

/// \brief Some random distribution/attribute key.  TODO: apply recommended schema, i.e.
/// an OID for each cluster + a dkey for each page.
static constexpr std::uint64_t kDistributionKey = 0x5a3c69f0cafe4a11;
static constexpr std::uint64_t kAttributeKey = 0x4243544b5344422d;

static constexpr daos_obj_id_t kOidAnchor{std::uint64_t(-1), 0};
static constexpr daos_obj_id_t kOidHeader{std::uint64_t(-2), 0};
static constexpr daos_obj_id_t kOidFooter{std::uint64_t(-3), 0};

static constexpr daos_oclass_id_t kCidMetadata = OC_SX;
} // namespace


////////////////////////////////////////////////////////////////////////////////


std::uint32_t
ROOT::Experimental::Detail::RDaosNTupleAnchor::Serialize(void *buffer) const
{
   using namespace ROOT::Experimental::Internal::RNTupleSerialization;
   if (buffer != nullptr) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);
      bytes += SerializeUInt32(fVersion, bytes);
      bytes += SerializeUInt32(fNBytesHeader, bytes);
      bytes += SerializeUInt32(fLenHeader, bytes);
      bytes += SerializeUInt32(fNBytesFooter, bytes);
      bytes += SerializeUInt32(fLenFooter, bytes);
      bytes += SerializeString(fObjClass, bytes);
   }
   return SerializeString(fObjClass, nullptr) + 20;
}

std::uint32_t
ROOT::Experimental::Detail::RDaosNTupleAnchor::Deserialize(const void *buffer)
{
   using namespace ROOT::Experimental::Internal::RNTupleSerialization;
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   bytes += DeserializeUInt32(bytes, &fVersion);
   bytes += DeserializeUInt32(bytes, &fNBytesHeader);
   bytes += DeserializeUInt32(bytes, &fLenHeader);
   bytes += DeserializeUInt32(bytes, &fNBytesFooter);
   bytes += DeserializeUInt32(bytes, &fLenFooter);
   bytes += DeserializeString(bytes, &fObjClass);
   return SerializeString(fObjClass, nullptr) + 20;
}

std::uint32_t
ROOT::Experimental::Detail::RDaosNTupleAnchor::GetSize()
{
   return RDaosNTupleAnchor().Serialize(nullptr)
      + ROOT::Experimental::Detail::RDaosObject::ObjClassId::kOCNameMaxLength;
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPageSinkDaos::RPageSinkDaos(std::string_view ntupleName, std::string_view uri,
   const RNTupleWriteOptions &options)
   : RPageSink(ntupleName, options)
   , fPageAllocator(std::make_unique<RPageAllocatorHeap>())
   , fURI(uri)
{
   R__LOG_WARNING(NTupleLog()) << "The DAOS backend is experimental and still under development. " <<
      "Do not store real data with this version of RNTuple!";
   fCompressor = std::make_unique<RNTupleCompressor>();
   EnableDefaultMetrics("RPageSinkDaos");
}


ROOT::Experimental::Detail::RPageSinkDaos::~RPageSinkDaos() = default;


void ROOT::Experimental::Detail::RPageSinkDaos::CreateImpl(const RNTupleModel & /* model */)
{
   auto opts = dynamic_cast<RNTupleWriteOptionsDaos *>(fOptions.get());
   fNTupleAnchor.fObjClass = opts ? opts->GetObjectClass() : RNTupleWriteOptionsDaos().GetObjectClass();
   auto oclass = RDaosObject::ObjClassId(fNTupleAnchor.fObjClass);
   if (oclass.IsUnknown())
      throw ROOT::Experimental::RException(R__FAIL("Unknown object class " + fNTupleAnchor.fObjClass));

   auto args = ParseDaosURI(fURI);
   auto pool = std::make_shared<RDaosPool>(args.fPoolUuid, args.fSvcReplicas);
   fDaosContainer = std::make_unique<RDaosContainer>(pool, args.fContainerUuid, /*create =*/ true);
   fDaosContainer->SetDefaultObjectClass(oclass);

   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szHeader = descriptor.GetHeaderSize();
   auto buffer = std::make_unique<unsigned char[]>(szHeader);
   descriptor.SerializeHeader(buffer.get());

   auto zipBuffer = std::make_unique<unsigned char[]>(szHeader);
   auto szZipHeader = fCompressor->Zip(buffer.get(), szHeader, GetWriteOptions().GetCompression(),
      [&zipBuffer](const void *b, size_t n, size_t o){ memcpy(zipBuffer.get() + o, b, n); } );
   WriteNTupleHeader(zipBuffer.get(), szZipHeader, szHeader);
}


ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkDaos::CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page)
{
   auto element = columnHandle.fColumn->GetElement();
   RPageStorage::RSealedPage sealedPage;
   {
      RNTupleAtomicTimer timer(fCounters->fTimeWallZip, fCounters->fTimeCpuZip);
      sealedPage = SealPage(page, *element, GetWriteOptions().GetCompression());
   }

   fCounters->fSzZip.Add(page.GetNBytes());
   return CommitSealedPageImpl(columnHandle.fId, sealedPage);
}


ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkDaos::CommitSealedPageImpl(
   DescriptorId_t /*columnId*/, const RPageStorage::RSealedPage &sealedPage)
{
   auto offsetData = fOid.fetch_add(1);
   {
      RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
      fDaosContainer->WriteSingleAkey(sealedPage.fBuffer, sealedPage.fSize,
                                      {offsetData, 0}, kDistributionKey, kAttributeKey);
   }

   RClusterDescriptor::RLocator result;
   result.fPosition = offsetData;
   result.fBytesOnStorage = sealedPage.fSize;
   fCounters->fNPageCommitted.Inc();
   fCounters->fSzWritePayload.Add(sealedPage.fSize);
   fNBytesCurrentCluster += sealedPage.fSize;
   return result;
}


std::uint64_t
ROOT::Experimental::Detail::RPageSinkDaos::CommitClusterImpl(ROOT::Experimental::NTupleSize_t /* nEntries */)
{
   return std::exchange(fNBytesCurrentCluster, 0);
}


void ROOT::Experimental::Detail::RPageSinkDaos::CommitDatasetImpl()
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szFooter = descriptor.GetFooterSize();
   auto buffer = std::make_unique<unsigned char []>(szFooter);
   descriptor.SerializeFooter(buffer.get());

   auto zipBuffer = std::make_unique<unsigned char []>(szFooter);
   auto szZipFooter = fCompressor->Zip(buffer.get(), szFooter, GetWriteOptions().GetCompression(),
      [&zipBuffer](const void *b, size_t n, size_t o){ memcpy(zipBuffer.get() + o, b, n); } );
   WriteNTupleFooter(zipBuffer.get(), szZipFooter, szFooter);
   WriteNTupleAnchor();
}


void ROOT::Experimental::Detail::RPageSinkDaos::WriteNTupleHeader(
		const void *data, size_t nbytes, size_t lenHeader)
{
   fDaosContainer->WriteSingleAkey(data, nbytes, kOidHeader, kDistributionKey,
                                   kAttributeKey, kCidMetadata);
   fNTupleAnchor.fLenHeader = lenHeader;
   fNTupleAnchor.fNBytesHeader = nbytes;
}

void ROOT::Experimental::Detail::RPageSinkDaos::WriteNTupleFooter(
		const void *data, size_t nbytes, size_t lenFooter)
{
   fDaosContainer->WriteSingleAkey(data, nbytes, kOidFooter, kDistributionKey,
                                   kAttributeKey, kCidMetadata);
   fNTupleAnchor.fLenFooter = lenFooter;
   fNTupleAnchor.fNBytesFooter = nbytes;
}

void ROOT::Experimental::Detail::RPageSinkDaos::WriteNTupleAnchor() {
   const auto ntplSize = RDaosNTupleAnchor::GetSize();
   auto buffer = std::make_unique<unsigned char[]>(ntplSize);
   fNTupleAnchor.Serialize(buffer.get());
   fDaosContainer->WriteSingleAkey(buffer.get(), ntplSize, kOidAnchor, kDistributionKey,
                                   kAttributeKey, kCidMetadata);
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSinkDaos::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   if (nElements == 0)
      throw RException(R__FAIL("invalid call: request empty page"));
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
   RPage newPage(columnId, mem, elementSize, nElements);
   newPage.GrowUnchecked(nElements);
   return newPage;
}

void ROOT::Experimental::Detail::RPageAllocatorDaos::DeletePage(const RPage& page)
{
   if (page.IsNull())
      return;
   delete[] reinterpret_cast<unsigned char *>(page.GetBuffer());
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPageSourceDaos::RPageSourceDaos(std::string_view ntupleName, std::string_view uri,
   const RNTupleReadOptions &options)
   : RPageSource(ntupleName, options)
   , fPageAllocator(std::make_unique<RPageAllocatorDaos>())
   , fPagePool(std::make_shared<RPagePool>())
   , fURI(uri)
   , fClusterPool(std::make_unique<RClusterPool>(*this))
{
   fDecompressor = std::make_unique<RNTupleDecompressor>();
   EnableDefaultMetrics("RPageSourceDaos");

   auto args = ParseDaosURI(uri);
   auto pool = std::make_shared<RDaosPool>(args.fPoolUuid, args.fSvcReplicas);
   fDaosContainer = std::make_unique<RDaosContainer>(pool, args.fContainerUuid);
}


ROOT::Experimental::Detail::RPageSourceDaos::~RPageSourceDaos() = default;


ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Detail::RPageSourceDaos::AttachImpl()
{
   RNTupleDescriptorBuilder descBuilder;
   RDaosNTupleAnchor ntpl;
   const auto ntplSize = RDaosNTupleAnchor::GetSize();
   auto buffer = std::make_unique<unsigned char[]>(ntplSize);
   fDaosContainer->ReadSingleAkey(buffer.get(), ntplSize, kOidAnchor, kDistributionKey,
                                  kAttributeKey, kCidMetadata);
   ntpl.Deserialize(buffer.get());

   auto oclass = RDaosObject::ObjClassId(ntpl.fObjClass);
   if (oclass.IsUnknown())
      throw ROOT::Experimental::RException(R__FAIL("Unknown object class " + ntpl.fObjClass));
   fDaosContainer->SetDefaultObjectClass(oclass);

   buffer = std::make_unique<unsigned char[]>(ntpl.fLenHeader);
   auto zipBuffer = std::make_unique<unsigned char[]>(ntpl.fNBytesHeader);
   fDaosContainer->ReadSingleAkey(zipBuffer.get(), ntpl.fNBytesHeader, kOidHeader, kDistributionKey,
                                  kAttributeKey, kCidMetadata);
   fDecompressor->Unzip(zipBuffer.get(), ntpl.fNBytesHeader, ntpl.fLenHeader, buffer.get());
   descBuilder.SetFromHeader(buffer.get());

   buffer = std::make_unique<unsigned char[]>(ntpl.fLenFooter);
   zipBuffer = std::make_unique<unsigned char[]>(ntpl.fNBytesFooter);
   fDaosContainer->ReadSingleAkey(zipBuffer.get(), ntpl.fNBytesFooter, kOidFooter, kDistributionKey,
                                  kAttributeKey, kCidMetadata);
   fDecompressor->Unzip(zipBuffer.get(), ntpl.fNBytesFooter, ntpl.fLenFooter, buffer.get());
   descBuilder.AddClustersFromFooter(buffer.get());

   return descBuilder.MoveDescriptor();
}


std::string ROOT::Experimental::Detail::RPageSourceDaos::GetObjectClass() const
{
   return fDaosContainer->GetDefaultObjectClass().ToString();
}


void ROOT::Experimental::Detail::RPageSourceDaos::LoadSealedPage(
   DescriptorId_t columnId, const RClusterIndex &clusterIndex, RSealedPage &sealedPage)
{
   const auto clusterId = clusterIndex.GetClusterId();
   const auto &clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);

   auto pageInfo = clusterDescriptor.GetPageRange(columnId).Find(clusterIndex.GetIndex());

   const auto bytesOnStorage = pageInfo.fLocator.fBytesOnStorage;
   sealedPage.fSize = bytesOnStorage;
   sealedPage.fNElements = pageInfo.fNElements;
   if (sealedPage.fBuffer) {
      fDaosContainer->ReadSingleAkey(const_cast<void *>(sealedPage.fBuffer), bytesOnStorage,
                                     {static_cast<decltype(daos_obj_id_t::lo)>(pageInfo.fLocator.fPosition), 0},
                                     kDistributionKey, kAttributeKey);
   }
}

ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceDaos::PopulatePageFromCluster(
   ColumnHandle_t columnHandle, const RClusterDescriptor &clusterDescriptor, ClusterSize_t::ValueType idxInCluster)
{
   const auto columnId = columnHandle.fId;
   const auto clusterId = clusterDescriptor.GetId();

   auto pageInfo = clusterDescriptor.GetPageRange(columnId).Find(idxInCluster);

   const auto element = columnHandle.fColumn->GetElement();
   const auto elementSize = element->GetSize();
   const auto bytesOnStorage = pageInfo.fLocator.fBytesOnStorage;

   const void *sealedPageBuffer = nullptr; // points either to directReadBuffer or to a read-only page in the cluster
   std::unique_ptr<unsigned char []> directReadBuffer; // only used if cluster pool is turned off

   if (fOptions.GetClusterCache() == RNTupleReadOptions::EClusterCache::kOff) {
      directReadBuffer = std::make_unique<unsigned char[]>(bytesOnStorage);
      fDaosContainer->ReadSingleAkey(directReadBuffer.get(), bytesOnStorage,
                                     {static_cast<decltype(daos_obj_id_t::lo)>(pageInfo.fLocator.fPosition), 0},
                                     kDistributionKey, kAttributeKey);
      fCounters->fNPageLoaded.Inc();
      fCounters->fNRead.Inc();
      fCounters->fSzReadPayload.Add(bytesOnStorage);
      sealedPageBuffer = directReadBuffer.get();
   } else {
      if (!fCurrentCluster || (fCurrentCluster->GetId() != clusterId) || !fCurrentCluster->ContainsColumn(columnId))
         fCurrentCluster = fClusterPool->GetCluster(clusterId, fActiveColumns);
      R__ASSERT(fCurrentCluster->ContainsColumn(columnId));

      auto cachedPage = fPagePool->GetPage(columnId, RClusterIndex(clusterId, idxInCluster));
      if (!cachedPage.IsNull())
         return cachedPage;

      ROnDiskPage::Key key(columnId, pageInfo.fPageNo);
      auto onDiskPage = fCurrentCluster->GetOnDiskPage(key);
      R__ASSERT(onDiskPage && (bytesOnStorage == onDiskPage->GetSize()));
      sealedPageBuffer = onDiskPage->GetAddress();
   }

   std::unique_ptr<unsigned char []> pageBuffer;
   {
      RNTupleAtomicTimer timer(fCounters->fTimeWallUnzip, fCounters->fTimeCpuUnzip);
      pageBuffer = UnsealPage({sealedPageBuffer, bytesOnStorage, pageInfo.fNElements}, *element);
      fCounters->fSzUnzip.Add(elementSize * pageInfo.fNElements);
   }

   const auto indexOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   auto newPage = fPageAllocator->NewPage(columnId, pageBuffer.release(), elementSize, pageInfo.fNElements);
   newPage.SetWindow(indexOffset + pageInfo.fFirstInPage, RPage::RClusterInfo(clusterId, indexOffset));
   fPagePool->RegisterPage(newPage,
      RPageDeleter([](const RPage &page, void * /*userData*/)
      {
         RPageAllocatorDaos::DeletePage(page);
      }, nullptr));
   fCounters->fNPagePopulated.Inc();
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
   const auto idxInCluster = clusterIndex.GetIndex();
   const auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, clusterIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   R__ASSERT(clusterId != kInvalidDescriptorId);
   const auto &clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
   return PopulatePageFromCluster(columnHandle, clusterDescriptor, idxInCluster);
}

void ROOT::Experimental::Detail::RPageSourceDaos::ReleasePage(RPage &page)
{
   fPagePool->ReturnPage(page);
}

std::unique_ptr<ROOT::Experimental::Detail::RPageSource> ROOT::Experimental::Detail::RPageSourceDaos::Clone() const
{
   auto clone = new RPageSourceDaos(fNTupleName, fURI, fOptions);
   return std::unique_ptr<RPageSourceDaos>(clone);
}

std::unique_ptr<ROOT::Experimental::Detail::RCluster>
ROOT::Experimental::Detail::RPageSourceDaos::LoadCluster(DescriptorId_t clusterId, const ColumnSet_t &columns)
{
   fCounters->fNClusterLoaded.Inc();

   const auto &clusterDesc = GetDescriptor().GetClusterDescriptor(clusterId);
   auto clusterLocator = clusterDesc.GetLocator();

   struct RDaosSealedPageLocator {
      RDaosSealedPageLocator() = default;
      RDaosSealedPageLocator(DescriptorId_t c, NTupleSize_t p, std::uint64_t o, std::uint64_t s, std::size_t b)
         : fColumnId(c), fPageNo(p), fObjectId(o), fSize(s), fBufPos(b) {}
      DescriptorId_t fColumnId = 0;
      NTupleSize_t fPageNo = 0;
      std::uint64_t fObjectId = 0;
      std::uint64_t fSize = 0;
      std::size_t fBufPos = 0;
   };

   // Collect the page necessary page meta-data and sum up the total size of the compressed and packed pages
   std::vector<RDaosSealedPageLocator> onDiskPages;
   std::size_t szPayload = 0;
   for (auto columnId : columns) {
      const auto &pageRange = clusterDesc.GetPageRange(columnId);
      NTupleSize_t pageNo = 0;
      for (const auto &pageInfo : pageRange.fPageInfos) {
         const auto &pageLocator = pageInfo.fLocator;
         onDiskPages.emplace_back(RDaosSealedPageLocator(
            columnId, pageNo, pageLocator.fPosition, pageLocator.fBytesOnStorage, szPayload));
         szPayload += pageLocator.fBytesOnStorage;
         ++pageNo;
      }
   }

   // Prepare the input vector for the RDaosContainer::ReadV() call
   std::vector<RDaosContainer::RWOperation> readRequests;
   auto buffer = new unsigned char[szPayload];
   for (auto &s : onDiskPages) {
      std::vector<d_iov_t> iovs(1);
      d_iov_set(&iovs[0], buffer + s.fBufPos, s.fSize);
      readRequests.emplace_back(daos_obj_id_t{s.fObjectId, 0},
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


void ROOT::Experimental::Detail::RPageSourceDaos::UnzipClusterImpl(RCluster *cluster)
{
   RNTupleAtomicTimer timer(fCounters->fTimeWallUnzip, fCounters->fTimeCpuUnzip);
   fTaskScheduler->Reset();

   const auto clusterId = cluster->GetId();
   const auto &clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);

   std::vector<std::unique_ptr<RColumnElementBase>> allElements;

   const auto &columnsInCluster = cluster->GetAvailColumns();
   for (const auto columnId : columnsInCluster) {
      const auto &columnDesc = fDescriptor.GetColumnDescriptor(columnId);

      allElements.emplace_back(RColumnElementBase::Generate(columnDesc.GetModel().GetType()));

      const auto &pageRange = clusterDescriptor.GetPageRange(columnId);
      std::uint64_t pageNo = 0;
      std::uint64_t firstInPage = 0;
      for (const auto &pi : pageRange.fPageInfos) {
         ROnDiskPage::Key key(columnId, pageNo);
         auto onDiskPage = cluster->GetOnDiskPage(key);
         R__ASSERT(onDiskPage && (onDiskPage->GetSize() == pi.fLocator.fBytesOnStorage));

         auto taskFunc =
            [this, columnId, clusterId, firstInPage, onDiskPage,
             element = allElements.back().get(),
             nElements = pi.fNElements,
             indexOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex
            ] () {
               auto pageBuffer = UnsealPage({onDiskPage->GetAddress(), onDiskPage->GetSize(), nElements}, *element);
               fCounters->fSzUnzip.Add(element->GetSize() * nElements);

               auto newPage = fPageAllocator->NewPage(columnId, pageBuffer.release(), element->GetSize(), nElements);
               newPage.SetWindow(indexOffset + firstInPage, RPage::RClusterInfo(clusterId, indexOffset));
               fPagePool->PreloadPage(newPage,
                  RPageDeleter([](const RPage &page, void * /*userData*/)
                  {
                     RPageAllocatorDaos::DeletePage(page);
                  }, nullptr));
            };

         fTaskScheduler->AddTask(taskFunc);

         firstInPage += pi.fNElements;
         pageNo++;
      } // for all pages in column
   } // for all columns in cluster

   fCounters->fNPagePopulated.Add(cluster->GetNOnDiskPages());

   fTaskScheduler->Wait();
}
