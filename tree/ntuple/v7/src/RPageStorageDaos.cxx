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
#include <ROOT/RNTupleSerialize.hxx>
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
#include <limits>
#include <utility>
#include <regex>

namespace {
using AttributeKey_t = ROOT::Experimental::Detail::RDaosContainer::AttributeKey_t;
using DistributionKey_t = ROOT::Experimental::Detail::RDaosContainer::DistributionKey_t;

/// \brief RNTuple page-DAOS mappings
enum EDaosMapping { kOidPerCluster, kOidPerPage };

struct RDaosKey {
   daos_obj_id_t fOid;
   DistributionKey_t fDkey;
   AttributeKey_t fAkey;
};

/// \brief Pre-defined keys for object store. `kDistributionKeyDefault` is the distribution key for metadata and
/// pagelist values; optionally it can be used for ntuple pages (if under the `kOidPerPage` mapping strategy).
/// `kAttributeKeyDefault` is the attribute key for ntuple pages under `kOidPerPage`.
/// `kAttributeKey{Anchor,Header,Footer}` are the respective attribute keys for anchor/header/footer metadata elements.
static constexpr DistributionKey_t kDistributionKeyDefault = 0x5a3c69f0cafe4a11;
static constexpr AttributeKey_t kAttributeKeyDefault = 0x4243544b53444229;
static constexpr AttributeKey_t kAttributeKeyAnchor = 0x4243544b5344422a;
static constexpr AttributeKey_t kAttributeKeyHeader = 0x4243544b5344422b;
static constexpr AttributeKey_t kAttributeKeyFooter = 0x4243544b5344422c;

/// \brief Pre-defined 64 LSb of the OIDs for ntuple metadata (holds anchor/header/footer) and clusters' pagelists.
static constexpr decltype(daos_obj_id_t::lo) kOidLowMetadata = -1;
static constexpr decltype(daos_obj_id_t::lo) kOidLowPageList = -2;

static constexpr daos_oclass_id_t kCidMetadata = OC_SX;

static constexpr EDaosMapping kDefaultDaosMapping = kOidPerCluster;

template <EDaosMapping mapping>
RDaosKey GetPageDaosKey(ROOT::Experimental::Detail::ntuple_index_t ntplId, long unsigned clusterId,
                        long unsigned columnId, long unsigned pageCount)
{
   if constexpr (mapping == kOidPerCluster) {
      return RDaosKey{daos_obj_id_t{static_cast<decltype(daos_obj_id_t::lo)>(clusterId),
                                    static_cast<decltype(daos_obj_id_t::hi)>(ntplId)},
                      static_cast<DistributionKey_t>(columnId), static_cast<AttributeKey_t>(pageCount)};
   } else if constexpr (mapping == kOidPerPage) {
      return RDaosKey{daos_obj_id_t{static_cast<decltype(daos_obj_id_t::lo)>(pageCount),
                                    static_cast<decltype(daos_obj_id_t::hi)>(ntplId)},
                      kDistributionKeyDefault, kAttributeKeyDefault};
   }
}

struct RDaosURI {
   /// \brief Label of the DAOS pool
   std::string fPoolLabel;
   /// \brief Label of the container for this RNTuple
   std::string fContainerLabel;
};

/**
  \brief Parse a DAOS RNTuple URI of the form 'daos://pool_id/container_id'.
*/
RDaosURI ParseDaosURI(std::string_view uri)
{
   std::regex re("daos://([^/]+)/(.+)");
   std::cmatch m;
   if (!std::regex_match(uri.data(), m, re))
      throw ROOT::Experimental::RException(R__FAIL("Invalid DAOS pool URI."));
   return {m[1], m[2]};
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

std::uint32_t ROOT::Experimental::Detail::RDaosNTupleAnchor::Serialize(void *buffer) const
{
   using RNTupleSerializer = ROOT::Experimental::Internal::RNTupleSerializer;
   if (buffer != nullptr) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);
      bytes += RNTupleSerializer::SerializeUInt32(fVersion, bytes);
      bytes += RNTupleSerializer::SerializeUInt32(fNBytesHeader, bytes);
      bytes += RNTupleSerializer::SerializeUInt32(fLenHeader, bytes);
      bytes += RNTupleSerializer::SerializeUInt32(fNBytesFooter, bytes);
      bytes += RNTupleSerializer::SerializeUInt32(fLenFooter, bytes);
      bytes += RNTupleSerializer::SerializeString(fObjClass, bytes);
   }
   return RNTupleSerializer::SerializeString(fObjClass, nullptr) + 20;
}

ROOT::Experimental::RResult<std::uint32_t>
ROOT::Experimental::Detail::RDaosNTupleAnchor::Deserialize(const void *buffer, std::uint32_t bufSize)
{
   if (bufSize < 20)
      return R__FAIL("DAOS anchor too short");

   using RNTupleSerializer = ROOT::Experimental::Internal::RNTupleSerializer;
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fVersion);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fNBytesHeader);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fLenHeader);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fNBytesFooter);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fLenFooter);
   auto result = RNTupleSerializer::DeserializeString(bytes, bufSize - 20, fObjClass);
   if (!result)
      return R__FORWARD_ERROR(result);
   return result.Unwrap() + 20;
}

std::uint32_t ROOT::Experimental::Detail::RDaosNTupleAnchor::GetSize()
{
   return RDaosNTupleAnchor().Serialize(nullptr) +
          ROOT::Experimental::Detail::RDaosObject::ObjClassId::kOCNameMaxLength;
}

int ROOT::Experimental::Detail::RDaosContainerNTupleLocator::InitNTupleDescriptorBuilder(
   ROOT::Experimental::Detail::RDaosContainer &cont, ROOT::Experimental::Detail::RNTupleDecompressor &decompressor,
   RNTupleDescriptorBuilder &builder)
{
   std::unique_ptr<unsigned char[]> buffer, zipBuffer;
   auto &anchor = fAnchor.emplace();
   int err;

   const auto anchorSize = ROOT::Experimental::Detail::RDaosNTupleAnchor::GetSize();
   daos_obj_id_t oidMetadata{kOidLowMetadata, static_cast<decltype(daos_obj_id_t::hi)>(this->GetIndex())};

   buffer = std::make_unique<unsigned char[]>(anchorSize);
   if ((err = cont.ReadSingleAkey(buffer.get(), anchorSize, oidMetadata, kDistributionKeyDefault, kAttributeKeyAnchor,
                                  kCidMetadata)))
      return err;

   anchor.Deserialize(buffer.get(), anchorSize).Unwrap();

   builder.SetOnDiskHeaderSize(anchor.fNBytesHeader);
   buffer = std::make_unique<unsigned char[]>(anchor.fLenHeader);
   zipBuffer = std::make_unique<unsigned char[]>(anchor.fNBytesHeader);
   if ((err = cont.ReadSingleAkey(zipBuffer.get(), anchor.fNBytesHeader, oidMetadata, kDistributionKeyDefault,
                                  kAttributeKeyHeader, kCidMetadata)))
      return err;
   decompressor.Unzip(zipBuffer.get(), anchor.fNBytesHeader, anchor.fLenHeader, buffer.get());
   ROOT::Experimental::Internal::RNTupleSerializer::DeserializeHeaderV1(buffer.get(), anchor.fLenHeader, builder);

   builder.AddToOnDiskFooterSize(anchor.fNBytesFooter);
   buffer = std::make_unique<unsigned char[]>(anchor.fLenFooter);
   zipBuffer = std::make_unique<unsigned char[]>(anchor.fNBytesFooter);
   if ((err = cont.ReadSingleAkey(zipBuffer.get(), anchor.fNBytesFooter, oidMetadata, kDistributionKeyDefault,
                                  kAttributeKeyFooter, kCidMetadata)))
      return err;
   decompressor.Unzip(zipBuffer.get(), anchor.fNBytesFooter, anchor.fLenFooter, buffer.get());
   ROOT::Experimental::Internal::RNTupleSerializer::DeserializeFooterV1(buffer.get(), anchor.fLenFooter, builder);

   return 0;
}

std::pair<ROOT::Experimental::Detail::RDaosContainerNTupleLocator, ROOT::Experimental::RNTupleDescriptorBuilder>
ROOT::Experimental::Detail::RDaosContainerNTupleLocator::LocateNTuple(RDaosContainer &cont,
                                                                      const std::string &ntupleName,
                                                                      RNTupleDecompressor &decompressor)
{
   auto result = std::make_pair(RDaosContainerNTupleLocator(ntupleName), RNTupleDescriptorBuilder());

   auto &loc = result.first;
   auto &builder = result.second;

   if (int err = loc.InitNTupleDescriptorBuilder(cont, decompressor, builder); !err) {
      if (ntupleName.empty() || ntupleName != builder.GetDescriptor().GetName()) {
         // Hash already taken by a differently-named ntuple.
         throw ROOT::Experimental::RException(
            R__FAIL("LocateNTuple: ntuple name '" + ntupleName + "' unavailable in this container."));
      }
   }
   return result;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::Detail::RPageSinkDaos::RPageSinkDaos(std::string_view ntupleName, std::string_view uri,
                                                         const RNTupleWriteOptions &options)
   : RPageSink(ntupleName, options), fPageAllocator(std::make_unique<RPageAllocatorHeap>()), fURI(uri)
{
   R__LOG_WARNING(NTupleLog()) << "The DAOS backend is experimental and still under development. "
                               << "Do not store real data with this version of RNTuple!";
   fCompressor = std::make_unique<RNTupleCompressor>();
   EnableDefaultMetrics("RPageSinkDaos");
}

ROOT::Experimental::Detail::RPageSinkDaos::~RPageSinkDaos() = default;

void ROOT::Experimental::Detail::RPageSinkDaos::CreateImpl(const RNTupleModel & /* model */,
                                                           unsigned char *serializedHeader, std::uint32_t length)
{
   auto opts = dynamic_cast<RNTupleWriteOptionsDaos *>(fOptions.get());
   fNTupleAnchor.fObjClass = opts ? opts->GetObjectClass() : RNTupleWriteOptionsDaos().GetObjectClass();
   auto oclass = RDaosObject::ObjClassId(fNTupleAnchor.fObjClass);
   if (oclass.IsUnknown())
      throw ROOT::Experimental::RException(R__FAIL("Unknown object class " + fNTupleAnchor.fObjClass));

   auto args = ParseDaosURI(fURI);
   auto pool = std::make_shared<RDaosPool>(args.fPoolLabel);
   fDaosContainer = std::make_unique<RDaosContainer>(pool, args.fContainerLabel, /*create =*/true);
   fDaosContainer->SetDefaultObjectClass(oclass);

   RNTupleDecompressor decompressor;
   auto [locator, _] = RDaosContainerNTupleLocator::LocateNTuple(*fDaosContainer, fNTupleName, decompressor);
   fNTupleIndex = locator.GetIndex();

   auto zipBuffer = std::make_unique<unsigned char[]>(length);
   auto szZipHeader = fCompressor->Zip(serializedHeader, length, GetWriteOptions().GetCompression(),
                                       RNTupleCompressor::MakeMemCopyWriter(zipBuffer.get()));
   WriteNTupleHeader(zipBuffer.get(), szZipHeader, length);
}

ROOT::Experimental::RNTupleLocator
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

ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Detail::RPageSinkDaos::CommitSealedPageImpl(DescriptorId_t columnId,
                                                                const RPageStorage::RSealedPage &sealedPage)
{
   auto offsetData = fPageId.fetch_add(1);
   DescriptorId_t clusterId = fDescriptorBuilder.GetDescriptor().GetNClusters();

   {
      RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
      RDaosKey daosKey = GetPageDaosKey<kDefaultDaosMapping>(fNTupleIndex, clusterId, columnId, offsetData);
      fDaosContainer->WriteSingleAkey(sealedPage.fBuffer, sealedPage.fSize, daosKey.fOid, daosKey.fDkey, daosKey.fAkey);
   }

   RNTupleLocator result;
   result.fPosition = offsetData;
   result.fBytesOnStorage = sealedPage.fSize;
   fCounters->fNPageCommitted.Inc();
   fCounters->fSzWritePayload.Add(sealedPage.fSize);
   fNBytesCurrentCluster += sealedPage.fSize;
   return result;
}

std::vector<ROOT::Experimental::RNTupleLocator>
ROOT::Experimental::Detail::RPageSinkDaos::CommitSealedPageVImpl(std::span<RPageStorage::RSealedPageGroup> ranges)
{
   RDaosContainer::MultiObjectRWOperation_t writeRequests;
   std::vector<ROOT::Experimental::RNTupleLocator> locators;
   size_t nPages =
      std::accumulate(ranges.begin(), ranges.end(), 0, [](size_t c, const RPageStorage::RSealedPageGroup &r) {
         return c + std::distance(r.fFirst, r.fLast);
      });
   locators.reserve(nPages);

   DescriptorId_t clusterId = fDescriptorBuilder.GetDescriptor().GetNClusters();
   std::size_t szPayload = 0;

   /// Aggregate batch of requests by object ID and distribution key, determined by the ntuple-DAOS mapping
   for (auto &range : ranges) {
      for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt) {
         const RPageStorage::RSealedPage &s = *sealedPageIt;
         d_iov_t pageIov;
         d_iov_set(&pageIov, const_cast<void *>(s.fBuffer), s.fSize);
         auto offsetData = fPageId.fetch_add(1);

         RDaosKey daosKey = GetPageDaosKey<kDefaultDaosMapping>(fNTupleIndex, clusterId, range.fColumnId, offsetData);
         auto odPair = RDaosContainer::ROidDkeyPair{daosKey.fOid, daosKey.fDkey};
         auto [it, ret] = writeRequests.emplace(odPair, RDaosContainer::RWOperation(odPair));
         it->second.insert(daosKey.fAkey, pageIov);

         RNTupleLocator locator;
         locator.fPosition = offsetData;
         locator.fBytesOnStorage = s.fSize;
         locators.push_back(locator);

         szPayload += s.fSize;
      }
   }
   fNBytesCurrentCluster += szPayload;

   {
      RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
      if (int err = fDaosContainer->WriteV(writeRequests))
         throw ROOT::Experimental::RException(R__FAIL("WriteV: error" + std::string(d_errstr(err))));
   }

   fCounters->fNPageCommitted.Add(nPages);
   fCounters->fSzWritePayload.Add(szPayload);

   return locators;
}

std::uint64_t
ROOT::Experimental::Detail::RPageSinkDaos::CommitClusterImpl(ROOT::Experimental::NTupleSize_t /* nEntries */)
{
   return std::exchange(fNBytesCurrentCluster, 0);
}

ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Detail::RPageSinkDaos::CommitClusterGroupImpl(unsigned char *serializedPageList,
                                                                  std::uint32_t length)
{
   auto bufPageListZip = std::make_unique<unsigned char[]>(length);
   auto szPageListZip = fCompressor->Zip(serializedPageList, length, GetWriteOptions().GetCompression(),
                                         RNTupleCompressor::MakeMemCopyWriter(bufPageListZip.get()));

   auto offsetData = fClusterGroupId.fetch_add(1);
   fDaosContainer->WriteSingleAkey(
      bufPageListZip.get(), szPageListZip,
      daos_obj_id_t{kOidLowPageList, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)}, kDistributionKeyDefault,
      offsetData, kCidMetadata);
   RNTupleLocator result;
   result.fPosition = offsetData;
   result.fBytesOnStorage = szPageListZip;
   fCounters->fSzWritePayload.Add(szPageListZip);
   return result;
}

void ROOT::Experimental::Detail::RPageSinkDaos::CommitDatasetImpl(unsigned char *serializedFooter, std::uint32_t length)
{
   auto bufFooterZip = std::make_unique<unsigned char[]>(length);
   auto szFooterZip = fCompressor->Zip(serializedFooter, length, GetWriteOptions().GetCompression(),
                                       RNTupleCompressor::MakeMemCopyWriter(bufFooterZip.get()));
   WriteNTupleFooter(bufFooterZip.get(), szFooterZip, length);
   WriteNTupleAnchor();
}

void ROOT::Experimental::Detail::RPageSinkDaos::WriteNTupleHeader(const void *data, size_t nbytes, size_t lenHeader)
{
   fDaosContainer->WriteSingleAkey(
      data, nbytes, daos_obj_id_t{kOidLowMetadata, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)},
      kDistributionKeyDefault, kAttributeKeyHeader, kCidMetadata);
   fNTupleAnchor.fLenHeader = lenHeader;
   fNTupleAnchor.fNBytesHeader = nbytes;
}

void ROOT::Experimental::Detail::RPageSinkDaos::WriteNTupleFooter(const void *data, size_t nbytes, size_t lenFooter)
{
   fDaosContainer->WriteSingleAkey(
      data, nbytes, daos_obj_id_t{kOidLowMetadata, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)},
      kDistributionKeyDefault, kAttributeKeyFooter, kCidMetadata);
   fNTupleAnchor.fLenFooter = lenFooter;
   fNTupleAnchor.fNBytesFooter = nbytes;
}

void ROOT::Experimental::Detail::RPageSinkDaos::WriteNTupleAnchor()
{
   const auto ntplSize = RDaosNTupleAnchor::GetSize();
   auto buffer = std::make_unique<unsigned char[]>(ntplSize);
   fNTupleAnchor.Serialize(buffer.get());
   fDaosContainer->WriteSingleAkey(
      buffer.get(), ntplSize, daos_obj_id_t{kOidLowMetadata, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)},
      kDistributionKeyDefault, kAttributeKeyAnchor, kCidMetadata);
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

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageAllocatorDaos::NewPage(ColumnId_t columnId, void *mem, std::size_t elementSize,
                                                        std::size_t nElements)
{
   RPage newPage(columnId, mem, elementSize, nElements);
   newPage.GrowUnchecked(nElements);
   return newPage;
}

void ROOT::Experimental::Detail::RPageAllocatorDaos::DeletePage(const RPage &page)
{
   if (page.IsNull())
      return;
   delete[] reinterpret_cast<unsigned char *>(page.GetBuffer());
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::Detail::RPageSourceDaos::RPageSourceDaos(std::string_view ntupleName, std::string_view uri,
                                                             const RNTupleReadOptions &options)
   : RPageSource(ntupleName, options), fPageAllocator(std::make_unique<RPageAllocatorDaos>()),
     fPagePool(std::make_shared<RPagePool>()), fURI(uri),
     fClusterPool(std::make_unique<RClusterPool>(*this, options.GetClusterBunchSize()))
{
   fDecompressor = std::make_unique<RNTupleDecompressor>();
   EnableDefaultMetrics("RPageSourceDaos");

   auto args = ParseDaosURI(uri);
   auto pool = std::make_shared<RDaosPool>(args.fPoolLabel);
   fDaosContainer = std::make_unique<RDaosContainer>(pool, args.fContainerLabel);
}

ROOT::Experimental::Detail::RPageSourceDaos::~RPageSourceDaos() = default;

ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Detail::RPageSourceDaos::AttachImpl()
{
   ROOT::Experimental::RNTupleDescriptor ntplDesc;
   std::unique_ptr<unsigned char[]> buffer, zipBuffer;

   auto [locator, descBuilder] =
      RDaosContainerNTupleLocator::LocateNTuple(*fDaosContainer, fNTupleName, *fDecompressor);
   if (!locator.IsValid())
      throw ROOT::Experimental::RException(
         R__FAIL("Attach: requested ntuple '" + fNTupleName + "' is not present in DAOS container."));

   auto oclass = RDaosObject::ObjClassId(locator.fAnchor->fObjClass);
   if (oclass.IsUnknown())
      throw ROOT::Experimental::RException(R__FAIL("Attach: unknown object class " + locator.fAnchor->fObjClass));

   fDaosContainer->SetDefaultObjectClass(oclass);
   fNTupleIndex = locator.GetIndex();

   ntplDesc = descBuilder.MoveDescriptor();
   daos_obj_id_t oidPageList{kOidLowPageList, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)};

   for (const auto &cgDesc : ntplDesc.GetClusterGroupIterable()) {
      buffer = std::make_unique<unsigned char[]>(cgDesc.GetPageListLength());
      zipBuffer = std::make_unique<unsigned char[]>(cgDesc.GetPageListLocator().fBytesOnStorage);
      fDaosContainer->ReadSingleAkey(zipBuffer.get(), cgDesc.GetPageListLocator().fBytesOnStorage, oidPageList,
                                     kDistributionKeyDefault, cgDesc.GetPageListLocator().GetPosition<std::uint64_t>(),
                                     kCidMetadata);
      fDecompressor->Unzip(zipBuffer.get(), cgDesc.GetPageListLocator().fBytesOnStorage, cgDesc.GetPageListLength(),
                           buffer.get());

      auto clusters = RClusterGroupDescriptorBuilder::GetClusterSummaries(ntplDesc, cgDesc.GetId());
      Internal::RNTupleSerializer::DeserializePageListV1(buffer.get(), cgDesc.GetPageListLength(), clusters);
      for (std::size_t i = 0; i < clusters.size(); ++i) {
         ntplDesc.AddClusterDetails(clusters[i].MoveDescriptor().Unwrap());
      }
   }

   return ntplDesc;
}

std::string ROOT::Experimental::Detail::RPageSourceDaos::GetObjectClass() const
{
   return fDaosContainer->GetDefaultObjectClass().ToString();
}

void ROOT::Experimental::Detail::RPageSourceDaos::LoadSealedPage(DescriptorId_t columnId,
                                                                 const RClusterIndex &clusterIndex,
                                                                 RSealedPage &sealedPage)
{
   const auto clusterId = clusterIndex.GetClusterId();

   RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   {
      auto descriptorGuard = GetSharedDescriptorGuard();
      const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterId);
      pageInfo = clusterDescriptor.GetPageRange(columnId).Find(clusterIndex.GetIndex());
   }

   const auto bytesOnStorage = pageInfo.fLocator.fBytesOnStorage;
   sealedPage.fSize = bytesOnStorage;
   sealedPage.fNElements = pageInfo.fNElements;
   if (sealedPage.fBuffer) {
      RDaosKey daosKey = GetPageDaosKey<kDefaultDaosMapping>(fNTupleIndex, clusterId, columnId,
                                                             pageInfo.fLocator.GetPosition<std::uint64_t>());
      fDaosContainer->ReadSingleAkey(const_cast<void *>(sealedPage.fBuffer), bytesOnStorage, daosKey.fOid,
                                     daosKey.fDkey, daosKey.fAkey);
   }
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSourceDaos::PopulatePageFromCluster(ColumnHandle_t columnHandle,
                                                                     const RClusterInfo &clusterInfo,
                                                                     ClusterSize_t::ValueType idxInCluster)
{
   const auto columnId = columnHandle.fId;
   const auto clusterId = clusterInfo.fClusterId;
   const auto &pageInfo = clusterInfo.fPageInfo;

   const auto element = columnHandle.fColumn->GetElement();
   const auto elementSize = element->GetSize();
   const auto bytesOnStorage = pageInfo.fLocator.fBytesOnStorage;

   const void *sealedPageBuffer = nullptr; // points either to directReadBuffer or to a read-only page in the cluster
   std::unique_ptr<unsigned char[]> directReadBuffer; // only used if cluster pool is turned off

   if (fOptions.GetClusterCache() == RNTupleReadOptions::EClusterCache::kOff) {
      directReadBuffer = std::make_unique<unsigned char[]>(bytesOnStorage);
      RDaosKey daosKey = GetPageDaosKey<kDefaultDaosMapping>(fNTupleIndex, clusterId, columnId,
                                                             pageInfo.fLocator.GetPosition<std::uint64_t>());
      fDaosContainer->ReadSingleAkey(directReadBuffer.get(), bytesOnStorage, daosKey.fOid, daosKey.fDkey,
                                     daosKey.fAkey);
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

   std::unique_ptr<unsigned char[]> pageBuffer;
   {
      RNTupleAtomicTimer timer(fCounters->fTimeWallUnzip, fCounters->fTimeCpuUnzip);
      pageBuffer = UnsealPage({sealedPageBuffer, bytesOnStorage, pageInfo.fNElements}, *element);
      fCounters->fSzUnzip.Add(elementSize * pageInfo.fNElements);
   }

   auto newPage = fPageAllocator->NewPage(columnId, pageBuffer.release(), elementSize, pageInfo.fNElements);
   newPage.SetWindow(clusterInfo.fColumnOffset + pageInfo.fFirstInPage,
                     RPage::RClusterInfo(clusterId, clusterInfo.fColumnOffset));
   fPagePool->RegisterPage(
      newPage,
      RPageDeleter([](const RPage &page, void * /*userData*/) { RPageAllocatorDaos::DeletePage(page); }, nullptr));
   fCounters->fNPagePopulated.Inc();
   return newPage;
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSourceDaos::PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex)
{
   const auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, globalIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   std::uint64_t idxInCluster;
   RClusterInfo clusterInfo;
   {
      auto descriptorGuard = GetSharedDescriptorGuard();
      clusterInfo.fClusterId = descriptorGuard->FindClusterId(columnId, globalIndex);
      R__ASSERT(clusterInfo.fClusterId != kInvalidDescriptorId);

      const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterInfo.fClusterId);
      clusterInfo.fColumnOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
      R__ASSERT(clusterInfo.fColumnOffset <= globalIndex);
      idxInCluster = globalIndex - clusterInfo.fColumnOffset;
      clusterInfo.fPageInfo = clusterDescriptor.GetPageRange(columnId).Find(idxInCluster);
   }
   return PopulatePageFromCluster(columnHandle, clusterInfo, idxInCluster);
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSourceDaos::PopulatePage(ColumnHandle_t columnHandle,
                                                          const RClusterIndex &clusterIndex)
{
   const auto clusterId = clusterIndex.GetClusterId();
   const auto idxInCluster = clusterIndex.GetIndex();
   const auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, clusterIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   R__ASSERT(clusterId != kInvalidDescriptorId);
   RClusterInfo clusterInfo;
   {
      auto descriptorGuard = GetSharedDescriptorGuard();
      const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterId);
      clusterInfo.fClusterId = clusterId;
      clusterInfo.fColumnOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
      clusterInfo.fPageInfo = clusterDescriptor.GetPageRange(columnId).Find(idxInCluster);
   }

   return PopulatePageFromCluster(columnHandle, clusterInfo, idxInCluster);
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

std::vector<std::unique_ptr<ROOT::Experimental::Detail::RCluster>>
ROOT::Experimental::Detail::RPageSourceDaos::LoadClusters(std::span<RCluster::RKey> clusterKeys)
{
   std::vector<std::unique_ptr<ROOT::Experimental::Detail::RCluster>> result;

   struct RDaosSealedPageLocator {
      RDaosSealedPageLocator() = default;
      RDaosSealedPageLocator(DescriptorId_t cl, DescriptorId_t co, NTupleSize_t p, std::uint64_t o, std::uint64_t s,
                             std::size_t b)
         : fClusterId(cl), fColumnId(co), fPageNo(p), fObjectId(o), fSize(s), fBufPos(b)
      {
      }
      DescriptorId_t fClusterId = 0;
      DescriptorId_t fColumnId = 0;
      NTupleSize_t fPageNo = 0;
      std::uint64_t fObjectId = 0;
      std::uint64_t fSize = 0;
      std::size_t fBufPos = 0;
   };

   std::vector<unsigned char *> clusterBuffers(clusterKeys.size());
   std::vector<std::unique_ptr<ROnDiskPageMapHeap>> pageMaps(clusterKeys.size());
   RDaosContainer::MultiObjectRWOperation_t readRequests;

   std::size_t szPayload = 0;
   unsigned nPages = 0;

   for (unsigned i = 0; i < clusterKeys.size(); ++i) {
      const auto &clusterKey = clusterKeys[i];
      auto clusterId = clusterKey.fClusterId;
      std::vector<RDaosSealedPageLocator> onDiskClusterPages;

      unsigned clusterBufSz = 0;
      fCounters->fNClusterLoaded.Inc();
      {
         auto descriptorGuard = GetSharedDescriptorGuard();
         const auto &clusterDesc = descriptorGuard->GetClusterDescriptor(clusterId);

         // Collect the necessary page meta-data and sum up the total size of the compressed and packed pages
         for (auto columnId : clusterKey.fColumnSet) {
            const auto &pageRange = clusterDesc.GetPageRange(columnId);
            NTupleSize_t columnPageCount = 0;
            for (const auto &pageInfo : pageRange.fPageInfos) {
               const auto &pageLocator = pageInfo.fLocator;
               onDiskClusterPages.push_back(RDaosSealedPageLocator(clusterId, columnId, columnPageCount,
                                                                   pageLocator.GetPosition<std::uint64_t>(),
                                                                   pageLocator.fBytesOnStorage, clusterBufSz));
               ++columnPageCount;
               clusterBufSz += pageLocator.fBytesOnStorage;
            }
            nPages += columnPageCount;
         }
      }
      szPayload += clusterBufSz;

      clusterBuffers[i] = new unsigned char[clusterBufSz];
      pageMaps[i] = std::make_unique<ROnDiskPageMapHeap>(std::unique_ptr<unsigned char[]>(clusterBuffers[i]));

      // Fill the cluster page maps and the input dictionary for the RDaosContainer::ReadV() call
      for (const auto &s : onDiskClusterPages) {
         // Register the on disk pages in a page map
         ROnDiskPage::Key key(s.fColumnId, s.fPageNo);
         pageMaps[i]->Register(key, ROnDiskPage(clusterBuffers[i] + s.fBufPos, s.fSize));

         // Prepare new read request batched up by object ID and distribution key
         d_iov_t iov;
         d_iov_set(&iov, clusterBuffers[i] + s.fBufPos, s.fSize);

         RDaosKey daosKey = GetPageDaosKey<kDefaultDaosMapping>(fNTupleIndex, s.fClusterId, s.fColumnId, s.fObjectId);
         auto odPair = RDaosContainer::ROidDkeyPair{daosKey.fOid, daosKey.fDkey};
         auto [it, ret] = readRequests.emplace(odPair, RDaosContainer::RWOperation(odPair));
         it->second.insert(daosKey.fAkey, iov);
      }
   }
   fCounters->fNPageLoaded.Add(nPages);
   fCounters->fSzReadPayload.Add(szPayload);

   {
      RNTupleAtomicTimer timer(fCounters->fTimeWallRead, fCounters->fTimeCpuRead);
      if (int err = fDaosContainer->ReadV(readRequests))
         throw ROOT::Experimental::RException(R__FAIL("ReadV: error" + std::string(d_errstr(err))));
   }
   fCounters->fNReadV.Inc();
   fCounters->fNRead.Add(nPages);

   // Assign each cluster its page map
   for (unsigned i = 0; i < clusterKeys.size(); ++i) {
      auto cluster = std::make_unique<RCluster>(clusterKeys[i].fClusterId);
      cluster->Adopt(std::move(pageMaps[i]));
      for (auto colId : clusterKeys[i].fColumnSet)
         cluster->SetColumnAvailable(colId);

      result.emplace_back(std::move(cluster));
   }
   return result;
}

void ROOT::Experimental::Detail::RPageSourceDaos::UnzipClusterImpl(RCluster *cluster)
{
   RNTupleAtomicTimer timer(fCounters->fTimeWallUnzip, fCounters->fTimeCpuUnzip);
   fTaskScheduler->Reset();

   const auto clusterId = cluster->GetId();
   auto descriptorGuard = GetSharedDescriptorGuard();
   const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterId);

   std::vector<std::unique_ptr<RColumnElementBase>> allElements;

   const auto &columnsInCluster = cluster->GetAvailColumns();
   for (const auto columnId : columnsInCluster) {
      const auto &columnDesc = descriptorGuard->GetColumnDescriptor(columnId);

      allElements.emplace_back(RColumnElementBase::Generate(columnDesc.GetModel().GetType()));

      const auto &pageRange = clusterDescriptor.GetPageRange(columnId);
      std::uint64_t pageNo = 0;
      std::uint64_t firstInPage = 0;
      for (const auto &pi : pageRange.fPageInfos) {
         ROnDiskPage::Key key(columnId, pageNo);
         auto onDiskPage = cluster->GetOnDiskPage(key);
         R__ASSERT(onDiskPage && (onDiskPage->GetSize() == pi.fLocator.fBytesOnStorage));

         auto taskFunc = [this, columnId, clusterId, firstInPage, onDiskPage, element = allElements.back().get(),
                          nElements = pi.fNElements,
                          indexOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex]() {
            auto pageBuffer = UnsealPage({onDiskPage->GetAddress(), onDiskPage->GetSize(), nElements}, *element);
            fCounters->fSzUnzip.Add(element->GetSize() * nElements);

            auto newPage = fPageAllocator->NewPage(columnId, pageBuffer.release(), element->GetSize(), nElements);
            newPage.SetWindow(indexOffset + firstInPage, RPage::RClusterInfo(clusterId, indexOffset));
            fPagePool->PreloadPage(
               newPage,
               RPageDeleter([](const RPage &page, void * /*userData*/) { RPageAllocatorDaos::DeletePage(page); },
                            nullptr));
         };

         fTaskScheduler->AddTask(taskFunc);

         firstInPage += pi.fNElements;
         pageNo++;
      } // for all pages in column
   }    // for all columns in cluster

   fCounters->fNPagePopulated.Add(cluster->GetNOnDiskPages());

   fTaskScheduler->Wait();
}
