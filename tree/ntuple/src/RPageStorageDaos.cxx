/// \file RPageStorageDaos.cxx
/// \ingroup NTuple
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
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleWriteOptionsDaos.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleUtils.hxx>
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
#include <cstring>
#include <limits>
#include <utility>
#include <regex>
#include <cassert>

namespace {
using AttributeKey_t = ROOT::Experimental::Internal::RDaosContainer::AttributeKey_t;
using DistributionKey_t = ROOT::Experimental::Internal::RDaosContainer::DistributionKey_t;
using ntuple_index_t = ROOT::Experimental::Internal::ntuple_index_t;
using ROOT::Internal::MakeUninitArray;
using ROOT::Internal::RCluster;
using ROOT::Internal::RNTupleCompressor;
using ROOT::Internal::RNTupleDecompressor;
using ROOT::Internal::RNTupleSerializer;

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
RDaosKey GetPageDaosKey(ROOT::Experimental::Internal::ntuple_index_t ntplId, long unsigned clusterId,
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
      throw ROOT::RException(R__FAIL("Invalid DAOS pool URI."));
   return {m[1], m[2]};
}

/// \brief Helper structure concentrating the functionality required to locate an ntuple within a DAOS container.
/// It includes a hashing function that converts the RNTuple's name into a 32-bit identifier; this value is used to
/// index the subspace for the ntuple among all objects in the container. A zero-value hash value is reserved for
/// storing any future metadata related to container-wide management; a zero-index ntuple is thus disallowed and
/// remapped to "1". Once the index is computed, `InitNTupleDescriptorBuilder()` can be called to return a
/// partially-filled builder with the ntuple's anchor, header and footer, lacking only pagelists. Upon that call,
/// a copy of the anchor is stored in `fAnchor`.
struct RDaosContainerNTupleLocator {
   std::string fName{};
   ntuple_index_t fIndex{};
   std::optional<ROOT::Experimental::Internal::RDaosNTupleAnchor> fAnchor;
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

   int InitNTupleDescriptorBuilder(ROOT::Experimental::Internal::RDaosContainer &cont,
                                   ROOT::Internal::RNTupleDescriptorBuilder &builder)
   {
      std::unique_ptr<unsigned char[]> buffer, zipBuffer;
      auto &anchor = fAnchor.emplace();
      int err;

      const auto anchorSize = ROOT::Experimental::Internal::RDaosNTupleAnchor::GetSize();
      daos_obj_id_t oidMetadata{kOidLowMetadata, static_cast<decltype(daos_obj_id_t::hi)>(this->GetIndex())};

      buffer = MakeUninitArray<unsigned char>(anchorSize);
      if ((err = cont.ReadSingleAkey(buffer.get(), anchorSize, oidMetadata, kDistributionKeyDefault,
                                     kAttributeKeyAnchor, kCidMetadata))) {
         return err;
      }

      anchor.Deserialize(buffer.get(), anchorSize).Unwrap();

      builder.SetVersion(anchor.fVersionEpoch, anchor.fVersionMajor, anchor.fVersionMinor, anchor.fVersionPatch);
      builder.SetOnDiskHeaderSize(anchor.fNBytesHeader);
      buffer = MakeUninitArray<unsigned char>(anchor.fLenHeader);
      zipBuffer = MakeUninitArray<unsigned char>(anchor.fNBytesHeader);
      if ((err = cont.ReadSingleAkey(zipBuffer.get(), anchor.fNBytesHeader, oidMetadata, kDistributionKeyDefault,
                                     kAttributeKeyHeader, kCidMetadata)))
         return err;
      RNTupleDecompressor::Unzip(zipBuffer.get(), anchor.fNBytesHeader, anchor.fLenHeader, buffer.get());
      RNTupleSerializer::DeserializeHeader(buffer.get(), anchor.fLenHeader, builder);

      builder.AddToOnDiskFooterSize(anchor.fNBytesFooter);
      buffer = MakeUninitArray<unsigned char>(anchor.fLenFooter);
      zipBuffer = MakeUninitArray<unsigned char>(anchor.fNBytesFooter);
      if ((err = cont.ReadSingleAkey(zipBuffer.get(), anchor.fNBytesFooter, oidMetadata, kDistributionKeyDefault,
                                     kAttributeKeyFooter, kCidMetadata)))
         return err;
      RNTupleDecompressor::Unzip(zipBuffer.get(), anchor.fNBytesFooter, anchor.fLenFooter, buffer.get());
      RNTupleSerializer::DeserializeFooter(buffer.get(), anchor.fLenFooter, builder);

      return 0;
   }

   static std::pair<RDaosContainerNTupleLocator, ROOT::Internal::RNTupleDescriptorBuilder>
   LocateNTuple(ROOT::Experimental::Internal::RDaosContainer &cont, const std::string &ntupleName)
   {
      auto result = std::make_pair(RDaosContainerNTupleLocator(ntupleName), ROOT::Internal::RNTupleDescriptorBuilder());

      auto &loc = result.first;
      auto &builder = result.second;

      if (int err = loc.InitNTupleDescriptorBuilder(cont, builder); !err) {
         if (ntupleName.empty() || ntupleName != builder.GetDescriptor().GetName()) {
            // Hash already taken by a differently-named ntuple.
            throw ROOT::RException(
               R__FAIL("LocateNTuple: ntuple name '" + ntupleName + "' unavailable in this container."));
         }
      }
      return result;
   }
};

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

std::uint32_t ROOT::Experimental::Internal::RDaosNTupleAnchor::Serialize(void *buffer) const
{
   if (buffer != nullptr) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);
      bytes += RNTupleSerializer::SerializeUInt64(fVersionAnchor, bytes);
      bytes += RNTupleSerializer::SerializeUInt16(fVersionEpoch, bytes);
      bytes += RNTupleSerializer::SerializeUInt16(fVersionMajor, bytes);
      bytes += RNTupleSerializer::SerializeUInt16(fVersionMinor, bytes);
      bytes += RNTupleSerializer::SerializeUInt16(fVersionPatch, bytes);
      bytes += RNTupleSerializer::SerializeUInt32(fNBytesHeader, bytes);
      bytes += RNTupleSerializer::SerializeUInt32(fLenHeader, bytes);
      bytes += RNTupleSerializer::SerializeUInt32(fNBytesFooter, bytes);
      bytes += RNTupleSerializer::SerializeUInt32(fLenFooter, bytes);
      bytes += RNTupleSerializer::SerializeString(fObjClass, bytes);
   }
   return RNTupleSerializer::SerializeString(fObjClass, nullptr) + 32;
}

ROOT::RResult<std::uint32_t>
ROOT::Experimental::Internal::RDaosNTupleAnchor::Deserialize(const void *buffer, std::uint32_t bufSize)
{
   if (bufSize < 32)
      return R__FAIL("DAOS anchor too short");

   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   bytes += RNTupleSerializer::DeserializeUInt64(bytes, fVersionAnchor);
   if (fVersionAnchor != RDaosNTupleAnchor().fVersionAnchor) {
      return R__FAIL("unsupported DAOS anchor version: " + std::to_string(fVersionAnchor));
   }

   bytes += RNTupleSerializer::DeserializeUInt16(bytes, fVersionEpoch);
   bytes += RNTupleSerializer::DeserializeUInt16(bytes, fVersionMajor);
   bytes += RNTupleSerializer::DeserializeUInt16(bytes, fVersionMinor);
   bytes += RNTupleSerializer::DeserializeUInt16(bytes, fVersionPatch);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fNBytesHeader);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fLenHeader);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fNBytesFooter);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fLenFooter);
   auto result = RNTupleSerializer::DeserializeString(bytes, bufSize - 32, fObjClass);
   if (!result)
      return R__FORWARD_ERROR(result);
   return result.Unwrap() + 32;
}

std::uint32_t ROOT::Experimental::Internal::RDaosNTupleAnchor::GetSize()
{
   return RDaosNTupleAnchor().Serialize(nullptr) + RDaosObject::ObjClassId::kOCNameMaxLength;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::Internal::RPageSinkDaos::RPageSinkDaos(std::string_view ntupleName, std::string_view uri,
                                                           const ROOT::RNTupleWriteOptions &options)
   : RPagePersistentSink(ntupleName, options), fURI(uri)
{
   static std::once_flag once;
   std::call_once(once, []() {
      R__LOG_WARNING(ROOT::Internal::NTupleLog()) << "The DAOS backend is experimental and still under development. "
                                                  << "Do not store real data with this version of RNTuple!";
   });
   EnableDefaultMetrics("RPageSinkDaos");
}

ROOT::Experimental::Internal::RPageSinkDaos::~RPageSinkDaos() = default;

void ROOT::Experimental::Internal::RPageSinkDaos::InitImpl(unsigned char *serializedHeader, std::uint32_t length)
{
   auto opts = dynamic_cast<RNTupleWriteOptionsDaos *>(fOptions.get());
   fNTupleAnchor.fObjClass = opts ? opts->GetObjectClass() : RNTupleWriteOptionsDaos().GetObjectClass();
   auto oclass = RDaosObject::ObjClassId(fNTupleAnchor.fObjClass);
   if (oclass.IsUnknown())
      throw ROOT::RException(R__FAIL("Unknown object class " + fNTupleAnchor.fObjClass));

   auto args = ParseDaosURI(fURI);
   auto pool = std::make_shared<RDaosPool>(args.fPoolLabel);

   fDaosContainer = std::make_unique<RDaosContainer>(pool, args.fContainerLabel, /*create =*/true);
   fDaosContainer->SetDefaultObjectClass(oclass);

   auto [locator, _] = RDaosContainerNTupleLocator::LocateNTuple(*fDaosContainer, fNTupleName);
   fNTupleIndex = locator.GetIndex();

   auto zipBuffer = MakeUninitArray<unsigned char>(length);
   auto szZipHeader =
      RNTupleCompressor::Zip(serializedHeader, length, GetWriteOptions().GetCompression(), zipBuffer.get());
   WriteNTupleHeader(zipBuffer.get(), szZipHeader, length);
}

ROOT::RNTupleLocator ROOT::Experimental::Internal::RPageSinkDaos::CommitPageImpl(ColumnHandle_t columnHandle,
                                                                                 const ROOT::Internal::RPage &page)
{
   auto element = columnHandle.fColumn->GetElement();
   RPageStorage::RSealedPage sealedPage;
   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallZip, fCounters->fTimeCpuZip);
      sealedPage = SealPage(page, *element);
   }

   fCounters->fSzZip.Add(page.GetNBytes());
   return CommitSealedPageImpl(columnHandle.fPhysicalId, sealedPage);
}

ROOT::RNTupleLocator
ROOT::Experimental::Internal::RPageSinkDaos::CommitSealedPageImpl(ROOT::DescriptorId_t physicalColumnId,
                                                                  const RPageStorage::RSealedPage &sealedPage)
{
   auto pageId = fPageId.fetch_add(1);
   ROOT::DescriptorId_t clusterId = fDescriptorBuilder.GetDescriptor().GetNActiveClusters();

   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
      RDaosKey daosKey = GetPageDaosKey<kDefaultDaosMapping>(fNTupleIndex, clusterId, physicalColumnId, pageId);
      fDaosContainer->WriteSingleAkey(sealedPage.GetBuffer(), sealedPage.GetBufferSize(), daosKey.fOid, daosKey.fDkey,
                                      daosKey.fAkey);
   }

   RNTupleLocator result;
   result.SetType(RNTupleLocator::kTypeDAOS);
   result.SetNBytesOnStorage(sealedPage.GetDataSize());
   result.SetPosition(ROOT::RNTupleLocatorObject64{pageId});
   fCounters->fNPageCommitted.Inc();
   fCounters->fSzWritePayload.Add(sealedPage.GetBufferSize());
   fNBytesCurrentCluster += sealedPage.GetBufferSize();
   return result;
}

std::vector<ROOT::RNTupleLocator>
ROOT::Experimental::Internal::RPageSinkDaos::CommitSealedPageVImpl(std::span<RPageStorage::RSealedPageGroup> ranges,
                                                                   const std::vector<bool> &mask)
{
   RDaosContainer::MultiObjectRWOperation_t writeRequests;
   std::vector<RNTupleLocator> locators;
   auto nPages = mask.size();
   locators.reserve(nPages);

   ROOT::DescriptorId_t clusterId = fDescriptorBuilder.GetDescriptor().GetNActiveClusters();
   int64_t payloadSz = 0;

   /// Aggregate batch of requests by object ID and distribution key, determined by the ntuple-DAOS mapping
   for (auto &range : ranges) {
      for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt) {
         const RPageStorage::RSealedPage &s = *sealedPageIt;

         const auto pageId = fPageId.fetch_add(1);

         d_iov_t pageIov;
         d_iov_set(&pageIov, const_cast<void *>(s.GetBuffer()), s.GetBufferSize());

         RDaosKey daosKey =
            GetPageDaosKey<kDefaultDaosMapping>(fNTupleIndex, clusterId, range.fPhysicalColumnId, pageId);
         auto odPair = RDaosContainer::ROidDkeyPair{daosKey.fOid, daosKey.fDkey};
         auto [it, ret] = writeRequests.emplace(odPair, RDaosContainer::RWOperation(odPair));
         it->second.Insert(daosKey.fAkey, pageIov);

         RNTupleLocator locator;
         locator.SetType(RNTupleLocator::kTypeDAOS);
         locator.SetNBytesOnStorage(s.GetDataSize());
         locator.SetPosition(ROOT::RNTupleLocatorObject64{pageId});
         locators.push_back(locator);

         payloadSz += s.GetBufferSize();
      }
   }
   fNBytesCurrentCluster += payloadSz;

   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
      if (int err = fDaosContainer->WriteV(writeRequests))
         throw ROOT::RException(R__FAIL("WriteV: error" + std::string(d_errstr(err))));
   }

   fCounters->fNPageCommitted.Add(nPages);
   fCounters->fSzWritePayload.Add(payloadSz);

   return locators;
}

std::uint64_t ROOT::Experimental::Internal::RPageSinkDaos::StageClusterImpl()
{
   return std::exchange(fNBytesCurrentCluster, 0);
}

ROOT::RNTupleLocator
ROOT::Experimental::Internal::RPageSinkDaos::CommitClusterGroupImpl(unsigned char *serializedPageList,
                                                                    std::uint32_t length)
{
   auto bufPageListZip = MakeUninitArray<unsigned char>(length);
   auto szPageListZip =
      RNTupleCompressor::Zip(serializedPageList, length, GetWriteOptions().GetCompression(), bufPageListZip.get());

   auto offsetData = fClusterGroupId.fetch_add(1);
   fDaosContainer->WriteSingleAkey(
      bufPageListZip.get(), szPageListZip,
      daos_obj_id_t{kOidLowPageList, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)}, kDistributionKeyDefault,
      offsetData, kCidMetadata);
   RNTupleLocator result;
   result.SetType(RNTupleLocator::kTypeDAOS);
   result.SetNBytesOnStorage(szPageListZip);
   result.SetPosition(RNTupleLocatorObject64{offsetData});
   fCounters->fSzWritePayload.Add(static_cast<int64_t>(szPageListZip));
   return result;
}

void ROOT::Experimental::Internal::RPageSinkDaos::CommitDatasetImpl(unsigned char *serializedFooter,
                                                                    std::uint32_t length)
{
   auto bufFooterZip = MakeUninitArray<unsigned char>(length);
   auto szFooterZip =
      RNTupleCompressor::Zip(serializedFooter, length, GetWriteOptions().GetCompression(), bufFooterZip.get());
   WriteNTupleFooter(bufFooterZip.get(), szFooterZip, length);
   WriteNTupleAnchor();
}

void ROOT::Experimental::Internal::RPageSinkDaos::WriteNTupleHeader(const void *data, size_t nbytes, size_t lenHeader)
{
   fDaosContainer->WriteSingleAkey(
      data, nbytes, daos_obj_id_t{kOidLowMetadata, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)},
      kDistributionKeyDefault, kAttributeKeyHeader, kCidMetadata);
   fNTupleAnchor.fLenHeader = lenHeader;
   fNTupleAnchor.fNBytesHeader = nbytes;
}

void ROOT::Experimental::Internal::RPageSinkDaos::WriteNTupleFooter(const void *data, size_t nbytes, size_t lenFooter)
{
   fDaosContainer->WriteSingleAkey(
      data, nbytes, daos_obj_id_t{kOidLowMetadata, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)},
      kDistributionKeyDefault, kAttributeKeyFooter, kCidMetadata);
   fNTupleAnchor.fLenFooter = lenFooter;
   fNTupleAnchor.fNBytesFooter = nbytes;
}

void ROOT::Experimental::Internal::RPageSinkDaos::WriteNTupleAnchor()
{
   const auto ntplSize = RDaosNTupleAnchor::GetSize();
   auto buffer = MakeUninitArray<unsigned char>(ntplSize);
   fNTupleAnchor.Serialize(buffer.get());
   fDaosContainer->WriteSingleAkey(
      buffer.get(), ntplSize, daos_obj_id_t{kOidLowMetadata, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)},
      kDistributionKeyDefault, kAttributeKeyAnchor, kCidMetadata);
}

std::unique_ptr<ROOT::Internal::RPageSink>
ROOT::Experimental::Internal::RPageSinkDaos::CloneWithDifferentName(std::string_view /*name*/,
                                                                    const ROOT::RNTupleWriteOptions & /*opts*/) const
{
   throw ROOT::RException(R__FAIL("cloning a DAOS sink is not implemented yet"));
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::Internal::RPageSourceDaos::RPageSourceDaos(std::string_view ntupleName, std::string_view uri,
                                                               const ROOT::RNTupleReadOptions &options)
   : RPageSource(ntupleName, options), fURI(uri)
{
   EnableDefaultMetrics("RPageSourceDaos");

   auto args = ParseDaosURI(uri);
   auto pool = std::make_shared<RDaosPool>(args.fPoolLabel);
   fDaosContainer = std::make_unique<RDaosContainer>(pool, args.fContainerLabel);
}

ROOT::Experimental::Internal::RPageSourceDaos::~RPageSourceDaos()
{
   fClusterPool.StopBackgroundThread();
}

ROOT::RNTupleDescriptor
ROOT::Experimental::Internal::RPageSourceDaos::AttachImpl(RNTupleSerializer::EDescriptorDeserializeMode mode)
{
   ROOT::RNTupleDescriptor ntplDesc;
   std::unique_ptr<unsigned char[]> buffer, zipBuffer;

   auto [locator, descBuilder] = RDaosContainerNTupleLocator::LocateNTuple(*fDaosContainer, fNTupleName);
   if (!locator.IsValid())
      throw ROOT::RException(
         R__FAIL("Attach: requested ntuple '" + fNTupleName + "' is not present in DAOS container."));

   auto oclass = RDaosObject::ObjClassId(locator.fAnchor->fObjClass);
   if (oclass.IsUnknown())
      throw ROOT::RException(R__FAIL("Attach: unknown object class " + locator.fAnchor->fObjClass));

   fDaosContainer->SetDefaultObjectClass(oclass);
   fNTupleIndex = locator.GetIndex();
   daos_obj_id_t oidPageList{kOidLowPageList, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)};

   auto desc = descBuilder.MoveDescriptor();

   for (const auto &cgDesc : desc.GetClusterGroupIterable()) {
      buffer = MakeUninitArray<unsigned char>(cgDesc.GetPageListLength());
      zipBuffer = MakeUninitArray<unsigned char>(cgDesc.GetPageListLocator().GetNBytesOnStorage());
      fDaosContainer->ReadSingleAkey(
         zipBuffer.get(), cgDesc.GetPageListLocator().GetNBytesOnStorage(), oidPageList, kDistributionKeyDefault,
         cgDesc.GetPageListLocator().GetPosition<RNTupleLocatorObject64>().GetLocation(), kCidMetadata);
      RNTupleDecompressor::Unzip(zipBuffer.get(), cgDesc.GetPageListLocator().GetNBytesOnStorage(),
                                 cgDesc.GetPageListLength(), buffer.get());

      RNTupleSerializer::DeserializePageList(buffer.get(), cgDesc.GetPageListLength(), cgDesc.GetId(), desc, mode);
   }

   return desc;
}

std::string ROOT::Experimental::Internal::RPageSourceDaos::GetObjectClass() const
{
   return fDaosContainer->GetDefaultObjectClass().ToString();
}

void ROOT::Experimental::Internal::RPageSourceDaos::LoadSealedPage(ROOT::DescriptorId_t physicalColumnId,
                                                                   RNTupleLocalIndex localIndex,
                                                                   RSealedPage &sealedPage)
{
   const auto clusterId = localIndex.GetClusterId();

   ROOT::RClusterDescriptor::RPageInfo pageInfo;
   {
      auto descriptorGuard = GetSharedDescriptorGuard();
      const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterId);
      pageInfo = clusterDescriptor.GetPageRange(physicalColumnId).Find(localIndex.GetIndexInCluster());
   }

   sealedPage.SetBufferSize(pageInfo.GetLocator().GetNBytesOnStorage() + pageInfo.HasChecksum() * kNBytesPageChecksum);
   sealedPage.SetNElements(pageInfo.GetNElements());
   sealedPage.SetHasChecksum(pageInfo.HasChecksum());
   if (!sealedPage.GetBuffer())
      return;

   if (pageInfo.GetLocator().GetType() == RNTupleLocator::kTypePageZero) {
      assert(!pageInfo.HasChecksum());
      memcpy(const_cast<void *>(sealedPage.GetBuffer()), ROOT::Internal::RPage::GetPageZeroBuffer(),
             sealedPage.GetBufferSize());
      return;
   }

   RDaosKey daosKey =
      GetPageDaosKey<kDefaultDaosMapping>(fNTupleIndex, clusterId, physicalColumnId,
                                          pageInfo.GetLocator().GetPosition<RNTupleLocatorObject64>().GetLocation());
   fDaosContainer->ReadSingleAkey(const_cast<void *>(sealedPage.GetBuffer()), sealedPage.GetBufferSize(), daosKey.fOid,
                                  daosKey.fDkey, daosKey.fAkey);

   sealedPage.VerifyChecksumIfEnabled().ThrowOnError();
}

ROOT::Internal::RPageRef ROOT::Experimental::Internal::RPageSourceDaos::LoadPageImpl(ColumnHandle_t columnHandle,
                                                                                     const RClusterInfo &clusterInfo,
                                                                                     ROOT::NTupleSize_t idxInCluster)
{
   const auto columnId = columnHandle.fPhysicalId;
   const auto clusterId = clusterInfo.fClusterId;
   const auto &pageInfo = clusterInfo.fPageInfo;

   const auto element = columnHandle.fColumn->GetElement();
   const auto elementSize = element->GetSize();
   const auto elementInMemoryType = element->GetIdentifier().fInMemoryType;

   if (pageInfo.GetLocator().GetType() == RNTupleLocator::kTypePageZero) {
      auto pageZero = fPageAllocator->NewPage(elementSize, pageInfo.GetNElements());
      pageZero.GrowUnchecked(pageInfo.GetNElements());
      memset(pageZero.GetBuffer(), 0, pageZero.GetNBytes());
      pageZero.SetWindow(clusterInfo.fColumnOffset + pageInfo.GetFirstElementIndex(),
                         ROOT::Internal::RPage::RClusterInfo(clusterId, clusterInfo.fColumnOffset));
      return fPagePool.RegisterPage(std::move(pageZero),
                                    ROOT::Internal::RPagePool::RKey{columnId, elementInMemoryType});
   }

   RSealedPage sealedPage;
   sealedPage.SetNElements(pageInfo.GetNElements());
   sealedPage.SetHasChecksum(pageInfo.HasChecksum());
   sealedPage.SetBufferSize(pageInfo.GetLocator().GetNBytesOnStorage() + pageInfo.HasChecksum() * kNBytesPageChecksum);
   std::unique_ptr<unsigned char[]> directReadBuffer; // only used if cluster pool is turned off

   if (fOptions.GetClusterCache() == ROOT::RNTupleReadOptions::EClusterCache::kOff) {
      directReadBuffer = MakeUninitArray<unsigned char>(sealedPage.GetBufferSize());
      RDaosKey daosKey = GetPageDaosKey<kDefaultDaosMapping>(
         fNTupleIndex, clusterId, columnId, pageInfo.GetLocator().GetPosition<RNTupleLocatorObject64>().GetLocation());
      fDaosContainer->ReadSingleAkey(directReadBuffer.get(), sealedPage.GetBufferSize(), daosKey.fOid, daosKey.fDkey,
                                     daosKey.fAkey);
      fCounters->fNPageRead.Inc();
      fCounters->fNRead.Inc();
      fCounters->fSzReadPayload.Add(sealedPage.GetBufferSize());
      sealedPage.SetBuffer(directReadBuffer.get());
   } else {
      if (!fCurrentCluster || (fCurrentCluster->GetId() != clusterId) || !fCurrentCluster->ContainsColumn(columnId))
         fCurrentCluster = fClusterPool.GetCluster(clusterId, fActivePhysicalColumns.ToColumnSet());
      R__ASSERT(fCurrentCluster->ContainsColumn(columnId));

      auto cachedPageRef = fPagePool.GetPage(ROOT::Internal::RPagePool::RKey{columnId, elementInMemoryType},
                                             RNTupleLocalIndex(clusterId, idxInCluster));
      if (!cachedPageRef.Get().IsNull())
         return cachedPageRef;

      ROOT::Internal::ROnDiskPage::Key key(columnId, pageInfo.GetPageNumber());
      auto onDiskPage = fCurrentCluster->GetOnDiskPage(key);
      R__ASSERT(onDiskPage && (sealedPage.GetBufferSize() == onDiskPage->GetSize()));
      sealedPage.SetBuffer(onDiskPage->GetAddress());
   }

   ROOT::Internal::RPage newPage;
   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallUnzip, fCounters->fTimeCpuUnzip);
      newPage = UnsealPage(sealedPage, *element).Unwrap();
      fCounters->fSzUnzip.Add(elementSize * pageInfo.GetNElements());
   }

   newPage.SetWindow(clusterInfo.fColumnOffset + pageInfo.GetFirstElementIndex(),
                     ROOT::Internal::RPage::RClusterInfo(clusterId, clusterInfo.fColumnOffset));
   fCounters->fNPageUnsealed.Inc();
   return fPagePool.RegisterPage(std::move(newPage), ROOT::Internal::RPagePool::RKey{columnId, elementInMemoryType});
}

std::unique_ptr<ROOT::Internal::RPageSource> ROOT::Experimental::Internal::RPageSourceDaos::CloneImpl() const
{
   auto clone = new RPageSourceDaos(fNTupleName, fURI, fOptions);
   return std::unique_ptr<RPageSourceDaos>(clone);
}

std::vector<std::unique_ptr<RCluster>>
ROOT::Experimental::Internal::RPageSourceDaos::LoadClusters(std::span<RCluster::RKey> clusterKeys)
{
   struct RDaosSealedPageLocator {
      ROOT::DescriptorId_t fClusterId = 0;
      ROOT::DescriptorId_t fColumnId = 0;
      ROOT::NTupleSize_t fPageNo = 0;
      std::uint64_t fPageId = 0;
      std::uint64_t fDataSize = 0;   // page payload
      std::uint64_t fBufferSize = 0; // page payload + checksum (if available)
   };

   // Prepares read requests for a single cluster; `readRequests` is modified by this function.  Requests are coalesced
   // by OID and distribution key.
   // TODO(jalopezg): this may be a private member function; that, however, requires additional changes given that
   // `RDaosContainer::MultiObjectRWOperation_t` cannot be forward-declared
   auto fnPrepareSingleCluster = [&](const RCluster::RKey &clusterKey,
                                     RDaosContainer::MultiObjectRWOperation_t &readRequests) {
      auto clusterId = clusterKey.fClusterId;
      std::vector<RDaosSealedPageLocator> onDiskPages;

      unsigned clusterBufSz = 0, nPages = 0;
      auto pageZeroMap = std::make_unique<ROOT::Internal::ROnDiskPageMap>();
      PrepareLoadCluster(
         clusterKey, *pageZeroMap,
         [&](ROOT::DescriptorId_t physicalColumnId, ROOT::NTupleSize_t pageNo,
             const ROOT::RClusterDescriptor::RPageInfo &pageInfo) {
            const auto &pageLocator = pageInfo.GetLocator();
            const auto pageId = pageLocator.GetPosition<RNTupleLocatorObject64>().GetLocation();
            const auto pageBufferSize = pageLocator.GetNBytesOnStorage() + pageInfo.HasChecksum() * kNBytesPageChecksum;
            onDiskPages.emplace_back(RDaosSealedPageLocator{clusterId, physicalColumnId, pageNo, pageId,
                                                            pageLocator.GetNBytesOnStorage(), pageBufferSize});

            ++nPages;
            clusterBufSz += pageBufferSize;
         });

      auto clusterBuffer = new unsigned char[clusterBufSz];
      auto pageMap =
         std::make_unique<ROOT::Internal::ROnDiskPageMapHeap>(std::unique_ptr<unsigned char[]>(clusterBuffer));

      // Fill the cluster page map and the read requests for the RDaosContainer::ReadV() call
      for (const auto &sealedLoc : onDiskPages) {
         ROOT::Internal::ROnDiskPage::Key key(sealedLoc.fColumnId, sealedLoc.fPageNo);
         pageMap->Register(key, ROOT::Internal::ROnDiskPage(clusterBuffer, sealedLoc.fBufferSize));

         // Prepare new read request batched up by object ID and distribution key
         d_iov_t iov;
         d_iov_set(&iov, clusterBuffer, sealedLoc.fBufferSize);

         RDaosKey daosKey = GetPageDaosKey<kDefaultDaosMapping>(fNTupleIndex, sealedLoc.fClusterId, sealedLoc.fColumnId,
                                                                sealedLoc.fPageId);
         auto odPair = RDaosContainer::ROidDkeyPair{daosKey.fOid, daosKey.fDkey};
         auto [itReq, ret] = readRequests.emplace(odPair, RDaosContainer::RWOperation(odPair));
         itReq->second.Insert(daosKey.fAkey, iov);

         clusterBuffer += sealedLoc.fBufferSize;
      }
      fCounters->fNPageRead.Add(nPages);
      fCounters->fSzReadPayload.Add(clusterBufSz);

      auto cluster = std::make_unique<RCluster>(clusterId);
      cluster->Adopt(std::move(pageMap));
      cluster->Adopt(std::move(pageZeroMap));
      for (auto colId : clusterKey.fPhysicalColumnSet)
         cluster->SetColumnAvailable(colId);
      return cluster;
   };

   fCounters->fNClusterLoaded.Add(clusterKeys.size());

   std::vector<std::unique_ptr<ROOT::Internal::RCluster>> clusters;
   RDaosContainer::MultiObjectRWOperation_t readRequests;
   for (auto key : clusterKeys) {
      clusters.emplace_back(fnPrepareSingleCluster(key, readRequests));
   }

   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallRead, fCounters->fTimeCpuRead);
      if (int err = fDaosContainer->ReadV(readRequests))
         throw ROOT::RException(R__FAIL("ReadV: error" + std::string(d_errstr(err))));
   }
   fCounters->fNReadV.Inc();
   fCounters->fNRead.Add(readRequests.size());

   return clusters;
}

void ROOT::Experimental::Internal::RPageSourceDaos::LoadStreamerInfo()
{
   R__LOG_WARNING(ROOT::Internal::NTupleLog()) << "DAOS-backed sources have no associated StreamerInfo to load.";
}
