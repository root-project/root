/// \file RPageStorageDaos.cxx
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
#include <tuple>
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

struct RDaosKey {
   daos_obj_id_t fOid;
   DistributionKey_t fDkey;
   AttributeKey_t fAkey;
};

/// \brief Pre-defined keys for object store. `kDistributionKeyDefault` is the distribution key for all objects,
/// `kAttributeKeyDefault` is the attribute key for all objects but anchor, header, footer.
/// `kAttributeKey{Anchor,Header,Footer}` are the respective attribute keys for anchor/header/footer metadata elements.
static constexpr DistributionKey_t kDistributionKeyDefault = 0x5a3c69f0cafe4a11;
static constexpr AttributeKey_t kAttributeKeyDefault = 0x4243544b53444229;
static constexpr AttributeKey_t kAttributeKeyAnchor = 0x4243544b5344422a;
static constexpr AttributeKey_t kAttributeKeyHeader = 0x4243544b5344422b;
static constexpr AttributeKey_t kAttributeKeyFooter = 0x4243544b5344422c;

/// \brief Pre-defined 64 LSb of the OIDs for ntuple metadata (holds anchor/header/footer) and clusters' pagelists.
static constexpr decltype(daos_obj_id_t::lo) kOidLowMetadata = -1;
static constexpr decltype(daos_obj_id_t::lo) kOidLowPageList = -2;

/// Because the object class becomes part of the object ID (encoded in the system-reserved 32 bits), we have to
/// hard-code the object class for the anchor.  Otherwise, we would need ask the user to specify the correct object
/// class in the RNTupleReadOptions when trying to open a previously written data set, which is not really acceptable.
/// The object class set in the RNTupleWriteOptions thus applies to all objects except the anchor.
static constexpr daos_oclass_id_t kCidAnchor = OC_UNKNOWN;

RDaosKey GetPageDaosKey(ROOT::Experimental::Internal::ntuple_index_t ntplId, long unsigned pageCount)
{
   return RDaosKey{daos_obj_id_t{static_cast<decltype(daos_obj_id_t::lo)>(pageCount),
                                 static_cast<decltype(daos_obj_id_t::hi)>(ntplId)},
                   kDistributionKeyDefault, kAttributeKeyDefault};
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
   explicit RDaosContainerNTupleLocator(const std::string &ntupleName) : fName(ntupleName), fIndex(Hash(ntupleName)) {}

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
      std::unique_ptr<unsigned char[]> buffer;
      auto &anchor = fAnchor.emplace();
      int err;

      const auto anchorSize = ROOT::Experimental::Internal::RDaosNTupleAnchor::GetSize();
      daos_obj_id_t oidMetadata{kOidLowMetadata, static_cast<decltype(daos_obj_id_t::hi)>(this->GetIndex())};

      buffer = MakeUninitArray<unsigned char>(anchorSize);
      if ((err = cont.ReadSingleAkey(buffer.get(), anchorSize, oidMetadata, kDistributionKeyDefault,
                                     kAttributeKeyAnchor, kCidAnchor))) {
         return err;
      }

      anchor.Deserialize(buffer.get(), anchorSize).Unwrap();

      builder.SetVersion(anchor.fVersionEpoch, anchor.fVersionMajor, anchor.fVersionMinor, anchor.fVersionPatch);
      builder.SetOnDiskHeaderSize(anchor.fNBytesHeader);
      builder.AddToOnDiskFooterSize(anchor.fNBytesFooter);

      return 0;
   }

   static std::pair<RDaosContainerNTupleLocator, ROOT::Internal::RNTupleDescriptorBuilder>
   LocateNTuple(ROOT::Experimental::Internal::RDaosContainer &cont, const std::string &ntupleName)
   {
      auto result = std::make_pair(RDaosContainerNTupleLocator(ntupleName), ROOT::Internal::RNTupleDescriptorBuilder());

      auto &loc = result.first;
      auto &builder = result.second;

      loc.InitNTupleDescriptorBuilder(cont, builder);
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

   auto args = ParseDaosURI(fURI);
   auto pool = std::make_unique<RDaosPool>(args.fPoolLabel);

   fDaosContainer = std::make_unique<RDaosContainer>(std::move(pool), args.fContainerLabel, /*create =*/true);
   fDaosContainer->SetDefaultObjectClass(fNTupleAnchor.fObjClass);

   auto [locator, _] = RDaosContainerNTupleLocator::LocateNTuple(*fDaosContainer, fNTupleName);
   fNTupleIndex = locator.GetIndex();

   auto zipBuffer = MakeUninitArray<unsigned char>(length);
   auto szZipHeader =
      RNTupleCompressor::Zip(serializedHeader, length, GetWriteOptions().GetCompression(), zipBuffer.get());
   WriteNTupleHeader(zipBuffer.get(), szZipHeader, length);
}

ROOT::RNTupleLocator
ROOT::Experimental::Internal::RPageSinkDaos::CommitSealedPageImpl(ROOT::DescriptorId_t,
                                                                  const RPageStorage::RSealedPage &sealedPage)
{
   auto pageId = fPageId.fetch_add(1);

   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
      RDaosKey daosKey = GetPageDaosKey(fNTupleIndex, pageId);
      fDaosContainer->WriteSingleAkey(sealedPage.GetBuffer(), sealedPage.GetBufferSize(), daosKey.fOid, daosKey.fDkey,
                                      daosKey.fAkey);
   }

   RNTupleLocator result;
   result.SetType(RNTupleLocator::kTypeObject64);
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

   int64_t payloadSz = 0;

   /// Aggregate batch of requests by object ID and distribution key, determined by the ntuple-DAOS mapping
   for (auto &range : ranges) {
      for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt) {
         const RPageStorage::RSealedPage &s = *sealedPageIt;

         const auto pageId = fPageId.fetch_add(1);

         d_iov_t pageIov;
         d_iov_set(&pageIov, const_cast<void *>(s.GetBuffer()), s.GetBufferSize());

         RDaosKey daosKey = GetPageDaosKey(fNTupleIndex, pageId);
         auto odPair = RDaosContainer::ROidDkeyPair{daosKey.fOid, daosKey.fDkey};
         auto [it, ret] = writeRequests.emplace(odPair, RDaosContainer::RWOperation(odPair));
         it->second.Insert(daosKey.fAkey, pageIov);

         RNTupleLocator locator;
         locator.SetType(RNTupleLocator::kTypeObject64);
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
   // clang-format off
   fDaosContainer->WriteSingleAkey(
      bufPageListZip.get(),
      szPageListZip,
      daos_obj_id_t{kOidLowPageList, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)},
      kDistributionKeyDefault,
      offsetData);
   // clang-format on
   RNTupleLocator result;
   result.SetType(RNTupleLocator::kTypeObject64);
   result.SetNBytesOnStorage(szPageListZip);
   result.SetPosition(RNTupleLocatorObject64{offsetData});
   fCounters->fSzWritePayload.Add(static_cast<int64_t>(szPageListZip));
   return result;
}

ROOT::Internal::RNTupleLink
ROOT::Experimental::Internal::RPageSinkDaos::CommitDatasetImpl(unsigned char *serializedFooter, std::uint32_t length)
{
   auto bufFooterZip = MakeUninitArray<unsigned char>(length);
   auto szFooterZip =
      RNTupleCompressor::Zip(serializedFooter, length, GetWriteOptions().GetCompression(), bufFooterZip.get());
   WriteNTupleFooter(bufFooterZip.get(), szFooterZip, length);
   WriteNTupleAnchor();

   // TODO: return the proper anchor locator+length
   return {};
}

void ROOT::Experimental::Internal::RPageSinkDaos::WriteNTupleHeader(const void *data, size_t nbytes, size_t lenHeader)
{
   fDaosContainer->WriteSingleAkey(
      data, nbytes, daos_obj_id_t{kOidLowMetadata, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)},
      kDistributionKeyDefault, kAttributeKeyHeader);
   fNTupleAnchor.fLenHeader = lenHeader;
   fNTupleAnchor.fNBytesHeader = nbytes;
}

void ROOT::Experimental::Internal::RPageSinkDaos::WriteNTupleFooter(const void *data, size_t nbytes, size_t lenFooter)
{
   fDaosContainer->WriteSingleAkey(
      data, nbytes, daos_obj_id_t{kOidLowMetadata, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)},
      kDistributionKeyDefault, kAttributeKeyFooter);
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
      kDistributionKeyDefault, kAttributeKeyAnchor, kCidAnchor);
}

std::unique_ptr<ROOT::Internal::RPageSink>
ROOT::Experimental::Internal::RPageSinkDaos::CloneAsHidden(std::string_view /*name*/,
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
   auto pool = std::make_unique<RDaosPool>(args.fPoolLabel);
   fDaosContainer = std::make_unique<RDaosContainer>(std::move(pool), args.fContainerLabel);
}

ROOT::Experimental::Internal::RPageSourceDaos::~RPageSourceDaos()
{
   StopClusterPoolBackgroundThread();
}

void ROOT::Experimental::Internal::RPageSourceDaos::LoadStructureImpl()
{
   RDaosContainerNTupleLocator ntupleLocator;
   std::tie(ntupleLocator, fDescriptorBuilder) =
      RDaosContainerNTupleLocator::LocateNTuple(*fDaosContainer, fNTupleName);
   if (!ntupleLocator.IsValid()) {
      throw ROOT::RException(
         R__FAIL("LoadStructureImpl: requested ntuple '" + fNTupleName + "' is not present in DAOS container."));
   }
   fAnchor = *ntupleLocator.fAnchor;
   fNTupleIndex = ntupleLocator.GetIndex();

   fDaosContainer->SetDefaultObjectClass(fAnchor.fObjClass);

   // Reserve enough space for the compressed and the uncompressed header/footer (see AttachImpl)
   const auto bufSize =
      fAnchor.fNBytesHeader + fAnchor.fNBytesFooter + std::max(fAnchor.fLenHeader, fAnchor.fLenFooter);
   fStructureBuffer.fBuffer = MakeUninitArray<unsigned char>(bufSize);
   fStructureBuffer.fPtrHeader = fStructureBuffer.fBuffer.get();
   fStructureBuffer.fPtrFooter = fStructureBuffer.fBuffer.get() + fAnchor.fNBytesHeader;

   int err;
   daos_obj_id_t oidMetadata{kOidLowMetadata, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)};

   if ((err = fDaosContainer->ReadSingleAkey(fStructureBuffer.fPtrHeader, fAnchor.fNBytesHeader, oidMetadata,
                                             kDistributionKeyDefault, kAttributeKeyHeader))) {
      throw ROOT::RException(R__FAIL("LoadStructureImpl: cannot load header: " + std::to_string(err)));
   }

   if ((err = fDaosContainer->ReadSingleAkey(fStructureBuffer.fPtrFooter, fAnchor.fNBytesFooter, oidMetadata,
                                             kDistributionKeyDefault, kAttributeKeyFooter))) {
      throw ROOT::RException(R__FAIL("LoadStructureImpl: cannot load footer: " + std::to_string(err)));
   }
}

ROOT::RNTupleDescriptor ROOT::Experimental::Internal::RPageSourceDaos::AttachImpl()
{
   auto unzipBuf = reinterpret_cast<unsigned char *>(fStructureBuffer.fPtrFooter) + fAnchor.fNBytesFooter;

   RNTupleDecompressor::Unzip(fStructureBuffer.fPtrHeader, fAnchor.fNBytesHeader, fAnchor.fLenHeader, unzipBuf);
   RNTupleSerializer::DeserializeHeader(unzipBuf, fAnchor.fLenHeader, fDescriptorBuilder);

   RNTupleDecompressor::Unzip(fStructureBuffer.fPtrFooter, fAnchor.fNBytesFooter, fAnchor.fLenFooter, unzipBuf);
   RNTupleSerializer::DeserializeFooter(unzipBuf, fAnchor.fLenFooter, fDescriptorBuilder);

   if (fDescriptorBuilder.GetDescriptor().GetName() != fNTupleName) {
      // Hash already taken by a differently-named ntuple.
      throw ROOT::RException(R__FAIL("LocateNTuple: ntuple name '" + fNTupleName + "' unavailable in this container."));
   }

   return fDescriptorBuilder.MoveDescriptor();
}

void ROOT::Experimental::Internal::RPageSourceDaos::LoadPageListImpl(const RNTupleLocator &locator,
                                                                     unsigned char *buffer)
{
   daos_obj_id_t oidPageList{kOidLowPageList, static_cast<decltype(daos_obj_id_t::hi)>(fNTupleIndex)};
   fDaosContainer->ReadSingleAkey(buffer, locator.GetNBytesOnStorage(), oidPageList, kDistributionKeyDefault,
                                  locator.GetPosition<RNTupleLocatorObject64>().GetLocation());
}

std::string ROOT::Experimental::Internal::RPageSourceDaos::GetObjectClass() const
{
   return fDaosContainer->GetDefaultObjectClass().ToString();
}

void ROOT::Experimental::Internal::RPageSourceDaos::LoadSealedPageImpl(const RNTupleLocator &locator,
                                                                       RSealedPage &sealedPage)
{
   RDaosKey daosKey = GetPageDaosKey(fNTupleIndex, locator.GetPosition<RNTupleLocatorObject64>().GetLocation());
   fDaosContainer->ReadSingleAkey(const_cast<void *>(sealedPage.GetBuffer()), sealedPage.GetBufferSize(), daosKey.fOid,
                                  daosKey.fDkey, daosKey.fAkey);
}

std::unique_ptr<ROOT::Internal::RPageSource> ROOT::Experimental::Internal::RPageSourceDaos::CloneImpl() const
{
   auto clone = std::make_unique<RPageSourceDaos>(fNTupleName, fURI, fOptions);
   clone->fAnchor = fAnchor;
   clone->fNTupleIndex = fNTupleIndex;
   if (!fAnchor.fObjClass.empty())
      clone->fDaosContainer->SetDefaultObjectClass(fAnchor.fObjClass);
   return clone;
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

         RDaosKey daosKey = GetPageDaosKey(fNTupleIndex, sealedLoc.fPageId);
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

std::unique_ptr<ROOT::Internal::RPageSource>
ROOT::Experimental::Internal::RPageSourceDaos::OpenWithDifferentAnchor(const ROOT::Internal::RNTupleLink &,
                                                                       const ROOT::RNTupleReadOptions &)
{
   throw ROOT::RException(R__FAIL("method not implemented"));
}
