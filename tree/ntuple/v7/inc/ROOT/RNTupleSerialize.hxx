/// \file ROOT/RNTupleSerialize.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Javier Lopez-Gomez <javier.lopez.gomez@cern.ch>
/// \date 2021-08-02
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleSerialize
#define ROOT7_RNTupleSerialize

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RSpan.hxx>

#include <Rtypes.h>

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

class TVirtualStreamerInfo;

namespace ROOT {
namespace Experimental {

enum class EColumnType;
enum class EExtraTypeInfoIds;
class RClusterDescriptor;
class RNTupleDescriptor;

namespace Internal {

class RClusterDescriptorBuilder;
class RNTupleDescriptorBuilder;

// clang-format off
/**
\class ROOT::Experimental::Internal::RNTupleSerializer
\ingroup NTuple
\brief A helper class for serializing and deserialization of the RNTuple binary format

All serialization and deserialization routines return the number of bytes processed (written or read).

The serialization routines can be called with a nullptr buffer, in which case only the size required to perform
a serialization is returned. Deserialization routines must be called with a buffer that is sufficiently large.

Deserialization errors throw exceptions. Only when indicated or when passed as a parameter is the buffer size checked.
*/
// clang-format on
class RNTupleSerializer {
public:
   static constexpr std::uint16_t kEnvelopeTypeHeader = 0x01;
   static constexpr std::uint16_t kEnvelopeTypeFooter = 0x02;
   static constexpr std::uint16_t kEnvelopeTypePageList = 0x03;

   static constexpr std::uint16_t kFlagRepetitiveField = 0x01;
   static constexpr std::uint16_t kFlagProjectedField = 0x02;
   static constexpr std::uint16_t kFlagHasTypeChecksum = 0x04;

   static constexpr std::uint16_t kFlagDeferredColumn = 0x01;
   static constexpr std::uint16_t kFlagHasValueRange = 0x02;

   static constexpr DescriptorId_t kZeroFieldId = std::uint64_t(-2);

   static constexpr int64_t kSuppressedColumnMarker = std::numeric_limits<std::int64_t>::min();

   // In the page sink and the streamer field, the seen streamer infos are stored in a map
   // with the unique streamer info number being the key. Sorted by unique number.
   using StreamerInfoMap_t = std::map<Int_t, TVirtualStreamerInfo *>;

   struct REnvelopeLink {
      std::uint64_t fLength = 0;
      RNTupleLocator fLocator;
   };

   struct RClusterSummary {
      std::uint64_t fFirstEntry = 0;
      std::uint64_t fNEntries = 0;
      std::uint8_t fFlags = 0;
   };

   struct RClusterGroup {
      std::uint64_t fMinEntry = 0;
      std::uint64_t fEntrySpan = 0;
      std::uint32_t fNClusters = 0;
      REnvelopeLink fPageListEnvelopeLink;
   };

   /// The serialization context is used for the piecewise serialization of a descriptor.  During header serialization,
   /// the mapping of in-memory field and column IDs to on-disk IDs is built so that it can be used for the
   /// footer serialization in a second step.
   class RContext {
   private:
      std::uint64_t fHeaderSize = 0;
      std::uint64_t fHeaderXxHash3 = 0;
      std::map<DescriptorId_t, DescriptorId_t> fMem2OnDiskFieldIDs;
      std::map<DescriptorId_t, DescriptorId_t> fMem2OnDiskColumnIDs;
      std::map<DescriptorId_t, DescriptorId_t> fMem2OnDiskClusterIDs;
      std::map<DescriptorId_t, DescriptorId_t> fMem2OnDiskClusterGroupIDs;
      std::vector<DescriptorId_t> fOnDisk2MemFieldIDs;
      std::vector<DescriptorId_t> fOnDisk2MemColumnIDs;
      std::vector<DescriptorId_t> fOnDisk2MemClusterIDs;
      std::vector<DescriptorId_t> fOnDisk2MemClusterGroupIDs;
      std::size_t fHeaderExtensionOffset = -1U;

   public:
      void SetHeaderSize(std::uint64_t size) { fHeaderSize = size; }
      std::uint64_t GetHeaderSize() const { return fHeaderSize; }
      void SetHeaderXxHash3(std::uint64_t xxhash3) { fHeaderXxHash3 = xxhash3; }
      std::uint64_t GetHeaderXxHash3() const { return fHeaderXxHash3; }
      /// Map an in-memory field ID to its on-disk counterpart. It is allowed to call this function multiple times for
      /// the same `memId`, in which case the return value is the on-disk ID assigned on the first call.
      DescriptorId_t MapFieldId(DescriptorId_t memId)
      {
         auto onDiskId = fOnDisk2MemFieldIDs.size();
         const auto &p = fMem2OnDiskFieldIDs.try_emplace(memId, onDiskId);
         if (p.second)
            fOnDisk2MemFieldIDs.push_back(memId);
         return (*p.first).second;
      }
      /// Map an in-memory column ID to its on-disk counterpart. It is allowed to call this function multiple times for
      /// the same `memId`, in which case the return value is the on-disk ID assigned on the first call.
      /// Note that we only map physical column IDs.  Logical column IDs of alias columns are shifted before the
      /// serialization of the extension header.  Also, we only need to query physical column IDs for the page list
      /// serialization.
      DescriptorId_t MapPhysicalColumnId(DescriptorId_t memId)
      {
         auto onDiskId = fOnDisk2MemColumnIDs.size();
         const auto &p = fMem2OnDiskColumnIDs.try_emplace(memId, onDiskId);
         if (p.second)
            fOnDisk2MemColumnIDs.push_back(memId);
         return (*p.first).second;
      }
      DescriptorId_t MapClusterId(DescriptorId_t memId)
      {
         auto onDiskId = fOnDisk2MemClusterIDs.size();
         fMem2OnDiskClusterIDs[memId] = onDiskId;
         fOnDisk2MemClusterIDs.push_back(memId);
         return onDiskId;
      }
      DescriptorId_t MapClusterGroupId(DescriptorId_t memId)
      {
         auto onDiskId = fOnDisk2MemClusterGroupIDs.size();
         fMem2OnDiskClusterGroupIDs[memId] = onDiskId;
         fOnDisk2MemClusterGroupIDs.push_back(memId);
         return onDiskId;
      }
      /// Map in-memory field and column IDs to their on-disk counterparts. This function is unconditionally called
      /// during header serialization.  This function must be manually called after an incremental schema update as page
      /// list serialization requires all columns to be mapped.
      void MapSchema(const RNTupleDescriptor &desc, bool forHeaderExtension);

      DescriptorId_t GetOnDiskFieldId(DescriptorId_t memId) const { return fMem2OnDiskFieldIDs.at(memId); }
      DescriptorId_t GetOnDiskColumnId(DescriptorId_t memId) const { return fMem2OnDiskColumnIDs.at(memId); }
      DescriptorId_t GetOnDiskClusterId(DescriptorId_t memId) const { return fMem2OnDiskClusterIDs.at(memId); }
      DescriptorId_t GetOnDiskClusterGroupId(DescriptorId_t memId) const
      {
         return fMem2OnDiskClusterGroupIDs.at(memId);
      }
      DescriptorId_t GetMemFieldId(DescriptorId_t onDiskId) const { return fOnDisk2MemFieldIDs[onDiskId]; }
      DescriptorId_t GetMemColumnId(DescriptorId_t onDiskId) const { return fOnDisk2MemColumnIDs[onDiskId]; }
      DescriptorId_t GetMemClusterId(DescriptorId_t onDiskId) const { return fOnDisk2MemClusterIDs[onDiskId]; }
      DescriptorId_t GetMemClusterGroupId(DescriptorId_t onDiskId) const
      {
         return fOnDisk2MemClusterGroupIDs[onDiskId];
      }

      /// Return a vector containing the in-memory field ID for each on-disk counterpart, in order, i.e. the `i`-th
      /// value corresponds to the in-memory field ID for `i`-th on-disk ID
      const std::vector<DescriptorId_t> &GetOnDiskFieldList() const { return fOnDisk2MemFieldIDs; }
      /// Mark the first on-disk field ID that is part of the schema extension
      void BeginHeaderExtension() { fHeaderExtensionOffset = fOnDisk2MemFieldIDs.size(); }
      /// Return the offset of the first element in `fOnDisk2MemFieldIDs` that is part of the schema extension
      std::size_t GetHeaderExtensionOffset() const { return fHeaderExtensionOffset; }
   };

   /// Writes a XxHash-3 64bit checksum of the byte range given by data and length.
   static std::uint32_t
   SerializeXxHash3(const unsigned char *data, std::uint64_t length, std::uint64_t &xxhash3, void *buffer);
   /// Expects an xxhash3 checksum in the 8 bytes following data + length and verifies it.
   static RResult<void> VerifyXxHash3(const unsigned char *data, std::uint64_t length, std::uint64_t &xxhash3);
   static RResult<void> VerifyXxHash3(const unsigned char *data, std::uint64_t length);

   static std::uint32_t SerializeInt16(std::int16_t val, void *buffer);
   static std::uint32_t DeserializeInt16(const void *buffer, std::int16_t &val);
   static std::uint32_t SerializeUInt16(std::uint16_t val, void *buffer);
   static std::uint32_t DeserializeUInt16(const void *buffer, std::uint16_t &val);

   static std::uint32_t SerializeInt32(std::int32_t val, void *buffer);
   static std::uint32_t DeserializeInt32(const void *buffer, std::int32_t &val);
   static std::uint32_t SerializeUInt32(std::uint32_t val, void *buffer);
   static std::uint32_t DeserializeUInt32(const void *buffer, std::uint32_t &val);

   static std::uint32_t SerializeInt64(std::int64_t val, void *buffer);
   static std::uint32_t DeserializeInt64(const void *buffer, std::int64_t &val);
   static std::uint32_t SerializeUInt64(std::uint64_t val, void *buffer);
   static std::uint32_t DeserializeUInt64(const void *buffer, std::uint64_t &val);

   static std::uint32_t SerializeString(const std::string &val, void *buffer);
   static RResult<std::uint32_t> DeserializeString(const void *buffer, std::uint64_t bufSize, std::string &val);

   /// While we could just interpret the enums as ints, we make the translation explicit
   /// in order to avoid accidentally changing the on-disk numbers when adjusting the enum classes.
   static std::uint32_t SerializeFieldStructure(ROOT::Experimental::ENTupleStructure structure, void *buffer);
   static std::uint32_t SerializeColumnType(ROOT::Experimental::EColumnType type, void *buffer);
   static std::uint32_t SerializeExtraTypeInfoId(ROOT::Experimental::EExtraTypeInfoIds id, void *buffer);
   static RResult<std::uint32_t>
   DeserializeFieldStructure(const void *buffer, ROOT::Experimental::ENTupleStructure &structure);
   static RResult<std::uint32_t> DeserializeColumnType(const void *buffer, ROOT::Experimental::EColumnType &type);
   static RResult<std::uint32_t>
   DeserializeExtraTypeInfoId(const void *buffer, ROOT::Experimental::EExtraTypeInfoIds &id);

   static std::uint32_t SerializeEnvelopePreamble(std::uint16_t envelopeType, void *buffer);
   static std::uint32_t SerializeEnvelopePostscript(unsigned char *envelope, std::uint64_t size);
   static std::uint32_t
   SerializeEnvelopePostscript(unsigned char *envelope, std::uint64_t size, std::uint64_t &xxhash3);
   // The bufSize must include the 8 bytes for the final xxhash3 checksum.
   static RResult<std::uint32_t>
   DeserializeEnvelope(const void *buffer, std::uint64_t bufSize, std::uint16_t expectedType);
   static RResult<std::uint32_t>
   DeserializeEnvelope(const void *buffer, std::uint64_t bufSize, std::uint16_t expectedType, std::uint64_t &xxhash3);

   static std::uint32_t SerializeRecordFramePreamble(void *buffer);
   static std::uint32_t SerializeListFramePreamble(std::uint32_t nitems, void *buffer);
   static std::uint32_t SerializeFramePostscript(void *frame, std::uint64_t size);
   static RResult<std::uint32_t>
   DeserializeFrameHeader(const void *buffer, std::uint64_t bufSize, std::uint64_t &frameSize, std::uint32_t &nitems);
   static RResult<std::uint32_t>
   DeserializeFrameHeader(const void *buffer, std::uint64_t bufSize, std::uint64_t &frameSize);

   // An empty flags vector will be serialized as a single, zero feature flag
   // The most significant bit in every flag is reserved and must _not_ be set
   static std::uint32_t SerializeFeatureFlags(const std::vector<std::uint64_t> &flags, void *buffer);
   static RResult<std::uint32_t>
   DeserializeFeatureFlags(const void *buffer, std::uint64_t bufSize, std::vector<std::uint64_t> &flags);

   static std::uint32_t SerializeLocator(const RNTupleLocator &locator, void *buffer);
   static std::uint32_t SerializeEnvelopeLink(const REnvelopeLink &envelopeLink, void *buffer);
   static RResult<std::uint32_t> DeserializeLocator(const void *buffer, std::uint64_t bufSize, RNTupleLocator &locator);
   static RResult<std::uint32_t>
   DeserializeEnvelopeLink(const void *buffer, std::uint64_t bufSize, REnvelopeLink &envelopeLink);

   static std::uint32_t SerializeClusterSummary(const RClusterSummary &clusterSummary, void *buffer);
   static std::uint32_t SerializeClusterGroup(const RClusterGroup &clusterGroup, void *buffer);
   static RResult<std::uint32_t>
   DeserializeClusterSummary(const void *buffer, std::uint64_t bufSize, RClusterSummary &clusterSummary);
   static RResult<std::uint32_t>
   DeserializeClusterGroup(const void *buffer, std::uint64_t bufSize, RClusterGroup &clusterGroup);

   /// Serialize the schema description in `desc` into `buffer`. If `forHeaderExtension` is true, serialize only the
   /// fields and columns tagged as part of the header extension (see `RNTupleDescriptorBuilder::BeginHeaderExtension`).
   static std::uint32_t SerializeSchemaDescription(void *buffer, const RNTupleDescriptor &desc, const RContext &context,
                                                   bool forHeaderExtension = false);
   static RResult<std::uint32_t>
   DeserializeSchemaDescription(const void *buffer, std::uint64_t bufSize, RNTupleDescriptorBuilder &descBuilder);

   static RContext SerializeHeader(void *buffer, const RNTupleDescriptor &desc);
   static std::uint32_t SerializePageList(void *buffer, const RNTupleDescriptor &desc,
                                          std::span<DescriptorId_t> physClusterIDs, const RContext &context);
   static std::uint32_t SerializeFooter(void *buffer, const RNTupleDescriptor &desc, const RContext &context);

   static RResult<void>
   DeserializeHeader(const void *buffer, std::uint64_t bufSize, RNTupleDescriptorBuilder &descBuilder);
   static RResult<void>
   DeserializeFooter(const void *buffer, std::uint64_t bufSize, RNTupleDescriptorBuilder &descBuilder);
   // The clusters vector must be initialized with the cluster summaries corresponding to the page list
   static RResult<void> DeserializePageList(const void *buffer, std::uint64_t bufSize, DescriptorId_t clusterGroupId,
                                            RNTupleDescriptor &desc);

   // Helper functions to (de-)serialize the streamer info type extra information
   static std::string SerializeStreamerInfos(const StreamerInfoMap_t &infos);
   static RResult<StreamerInfoMap_t> DeserializeStreamerInfos(const std::string &extraTypeInfoContent);
}; // class RNTupleSerializer

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleSerialize
