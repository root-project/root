/// \file ROOT/RNTupleSerialize.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
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

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

enum class EColumnType;
class RClusterDescriptor;
class RClusterDescriptorBuilder;
class RNTupleDescriptor;
class RNTupleDescriptorBuilder;


namespace Internal {

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
   /// In order to handle changes to the serialization routine in future ntuple versions
   static constexpr std::uint16_t kEnvelopeCurrentVersion = 1;
   static constexpr std::uint16_t kEnvelopeMinVersion     = 1;
   static constexpr std::uint32_t kReleaseCandidateTag    = 1;

   static constexpr std::uint16_t kFlagRepetitiveField = 0x01;
   static constexpr std::uint16_t kFlagAliasField      = 0x02;

   static constexpr std::uint32_t kFlagSortAscColumn     = 0x01;
   static constexpr std::uint32_t kFlagSortDesColumn     = 0x02;
   static constexpr std::uint32_t kFlagNonNegativeColumn = 0x04;

   static constexpr DescriptorId_t kZeroFieldId = std::uint64_t(-2);

   struct REnvelopeLink {
      std::uint32_t fUnzippedSize = 0;
      RNTupleLocator fLocator;
   };

   struct RClusterSummary {
      std::uint64_t fFirstEntry = 0;
      std::uint64_t fNEntries = 0;
      /// -1 for "all columns"
      std::int32_t fColumnGroupID = -1;
   };

   struct RClusterGroup {
      std::uint32_t fNClusters = 0;
      REnvelopeLink fPageListEnvelopeLink;
   };

   /// The serialization context is used for the piecewise serialization of a descriptor.  During header serialization,
   /// the mapping of in-memory field and column IDs to physical IDs is built so that it can be used for the
   /// footer serialization in a second step.
   class RContext {
   private:
      std::uint32_t fHeaderSize = 0;
      std::uint32_t fHeaderCrc32 = 0;
      std::map<DescriptorId_t, DescriptorId_t> fMem2PhysFieldIDs;
      std::map<DescriptorId_t, DescriptorId_t> fMem2PhysColumnIDs;
      std::map<DescriptorId_t, DescriptorId_t> fMem2PhysClusterIDs;
      std::map<DescriptorId_t, DescriptorId_t> fMem2PhysClusterGroupIDs;
      std::vector<DescriptorId_t> fPhys2MemFieldIDs;
      std::vector<DescriptorId_t> fPhys2MemColumnIDs;
      std::vector<DescriptorId_t> fPhys2MemClusterIDs;
      std::vector<DescriptorId_t> fPhys2MemClusterGroupIDs;

   public:
      void SetHeaderSize(std::uint32_t size) { fHeaderSize = size; }
      std::uint32_t GetHeaderSize() const { return fHeaderSize; }
      void SetHeaderCRC32(std::uint32_t crc32) { fHeaderCrc32 = crc32; }
      std::uint32_t GetHeaderCRC32() const { return fHeaderCrc32; }
      DescriptorId_t MapFieldId(DescriptorId_t memId) {
         auto physId = fPhys2MemFieldIDs.size();
         fMem2PhysFieldIDs[memId] = physId;
         fPhys2MemFieldIDs.push_back(memId);
         return physId;
      }
      DescriptorId_t MapColumnId(DescriptorId_t memId) {
         auto physId = fPhys2MemColumnIDs.size();
         fMem2PhysColumnIDs[memId] = physId;
         fPhys2MemColumnIDs.push_back(memId);
         return physId;
      }
      DescriptorId_t MapClusterId(DescriptorId_t memId) {
         auto physId = fPhys2MemClusterIDs.size();
         fMem2PhysClusterIDs[memId] = physId;
         fPhys2MemClusterIDs.push_back(memId);
         return physId;
      }
      DescriptorId_t MapClusterGroupId(DescriptorId_t memId)
      {
         auto physId = fPhys2MemClusterGroupIDs.size();
         fMem2PhysClusterGroupIDs[memId] = physId;
         fPhys2MemClusterGroupIDs.push_back(memId);
         return physId;
      }
      DescriptorId_t GetPhysFieldId(DescriptorId_t memId) const { return fMem2PhysFieldIDs.at(memId); }
      DescriptorId_t GetPhysColumnId(DescriptorId_t memId) const { return fMem2PhysColumnIDs.at(memId); }
      DescriptorId_t GetPhysClusterId(DescriptorId_t memId) const { return fMem2PhysClusterIDs.at(memId); }
      DescriptorId_t GetPhysClusterGroupId(DescriptorId_t memId) const { return fMem2PhysClusterGroupIDs.at(memId); }
      DescriptorId_t GetMemFieldId(DescriptorId_t physId) const { return fPhys2MemFieldIDs[physId]; }
      DescriptorId_t GetMemColumnId(DescriptorId_t physId) const { return fPhys2MemColumnIDs[physId]; }
      DescriptorId_t GetMemClusterId(DescriptorId_t physId) const { return fPhys2MemClusterIDs[physId]; }
      DescriptorId_t GetMemClusterGroupId(DescriptorId_t physId) const { return fPhys2MemClusterGroupIDs[physId]; }
   };

   /// Writes a CRC32 checksum of the byte range given by data and length.
   static std::uint32_t SerializeCRC32(const unsigned char *data, std::uint32_t length,
                                       std::uint32_t &crc32, void *buffer);
   /// Expects a CRC32 checksum in the 4 bytes following data + length and verifies it.
   static RResult<void> VerifyCRC32(const unsigned char *data, std::uint32_t length, std::uint32_t &crc32);
   static RResult<void> VerifyCRC32(const unsigned char *data, std::uint32_t length);

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
   static RResult<std::uint32_t> DeserializeString(const void *buffer, std::uint32_t bufSize, std::string &val);

   /// While we could just interpret the enums as ints, we make the translation explicit
   /// in order to avoid accidentally changing the on-disk numbers when adjusting the enum classes.
   static std::uint16_t SerializeFieldStructure(ROOT::Experimental::ENTupleStructure structure, void *buffer);
   static std::uint16_t SerializeColumnType(ROOT::Experimental::EColumnType type, void *buffer);
   static RResult<std::uint16_t> DeserializeFieldStructure(const void *buffer, ROOT::Experimental::ENTupleStructure &structure);
   static RResult<std::uint16_t> DeserializeColumnType(const void *buffer, ROOT::Experimental::EColumnType &type);

   static std::uint32_t SerializeEnvelopePreamble(void *buffer);
   static std::uint32_t SerializeEnvelopePostscript(const unsigned char *envelope, std::uint32_t size, void *buffer);
   static std::uint32_t SerializeEnvelopePostscript(const unsigned char *envelope, std::uint32_t size,
                                                    std::uint32_t &crc32, void *buffer);
   // The bufSize must include the 4 bytes for the final CRC32 checksum.
   static RResult<std::uint32_t> DeserializeEnvelope(const void *buffer, std::uint32_t bufSize);
   static RResult<std::uint32_t> DeserializeEnvelope(const void *buffer, std::uint32_t bufSize, std::uint32_t &crc32);

   static std::uint32_t SerializeRecordFramePreamble(void *buffer);
   static std::uint32_t SerializeListFramePreamble(std::uint32_t nitems, void *buffer);
   static std::uint32_t SerializeFramePostscript(void *frame, std::int32_t size);
   static RResult<std::uint32_t> DeserializeFrameHeader(const void *buffer, std::uint32_t bufSize,
                                                        std::uint32_t &frameSize, std::uint32_t &nitems);
   static RResult<std::uint32_t> DeserializeFrameHeader(const void *buffer, std::uint32_t bufSize,
                                                        std::uint32_t &frameSize);

   // An empty flags vector will be serialized as a single, zero feature flag
   // The most significant bit in every flag is reserved and must _not_ be set
   static std::uint32_t SerializeFeatureFlags(const std::vector<std::int64_t> &flags, void *buffer);
   static RResult<std::uint32_t> DeserializeFeatureFlags(const void *buffer, std::uint32_t bufSize,
                                                         std::vector<std::int64_t> &flags);

   static std::uint32_t SerializeLocator(const RNTupleLocator &locator, void *buffer);
   static std::uint32_t SerializeEnvelopeLink(const REnvelopeLink &envelopeLink, void *buffer);
   static RResult<std::uint32_t> DeserializeLocator(const void *buffer, std::uint32_t bufSize, RNTupleLocator &locator);
   static RResult<std::uint32_t> DeserializeEnvelopeLink(const void *buffer, std::uint32_t bufSize,
                                                         REnvelopeLink &envelopeLink);

   static std::uint32_t SerializeClusterSummary(const RClusterSummary &clusterSummary, void *buffer);
   static std::uint32_t SerializeClusterGroup(const RClusterGroup &clusterGroup, void *buffer);
   static RResult<std::uint32_t> DeserializeClusterSummary(const void *buffer, std::uint32_t bufSize,
                                                           RClusterSummary &clusterSummary);
   static RResult<std::uint32_t> DeserializeClusterGroup(const void *buffer, std::uint32_t bufSize,
                                                         RClusterGroup &clusterGroup);

   static RContext SerializeHeaderV1(void *buffer, const RNTupleDescriptor &desc);
   static std::uint32_t SerializePageListV1(void *buffer,
                                            const RNTupleDescriptor &desc,
                                            std::span<DescriptorId_t> physClusterIDs,
                                            const RContext &context);
   static std::uint32_t SerializeFooterV1(void *buffer, const RNTupleDescriptor &desc, const RContext &context);

   static RResult<void> DeserializeHeaderV1(const void *buffer,
                                            std::uint32_t bufSize,
                                            RNTupleDescriptorBuilder &descBuilder);
   static RResult<void> DeserializeFooterV1(const void *buffer,
                                            std::uint32_t bufSize,
                                            RNTupleDescriptorBuilder &descBuilder);
   // The clusters vector must be initialized with the cluster summaries corresponding to the page list
   static RResult<void> DeserializePageListV1(const void *buffer,
                                              std::uint32_t bufSize,
                                              std::vector<RClusterDescriptorBuilder> &clusters);
}; // class RNTupleSerializer

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleSerialize
