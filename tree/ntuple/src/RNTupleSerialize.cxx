/// \file RNTupleSerialize.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Javier Lopez-Gomez <javier.lopez.gomez@cern.ch>
/// \date 2021-08-02

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RColumnElementBase.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleUtils.hxx>

#include <RVersion.h>
#include <TBufferFile.h>
#include <TClass.h>
#include <TList.h>
#include <TStreamerInfo.h>
#include <TVirtualStreamerInfo.h>
#include <xxhash.h>

#include <cassert>
#include <cmath>
#include <cstring> // for memcpy
#include <deque>
#include <functional>
#include <limits>
#include <set>
#include <unordered_map>

using ROOT::Internal::RClusterDescriptorBuilder;
using ROOT::Internal::RClusterGroupDescriptorBuilder;
using ROOT::Internal::RColumnDescriptorBuilder;
using ROOT::Internal::RExtraTypeInfoDescriptorBuilder;
using ROOT::Internal::RFieldDescriptorBuilder;
using ROOT::Internal::RNTupleDescriptorBuilder;

namespace {
using RNTupleSerializer = ROOT::Internal::RNTupleSerializer;

ROOT::RResult<std::uint32_t> SerializeField(const ROOT::RFieldDescriptor &fieldDesc,
                                            ROOT::DescriptorId_t onDiskParentId,
                                            ROOT::DescriptorId_t onDiskProjectionSourceId, void *buffer)
{

   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   pos += RNTupleSerializer::SerializeRecordFramePreamble(*where);

   pos += RNTupleSerializer::SerializeUInt32(fieldDesc.GetFieldVersion(), *where);
   pos += RNTupleSerializer::SerializeUInt32(fieldDesc.GetTypeVersion(), *where);
   pos += RNTupleSerializer::SerializeUInt32(onDiskParentId, *where);
   if (auto res = RNTupleSerializer::SerializeFieldStructure(fieldDesc.GetStructure(), *where)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   std::uint16_t flags = 0;
   if (fieldDesc.GetNRepetitions() > 0)
      flags |= RNTupleSerializer::kFlagRepetitiveField;
   if (fieldDesc.IsProjectedField())
      flags |= RNTupleSerializer::kFlagProjectedField;
   if (fieldDesc.GetTypeChecksum().has_value())
      flags |= RNTupleSerializer::kFlagHasTypeChecksum;
   pos += RNTupleSerializer::SerializeUInt16(flags, *where);

   pos += RNTupleSerializer::SerializeString(fieldDesc.GetFieldName(), *where);
   pos += RNTupleSerializer::SerializeString(fieldDesc.GetTypeName(), *where);
   pos += RNTupleSerializer::SerializeString(fieldDesc.GetTypeAlias(), *where);
   pos += RNTupleSerializer::SerializeString(fieldDesc.GetFieldDescription(), *where);

   if (flags & RNTupleSerializer::kFlagRepetitiveField) {
      pos += RNTupleSerializer::SerializeUInt64(fieldDesc.GetNRepetitions(), *where);
   }
   if (flags & RNTupleSerializer::kFlagProjectedField) {
      pos += RNTupleSerializer::SerializeUInt32(onDiskProjectionSourceId, *where);
   }
   if (flags & RNTupleSerializer::kFlagHasTypeChecksum) {
      pos += RNTupleSerializer::SerializeUInt32(fieldDesc.GetTypeChecksum().value(), *where);
   }

   auto size = pos - base;
   RNTupleSerializer::SerializeFramePostscript(base, size);

   return size;
}

// clang-format off
/// Serialize, in order, fields enumerated in `fieldList` to `buffer`.  `firstOnDiskId` specifies the on-disk ID for the
/// first element in the `fieldList` sequence. Before calling this function `RContext::MapSchema()` should have been
/// called on `context` in order to map in-memory field IDs to their on-disk counterpart.
/// \return The number of bytes written to the output buffer; if `buffer` is `nullptr` no data is serialized and the
/// required buffer size is returned
// clang-format on
ROOT::RResult<std::uint32_t>
SerializeFieldList(const ROOT::RNTupleDescriptor &desc, std::span<const ROOT::DescriptorId_t> fieldList,
                   std::size_t firstOnDiskId, const ROOT::Internal::RNTupleSerializer::RContext &context, void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   auto fieldZeroId = desc.GetFieldZeroId();
   ROOT::DescriptorId_t onDiskFieldId = firstOnDiskId;
   for (auto fieldId : fieldList) {
      const auto &f = desc.GetFieldDescriptor(fieldId);
      auto onDiskParentId =
         (f.GetParentId() == fieldZeroId) ? onDiskFieldId : context.GetOnDiskFieldId(f.GetParentId());
      auto onDiskProjectionSourceId =
         f.IsProjectedField() ? context.GetOnDiskFieldId(f.GetProjectionSourceId()) : ROOT::kInvalidDescriptorId;
      if (auto res = SerializeField(f, onDiskParentId, onDiskProjectionSourceId, *where)) {
         pos += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }
      ++onDiskFieldId;
   }

   return pos - base;
}

ROOT::RResult<std::uint32_t>
DeserializeField(const void *buffer, std::uint64_t bufSize, ROOT::Internal::RFieldDescriptorBuilder &fieldDesc)
{
   using ENTupleStructure = ROOT::ENTupleStructure;

   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint64_t frameSize;
   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - base); };
   if (auto res = RNTupleSerializer::DeserializeFrameHeader(bytes, bufSize, frameSize)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   std::uint32_t fieldVersion;
   std::uint32_t typeVersion;
   std::uint32_t parentId;
   // initialize properly for call to SerializeFieldStructure()
   ENTupleStructure structure{ENTupleStructure::kLeaf};
   std::uint16_t flags;
   std::uint32_t result;
   if (auto res = RNTupleSerializer::SerializeFieldStructure(structure, nullptr)) {
      result = res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   if (fnFrameSizeLeft() < 3 * sizeof(std::uint32_t) + result + sizeof(std::uint16_t)) {
      return R__FAIL("field record frame too short");
   }
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fieldVersion);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, typeVersion);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, parentId);
   if (auto res = RNTupleSerializer::DeserializeFieldStructure(bytes, structure)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   bytes += RNTupleSerializer::DeserializeUInt16(bytes, flags);
   fieldDesc.FieldVersion(fieldVersion).TypeVersion(typeVersion).ParentId(parentId).Structure(structure);

   std::string fieldName;
   std::string typeName;
   std::string aliasName;
   std::string description;
   if (auto res = RNTupleSerializer::DeserializeString(bytes, fnFrameSizeLeft(), fieldName)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   if (auto res = RNTupleSerializer::DeserializeString(bytes, fnFrameSizeLeft(), typeName)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   if (auto res = RNTupleSerializer::DeserializeString(bytes, fnFrameSizeLeft(), aliasName)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   if (auto res = RNTupleSerializer::DeserializeString(bytes, fnFrameSizeLeft(), description)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   fieldDesc.FieldName(fieldName).TypeName(typeName).TypeAlias(aliasName).FieldDescription(description);

   if (flags & RNTupleSerializer::kFlagRepetitiveField) {
      if (fnFrameSizeLeft() < sizeof(std::uint64_t))
         return R__FAIL("field record frame too short");
      std::uint64_t nRepetitions;
      bytes += RNTupleSerializer::DeserializeUInt64(bytes, nRepetitions);
      fieldDesc.NRepetitions(nRepetitions);
   }

   if (flags & RNTupleSerializer::kFlagProjectedField) {
      if (fnFrameSizeLeft() < sizeof(std::uint32_t))
         return R__FAIL("field record frame too short");
      std::uint32_t projectionSourceId;
      bytes += RNTupleSerializer::DeserializeUInt32(bytes, projectionSourceId);
      fieldDesc.ProjectionSourceId(projectionSourceId);
   }

   if (flags & RNTupleSerializer::kFlagHasTypeChecksum) {
      if (fnFrameSizeLeft() < sizeof(std::uint32_t))
         return R__FAIL("field record frame too short");
      std::uint32_t typeChecksum;
      bytes += RNTupleSerializer::DeserializeUInt32(bytes, typeChecksum);
      fieldDesc.TypeChecksum(typeChecksum);
   }

   return frameSize;
}

ROOT::RResult<std::uint32_t> SerializePhysicalColumn(const ROOT::RColumnDescriptor &columnDesc,
                                                     const ROOT::Internal::RNTupleSerializer::RContext &context,
                                                     void *buffer)
{
   R__ASSERT(!columnDesc.IsAliasColumn());

   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   pos += RNTupleSerializer::SerializeRecordFramePreamble(*where);

   if (auto res = RNTupleSerializer::SerializeColumnType(columnDesc.GetType(), *where)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   pos += RNTupleSerializer::SerializeUInt16(columnDesc.GetBitsOnStorage(), *where);
   pos += RNTupleSerializer::SerializeUInt32(context.GetOnDiskFieldId(columnDesc.GetFieldId()), *where);
   std::uint16_t flags = 0;
   if (columnDesc.IsDeferredColumn())
      flags |= RNTupleSerializer::kFlagDeferredColumn;
   if (columnDesc.GetValueRange().has_value())
      flags |= RNTupleSerializer::kFlagHasValueRange;
   std::int64_t firstElementIdx = columnDesc.GetFirstElementIndex();
   if (columnDesc.IsSuppressedDeferredColumn())
      firstElementIdx = -firstElementIdx;
   pos += RNTupleSerializer::SerializeUInt16(flags, *where);
   pos += RNTupleSerializer::SerializeUInt16(columnDesc.GetRepresentationIndex(), *where);
   if (flags & RNTupleSerializer::kFlagDeferredColumn)
      pos += RNTupleSerializer::SerializeInt64(firstElementIdx, *where);
   if (flags & RNTupleSerializer::kFlagHasValueRange) {
      auto [min, max] = *columnDesc.GetValueRange();
      std::uint64_t intMin, intMax;
      static_assert(sizeof(min) == sizeof(intMin) && sizeof(max) == sizeof(intMax));
      memcpy(&intMin, &min, sizeof(min));
      memcpy(&intMax, &max, sizeof(max));
      pos += RNTupleSerializer::SerializeUInt64(intMin, *where);
      pos += RNTupleSerializer::SerializeUInt64(intMax, *where);
   }

   if (auto res = RNTupleSerializer::SerializeFramePostscript(buffer ? base : nullptr, pos - base)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   return pos - base;
}

ROOT::RResult<std::uint32_t> SerializeColumnsOfFields(const ROOT::RNTupleDescriptor &desc,
                                                      std::span<const ROOT::DescriptorId_t> fieldList,
                                                      const ROOT::Internal::RNTupleSerializer::RContext &context,
                                                      void *buffer, bool forHeaderExtension)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   const auto *xHeader = !forHeaderExtension ? desc.GetHeaderExtension() : nullptr;

   for (auto parentId : fieldList) {
      // If we're serializing the non-extended header and we already have a header extension (which may happen if
      // we load an RNTuple for incremental merging), we need to skip all the extended fields, as they need to be
      // written in the header extension, not in the regular header.
      if (xHeader && xHeader->ContainsField(parentId))
         continue;

      for (const auto &c : desc.GetColumnIterable(parentId)) {
         if (c.IsAliasColumn() || (xHeader && xHeader->ContainsExtendedColumnRepresentation(c.GetLogicalId())))
            continue;

         if (auto res = SerializePhysicalColumn(c, context, *where)) {
            pos += res.Unwrap();
         } else {
            return R__FORWARD_ERROR(res);
         }
      }
   }

   return pos - base;
}

ROOT::RResult<std::uint32_t>
DeserializeColumn(const void *buffer, std::uint64_t bufSize, ROOT::Internal::RColumnDescriptorBuilder &columnDesc)
{
   using ROOT::ENTupleColumnType;

   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint64_t frameSize;
   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - base); };
   if (auto res = RNTupleSerializer::DeserializeFrameHeader(bytes, bufSize, frameSize)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   // Initialize properly for SerializeColumnType
   ENTupleColumnType type{ENTupleColumnType::kIndex32};
   std::uint16_t bitsOnStorage;
   std::uint32_t fieldId;
   std::uint16_t flags;
   std::uint16_t representationIndex;
   std::int64_t firstElementIdx = 0;
   if (fnFrameSizeLeft() < RNTupleSerializer::SerializeColumnType(type, nullptr).Unwrap() + sizeof(std::uint16_t) +
                              2 * sizeof(std::uint32_t)) {
      return R__FAIL("column record frame too short");
   }
   if (auto res = RNTupleSerializer::DeserializeColumnType(bytes, type)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   bytes += RNTupleSerializer::DeserializeUInt16(bytes, bitsOnStorage);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fieldId);
   bytes += RNTupleSerializer::DeserializeUInt16(bytes, flags);
   bytes += RNTupleSerializer::DeserializeUInt16(bytes, representationIndex);
   if (flags & RNTupleSerializer::kFlagDeferredColumn) {
      if (fnFrameSizeLeft() < sizeof(std::uint64_t))
         return R__FAIL("column record frame too short");
      bytes += RNTupleSerializer::DeserializeInt64(bytes, firstElementIdx);
   }
   if (flags & RNTupleSerializer::kFlagHasValueRange) {
      if (fnFrameSizeLeft() < 2 * sizeof(std::uint64_t))
         return R__FAIL("field record frame too short");
      std::uint64_t minInt, maxInt;
      bytes += RNTupleSerializer::DeserializeUInt64(bytes, minInt);
      bytes += RNTupleSerializer::DeserializeUInt64(bytes, maxInt);
      double min, max;
      memcpy(&min, &minInt, sizeof(min));
      memcpy(&max, &maxInt, sizeof(max));
      columnDesc.ValueRange(min, max);
   }

   columnDesc.FieldId(fieldId).BitsOnStorage(bitsOnStorage).Type(type).RepresentationIndex(representationIndex);
   columnDesc.FirstElementIndex(std::abs(firstElementIdx));
   if (firstElementIdx < 0)
      columnDesc.SetSuppressedDeferred();

   return frameSize;
}

ROOT::RResult<std::uint32_t> SerializeExtraTypeInfo(const ROOT::RExtraTypeInfoDescriptor &desc, void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   pos += RNTupleSerializer::SerializeRecordFramePreamble(*where);

   if (auto res = RNTupleSerializer::SerializeExtraTypeInfoId(desc.GetContentId(), *where)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   pos += RNTupleSerializer::SerializeUInt32(desc.GetTypeVersion(), *where);
   pos += RNTupleSerializer::SerializeString(desc.GetTypeName(), *where);
   pos += RNTupleSerializer::SerializeString(desc.GetContent(), *where);

   auto size = pos - base;
   RNTupleSerializer::SerializeFramePostscript(base, size);

   return size;
}

ROOT::RResult<std::uint32_t> SerializeExtraTypeInfoList(const ROOT::RNTupleDescriptor &ntplDesc, void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   for (const auto &extraTypeInfoDesc : ntplDesc.GetExtraTypeInfoIterable()) {
      if (auto res = SerializeExtraTypeInfo(extraTypeInfoDesc, *where)) {
         pos += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }
   }

   return pos - base;
}

ROOT::RResult<std::uint32_t> DeserializeExtraTypeInfo(const void *buffer, std::uint64_t bufSize,
                                                      ROOT::Internal::RExtraTypeInfoDescriptorBuilder &desc)
{
   using ROOT::EExtraTypeInfoIds;

   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint64_t frameSize;
   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - base); };
   auto result = RNTupleSerializer::DeserializeFrameHeader(bytes, bufSize, frameSize);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   EExtraTypeInfoIds contentId{EExtraTypeInfoIds::kInvalid};
   std::uint32_t typeVersion;
   if (fnFrameSizeLeft() < 2 * sizeof(std::uint32_t)) {
      return R__FAIL("extra type info record frame too short");
   }
   result = RNTupleSerializer::DeserializeExtraTypeInfoId(bytes, contentId);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, typeVersion);

   std::string typeName;
   std::string content;
   result = RNTupleSerializer::DeserializeString(bytes, fnFrameSizeLeft(), typeName).Unwrap();
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   result = RNTupleSerializer::DeserializeString(bytes, fnFrameSizeLeft(), content).Unwrap();
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   desc.ContentId(contentId).TypeVersion(typeVersion).TypeName(typeName).Content(content);

   return frameSize;
}

std::uint32_t SerializeLocatorPayloadLarge(const ROOT::RNTupleLocator &locator, unsigned char *buffer)
{
   if (buffer) {
      RNTupleSerializer::SerializeUInt64(locator.GetNBytesOnStorage(), buffer);
      RNTupleSerializer::SerializeUInt64(locator.GetPosition<std::uint64_t>(), buffer + sizeof(std::uint64_t));
   }
   return sizeof(std::uint64_t) + sizeof(std::uint64_t);
}

void DeserializeLocatorPayloadLarge(const unsigned char *buffer, ROOT::RNTupleLocator &locator)
{
   std::uint64_t nBytesOnStorage;
   std::uint64_t position;
   RNTupleSerializer::DeserializeUInt64(buffer, nBytesOnStorage);
   RNTupleSerializer::DeserializeUInt64(buffer + sizeof(std::uint64_t), position);
   locator.SetNBytesOnStorage(nBytesOnStorage);
   locator.SetPosition(position);
}

std::uint32_t SerializeLocatorPayloadObject64(const ROOT::RNTupleLocator &locator, unsigned char *buffer)
{
   const auto &data = locator.GetPosition<ROOT::RNTupleLocatorObject64>();
   const uint32_t sizeofNBytesOnStorage = (locator.GetNBytesOnStorage() > std::numeric_limits<std::uint32_t>::max())
                                             ? sizeof(std::uint64_t)
                                             : sizeof(std::uint32_t);
   if (buffer) {
      if (sizeofNBytesOnStorage == sizeof(std::uint32_t)) {
         RNTupleSerializer::SerializeUInt32(locator.GetNBytesOnStorage(), buffer);
      } else {
         RNTupleSerializer::SerializeUInt64(locator.GetNBytesOnStorage(), buffer);
      }
      RNTupleSerializer::SerializeUInt64(data.GetLocation(), buffer + sizeofNBytesOnStorage);
   }
   return sizeofNBytesOnStorage + sizeof(std::uint64_t);
}

ROOT::RResult<void> DeserializeLocatorPayloadObject64(const unsigned char *buffer, std::uint32_t sizeofLocatorPayload,
                                                      ROOT::RNTupleLocator &locator)
{
   std::uint64_t location;
   if (sizeofLocatorPayload == 12) {
      std::uint32_t nBytesOnStorage;
      RNTupleSerializer::DeserializeUInt32(buffer, nBytesOnStorage);
      locator.SetNBytesOnStorage(nBytesOnStorage);
      RNTupleSerializer::DeserializeUInt64(buffer + sizeof(std::uint32_t), location);
   } else if (sizeofLocatorPayload == 16) {
      std::uint64_t nBytesOnStorage;
      RNTupleSerializer::DeserializeUInt64(buffer, nBytesOnStorage);
      locator.SetNBytesOnStorage(nBytesOnStorage);
      RNTupleSerializer::DeserializeUInt64(buffer + sizeof(std::uint64_t), location);
   } else {
      return R__FAIL("invalid DAOS locator payload size: " + std::to_string(sizeofLocatorPayload));
   }
   locator.SetPosition(ROOT::RNTupleLocatorObject64{location});
   return ROOT::RResult<void>::Success();
}

std::uint32_t SerializeAliasColumn(const ROOT::RColumnDescriptor &columnDesc,
                                   const ROOT::Internal::RNTupleSerializer::RContext &context, void *buffer)
{
   R__ASSERT(columnDesc.IsAliasColumn());

   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   pos += RNTupleSerializer::SerializeRecordFramePreamble(*where);

   pos += RNTupleSerializer::SerializeUInt32(context.GetOnDiskColumnId(columnDesc.GetPhysicalId()), *where);
   pos += RNTupleSerializer::SerializeUInt32(context.GetOnDiskFieldId(columnDesc.GetFieldId()), *where);

   pos += RNTupleSerializer::SerializeFramePostscript(buffer ? base : nullptr, pos - base).Unwrap();

   return pos - base;
}

std::uint32_t SerializeAliasColumnsOfFields(const ROOT::RNTupleDescriptor &desc,
                                            std::span<const ROOT::DescriptorId_t> fieldList,
                                            const ROOT::Internal::RNTupleSerializer::RContext &context, void *buffer,
                                            bool forHeaderExtension)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   const auto *xHeader = !forHeaderExtension ? desc.GetHeaderExtension() : nullptr;

   for (auto parentId : fieldList) {
      if (xHeader && xHeader->ContainsField(parentId))
         continue;

      for (const auto &c : desc.GetColumnIterable(parentId)) {
         if (!c.IsAliasColumn() || (xHeader && xHeader->ContainsExtendedColumnRepresentation(c.GetLogicalId())))
            continue;

         pos += SerializeAliasColumn(c, context, *where);
      }
   }

   return pos - base;
}

ROOT::RResult<std::uint32_t> DeserializeAliasColumn(const void *buffer, std::uint64_t bufSize,
                                                    std::uint32_t &physicalColumnId, std::uint32_t &fieldId)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint64_t frameSize;
   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - base); };
   auto result = RNTupleSerializer::DeserializeFrameHeader(bytes, bufSize, frameSize);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   if (fnFrameSizeLeft() < 2 * sizeof(std::uint32_t)) {
      return R__FAIL("alias column record frame too short");
   }

   bytes += RNTupleSerializer::DeserializeUInt32(bytes, physicalColumnId);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fieldId);

   return frameSize;
}

} // anonymous namespace

std::uint32_t ROOT::Internal::RNTupleSerializer::SerializeXxHash3(const unsigned char *data, std::uint64_t length,
                                                                  std::uint64_t &xxhash3, void *buffer)
{
   if (buffer != nullptr) {
      xxhash3 = XXH3_64bits(data, length);
      SerializeUInt64(xxhash3, buffer);
   }
   return 8;
}

ROOT::RResult<void> ROOT::Internal::RNTupleSerializer::VerifyXxHash3(const unsigned char *data, std::uint64_t length,
                                                                     std::uint64_t &xxhash3)
{
   auto checksumReal = XXH3_64bits(data, length);
   DeserializeUInt64(data + length, xxhash3);
   if (xxhash3 != checksumReal)
      return R__FAIL("XxHash-3 checksum mismatch");
   return RResult<void>::Success();
}

ROOT::RResult<void> ROOT::Internal::RNTupleSerializer::VerifyXxHash3(const unsigned char *data, std::uint64_t length)
{
   std::uint64_t xxhash3;
   return R__FORWARD_RESULT(VerifyXxHash3(data, length, xxhash3));
}

std::uint32_t ROOT::Internal::RNTupleSerializer::SerializeInt16(std::int16_t val, void *buffer)
{
   if (buffer != nullptr) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);
      bytes[0] = (val & 0x00FF);
      bytes[1] = (val & 0xFF00) >> 8;
   }
   return 2;
}

std::uint32_t ROOT::Internal::RNTupleSerializer::DeserializeInt16(const void *buffer, std::int16_t &val)
{
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   val = std::int16_t(bytes[0]) + (std::int16_t(bytes[1]) << 8);
   return 2;
}

std::uint32_t ROOT::Internal::RNTupleSerializer::SerializeUInt16(std::uint16_t val, void *buffer)
{
   return SerializeInt16(val, buffer);
}

std::uint32_t ROOT::Internal::RNTupleSerializer::DeserializeUInt16(const void *buffer, std::uint16_t &val)
{
   return DeserializeInt16(buffer, *reinterpret_cast<std::int16_t *>(&val));
}

std::uint32_t ROOT::Internal::RNTupleSerializer::SerializeInt32(std::int32_t val, void *buffer)
{
   if (buffer != nullptr) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);
      bytes[0] = (val & 0x000000FF);
      bytes[1] = (val & 0x0000FF00) >> 8;
      bytes[2] = (val & 0x00FF0000) >> 16;
      bytes[3] = (val & 0xFF000000) >> 24;
   }
   return 4;
}

std::uint32_t ROOT::Internal::RNTupleSerializer::DeserializeInt32(const void *buffer, std::int32_t &val)
{
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   val = std::int32_t(bytes[0]) + (std::int32_t(bytes[1]) << 8) + (std::int32_t(bytes[2]) << 16) +
         (std::int32_t(bytes[3]) << 24);
   return 4;
}

std::uint32_t ROOT::Internal::RNTupleSerializer::SerializeUInt32(std::uint32_t val, void *buffer)
{
   return SerializeInt32(val, buffer);
}

std::uint32_t ROOT::Internal::RNTupleSerializer::DeserializeUInt32(const void *buffer, std::uint32_t &val)
{
   return DeserializeInt32(buffer, *reinterpret_cast<std::int32_t *>(&val));
}

std::uint32_t ROOT::Internal::RNTupleSerializer::SerializeInt64(std::int64_t val, void *buffer)
{
   if (buffer != nullptr) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);
      bytes[0] = (val & 0x00000000000000FF);
      bytes[1] = (val & 0x000000000000FF00) >> 8;
      bytes[2] = (val & 0x0000000000FF0000) >> 16;
      bytes[3] = (val & 0x00000000FF000000) >> 24;
      bytes[4] = (val & 0x000000FF00000000) >> 32;
      bytes[5] = (val & 0x0000FF0000000000) >> 40;
      bytes[6] = (val & 0x00FF000000000000) >> 48;
      bytes[7] = (val & 0xFF00000000000000) >> 56;
   }
   return 8;
}

std::uint32_t ROOT::Internal::RNTupleSerializer::DeserializeInt64(const void *buffer, std::int64_t &val)
{
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   val = std::int64_t(bytes[0]) + (std::int64_t(bytes[1]) << 8) + (std::int64_t(bytes[2]) << 16) +
         (std::int64_t(bytes[3]) << 24) + (std::int64_t(bytes[4]) << 32) + (std::int64_t(bytes[5]) << 40) +
         (std::int64_t(bytes[6]) << 48) + (std::int64_t(bytes[7]) << 56);
   return 8;
}

std::uint32_t ROOT::Internal::RNTupleSerializer::SerializeUInt64(std::uint64_t val, void *buffer)
{
   return SerializeInt64(val, buffer);
}

std::uint32_t ROOT::Internal::RNTupleSerializer::DeserializeUInt64(const void *buffer, std::uint64_t &val)
{
   return DeserializeInt64(buffer, *reinterpret_cast<std::int64_t *>(&val));
}

std::uint32_t ROOT::Internal::RNTupleSerializer::SerializeString(const std::string &val, void *buffer)
{
   if (buffer) {
      auto pos = reinterpret_cast<unsigned char *>(buffer);
      pos += SerializeUInt32(val.length(), pos);
      memcpy(pos, val.data(), val.length());
   }
   return sizeof(std::uint32_t) + val.length();
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::DeserializeString(const void *buffer, std::uint64_t bufSize, std::string &val)
{
   if (bufSize < sizeof(std::uint32_t))
      return R__FAIL("string buffer too short");
   bufSize -= sizeof(std::uint32_t);

   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint32_t length;
   bytes += DeserializeUInt32(buffer, length);
   if (bufSize < length)
      return R__FAIL("string buffer too short");

   val.resize(length);
   memcpy(&val[0], bytes, length);
   return sizeof(std::uint32_t) + length;
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::SerializeColumnType(ENTupleColumnType type, void *buffer)
{
   switch (type) {
   case ENTupleColumnType::kBit: return SerializeUInt16(0x00, buffer);
   case ENTupleColumnType::kByte: return SerializeUInt16(0x01, buffer);
   case ENTupleColumnType::kChar: return SerializeUInt16(0x02, buffer);
   case ENTupleColumnType::kInt8: return SerializeUInt16(0x03, buffer);
   case ENTupleColumnType::kUInt8: return SerializeUInt16(0x04, buffer);
   case ENTupleColumnType::kInt16: return SerializeUInt16(0x05, buffer);
   case ENTupleColumnType::kUInt16: return SerializeUInt16(0x06, buffer);
   case ENTupleColumnType::kInt32: return SerializeUInt16(0x07, buffer);
   case ENTupleColumnType::kUInt32: return SerializeUInt16(0x08, buffer);
   case ENTupleColumnType::kInt64: return SerializeUInt16(0x09, buffer);
   case ENTupleColumnType::kUInt64: return SerializeUInt16(0x0A, buffer);
   case ENTupleColumnType::kReal16: return SerializeUInt16(0x0B, buffer);
   case ENTupleColumnType::kReal32: return SerializeUInt16(0x0C, buffer);
   case ENTupleColumnType::kReal64: return SerializeUInt16(0x0D, buffer);
   case ENTupleColumnType::kIndex32: return SerializeUInt16(0x0E, buffer);
   case ENTupleColumnType::kIndex64: return SerializeUInt16(0x0F, buffer);
   case ENTupleColumnType::kSwitch: return SerializeUInt16(0x10, buffer);
   case ENTupleColumnType::kSplitInt16: return SerializeUInt16(0x11, buffer);
   case ENTupleColumnType::kSplitUInt16: return SerializeUInt16(0x12, buffer);
   case ENTupleColumnType::kSplitInt32: return SerializeUInt16(0x13, buffer);
   case ENTupleColumnType::kSplitUInt32: return SerializeUInt16(0x14, buffer);
   case ENTupleColumnType::kSplitInt64: return SerializeUInt16(0x15, buffer);
   case ENTupleColumnType::kSplitUInt64: return SerializeUInt16(0x16, buffer);
   case ENTupleColumnType::kSplitReal32: return SerializeUInt16(0x18, buffer);
   case ENTupleColumnType::kSplitReal64: return SerializeUInt16(0x19, buffer);
   case ENTupleColumnType::kSplitIndex32: return SerializeUInt16(0x1A, buffer);
   case ENTupleColumnType::kSplitIndex64: return SerializeUInt16(0x1B, buffer);
   case ENTupleColumnType::kReal32Trunc: return SerializeUInt16(0x1C, buffer);
   case ENTupleColumnType::kReal32Quant: return SerializeUInt16(0x1D, buffer);
   default:
      if (type == ROOT::Internal::kTestFutureColumnType)
         return SerializeUInt16(0x99, buffer);
      return R__FAIL("unexpected column type");
   }
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::DeserializeColumnType(const void *buffer, ENTupleColumnType &type)
{
   std::uint16_t onDiskType;
   auto result = DeserializeUInt16(buffer, onDiskType);

   switch (onDiskType) {
   case 0x00: type = ENTupleColumnType::kBit; break;
   case 0x01: type = ENTupleColumnType::kByte; break;
   case 0x02: type = ENTupleColumnType::kChar; break;
   case 0x03: type = ENTupleColumnType::kInt8; break;
   case 0x04: type = ENTupleColumnType::kUInt8; break;
   case 0x05: type = ENTupleColumnType::kInt16; break;
   case 0x06: type = ENTupleColumnType::kUInt16; break;
   case 0x07: type = ENTupleColumnType::kInt32; break;
   case 0x08: type = ENTupleColumnType::kUInt32; break;
   case 0x09: type = ENTupleColumnType::kInt64; break;
   case 0x0A: type = ENTupleColumnType::kUInt64; break;
   case 0x0B: type = ENTupleColumnType::kReal16; break;
   case 0x0C: type = ENTupleColumnType::kReal32; break;
   case 0x0D: type = ENTupleColumnType::kReal64; break;
   case 0x0E: type = ENTupleColumnType::kIndex32; break;
   case 0x0F: type = ENTupleColumnType::kIndex64; break;
   case 0x10: type = ENTupleColumnType::kSwitch; break;
   case 0x11: type = ENTupleColumnType::kSplitInt16; break;
   case 0x12: type = ENTupleColumnType::kSplitUInt16; break;
   case 0x13: type = ENTupleColumnType::kSplitInt32; break;
   case 0x14: type = ENTupleColumnType::kSplitUInt32; break;
   case 0x15: type = ENTupleColumnType::kSplitInt64; break;
   case 0x16: type = ENTupleColumnType::kSplitUInt64; break;
   case 0x18: type = ENTupleColumnType::kSplitReal32; break;
   case 0x19: type = ENTupleColumnType::kSplitReal64; break;
   case 0x1A: type = ENTupleColumnType::kSplitIndex32; break;
   case 0x1B: type = ENTupleColumnType::kSplitIndex64; break;
   case 0x1C: type = ENTupleColumnType::kReal32Trunc; break;
   case 0x1D: type = ENTupleColumnType::kReal32Quant; break;
   // case 0x99 => kTestFutureColumnType missing on purpose
   default:
      // may be a column type introduced by a future version
      type = ENTupleColumnType::kUnknown;
      break;
   }
   return result;
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::SerializeFieldStructure(ROOT::ENTupleStructure structure, void *buffer)
{
   using ENTupleStructure = ROOT::ENTupleStructure;
   switch (structure) {
   case ENTupleStructure::kLeaf: return SerializeUInt16(0x00, buffer);
   case ENTupleStructure::kCollection: return SerializeUInt16(0x01, buffer);
   case ENTupleStructure::kRecord: return SerializeUInt16(0x02, buffer);
   case ENTupleStructure::kVariant: return SerializeUInt16(0x03, buffer);
   case ENTupleStructure::kStreamer: return SerializeUInt16(0x04, buffer);
   default:
      if (structure == ROOT::Internal::kTestFutureFieldStructure)
         return SerializeUInt16(0x99, buffer);
      return R__FAIL("unexpected field structure type");
   }
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::DeserializeFieldStructure(const void *buffer, ROOT::ENTupleStructure &structure)
{
   using ENTupleStructure = ROOT::ENTupleStructure;
   std::uint16_t onDiskValue;
   auto result = DeserializeUInt16(buffer, onDiskValue);
   switch (onDiskValue) {
   case 0x00: structure = ENTupleStructure::kLeaf; break;
   case 0x01: structure = ENTupleStructure::kCollection; break;
   case 0x02: structure = ENTupleStructure::kRecord; break;
   case 0x03: structure = ENTupleStructure::kVariant; break;
   case 0x04: structure = ENTupleStructure::kStreamer; break;
   // case 0x99 => kTestFutureFieldStructure intentionally missing
   default: structure = ENTupleStructure::kUnknown;
   }
   return result;
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::SerializeExtraTypeInfoId(ROOT::EExtraTypeInfoIds id, void *buffer)
{
   switch (id) {
   case ROOT::EExtraTypeInfoIds::kStreamerInfo: return SerializeUInt32(0x00, buffer);
   default: return R__FAIL("unexpected extra type info id");
   }
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::DeserializeExtraTypeInfoId(const void *buffer, ROOT::EExtraTypeInfoIds &id)
{
   std::uint32_t onDiskValue;
   auto result = DeserializeUInt32(buffer, onDiskValue);
   switch (onDiskValue) {
   case 0x00: id = ROOT::EExtraTypeInfoIds::kStreamerInfo; break;
   default:
      id = ROOT::EExtraTypeInfoIds::kInvalid;
      R__LOG_DEBUG(0, ROOT::Internal::NTupleLog()) << "Unknown extra type info id: " << onDiskValue;
   }
   return result;
}

std::uint32_t ROOT::Internal::RNTupleSerializer::SerializeEnvelopePreamble(std::uint16_t envelopeType, void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   pos += SerializeUInt64(envelopeType, *where);
   // The 48bits size information is filled in the postscript
   return pos - base;
}

ROOT::RResult<std::uint32_t> ROOT::Internal::RNTupleSerializer::SerializeEnvelopePostscript(unsigned char *envelope,
                                                                                            std::uint64_t size,
                                                                                            std::uint64_t &xxhash3)
{
   if (size < sizeof(std::uint64_t))
      return R__FAIL("envelope size too small");
   if (size >= static_cast<uint64_t>(1) << 48)
      return R__FAIL("envelope size too big");
   if (envelope) {
      std::uint64_t typeAndSize;
      DeserializeUInt64(envelope, typeAndSize);
      typeAndSize |= (size + 8) << 16;
      SerializeUInt64(typeAndSize, envelope);
   }
   return SerializeXxHash3(envelope, size, xxhash3, envelope ? (envelope + size) : nullptr);
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::SerializeEnvelopePostscript(unsigned char *envelope, std::uint64_t size)
{
   std::uint64_t xxhash3;
   return R__FORWARD_RESULT(SerializeEnvelopePostscript(envelope, size, xxhash3));
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::DeserializeEnvelope(const void *buffer, std::uint64_t bufSize,
                                                       std::uint16_t expectedType, std::uint64_t &xxhash3)
{
   const std::uint64_t minEnvelopeSize = sizeof(std::uint64_t) + sizeof(std::uint64_t);
   if (bufSize < minEnvelopeSize)
      return R__FAIL("invalid envelope buffer, too short");

   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   auto base = bytes;

   std::uint64_t typeAndSize;
   bytes += DeserializeUInt64(bytes, typeAndSize);

   std::uint16_t envelopeType = typeAndSize & 0xFFFF;
   if (envelopeType != expectedType) {
      return R__FAIL("envelope type mismatch: expected " + std::to_string(expectedType) + ", found " +
                     std::to_string(envelopeType));
   }

   std::uint64_t envelopeSize = typeAndSize >> 16;
   if (bufSize < envelopeSize)
      return R__FAIL("envelope buffer size too small");
   if (envelopeSize < minEnvelopeSize)
      return R__FAIL("invalid envelope, too short");

   auto result = VerifyXxHash3(base, envelopeSize - 8, xxhash3);
   if (!result)
      return R__FORWARD_ERROR(result);

   return sizeof(typeAndSize);
}

ROOT::RResult<std::uint32_t> ROOT::Internal::RNTupleSerializer::DeserializeEnvelope(const void *buffer,
                                                                                    std::uint64_t bufSize,
                                                                                    std::uint16_t expectedType)
{
   std::uint64_t xxhash3;
   return R__FORWARD_RESULT(DeserializeEnvelope(buffer, bufSize, expectedType, xxhash3));
}

std::uint32_t ROOT::Internal::RNTupleSerializer::SerializeRecordFramePreamble(void *buffer)
{
   // Marker: multiply the final size with 1
   return SerializeInt64(1, buffer);
}

std::uint32_t ROOT::Internal::RNTupleSerializer::SerializeListFramePreamble(std::uint32_t nitems, void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   // Marker: multiply the final size with -1
   pos += SerializeInt64(-1, *where);
   pos += SerializeUInt32(nitems, *where);
   return pos - base;
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::SerializeFramePostscript(void *frame, std::uint64_t size)
{
   auto preambleSize = sizeof(std::int64_t);
   if (size < preambleSize)
      return R__FAIL("frame too short: " + std::to_string(size));
   if (frame) {
      std::int64_t marker;
      DeserializeInt64(frame, marker);
      if ((marker < 0) && (size < (sizeof(std::uint32_t) + preambleSize)))
         return R__FAIL("frame too short: " + std::to_string(size));
      SerializeInt64(marker * static_cast<int64_t>(size), frame);
   }
   return 0;
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::DeserializeFrameHeader(const void *buffer, std::uint64_t bufSize,
                                                          std::uint64_t &frameSize, std::uint32_t &nitems)
{
   std::uint64_t minSize = sizeof(std::int64_t);
   if (bufSize < minSize)
      return R__FAIL("frame too short");

   std::int64_t *ssize = reinterpret_cast<std::int64_t *>(&frameSize);
   DeserializeInt64(buffer, *ssize);

   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   bytes += minSize;

   if (*ssize >= 0) {
      // Record frame
      nitems = 1;
   } else {
      // List frame
      minSize += sizeof(std::uint32_t);
      if (bufSize < minSize)
         return R__FAIL("frame too short");
      bytes += DeserializeUInt32(bytes, nitems);
      *ssize = -(*ssize);
   }

   if (frameSize < minSize)
      return R__FAIL("corrupt frame size");
   if (bufSize < frameSize)
      return R__FAIL("frame too short");

   return bytes - reinterpret_cast<const unsigned char *>(buffer);
}

ROOT::RResult<std::uint32_t> ROOT::Internal::RNTupleSerializer::DeserializeFrameHeader(const void *buffer,
                                                                                       std::uint64_t bufSize,
                                                                                       std::uint64_t &frameSize)
{
   std::uint32_t nitems;
   return R__FORWARD_RESULT(DeserializeFrameHeader(buffer, bufSize, frameSize, nitems));
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::SerializeFeatureFlags(const std::vector<std::uint64_t> &flags, void *buffer)
{
   if (flags.empty())
      return SerializeUInt64(0, buffer);

   if (buffer) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);

      for (unsigned i = 0; i < flags.size(); ++i) {
         if (flags[i] & 0x8000000000000000)
            return R__FAIL("feature flag out of bounds");

         // The MSb indicates that another Int64 follows; set this bit to 1 for all except the last element
         if (i == (flags.size() - 1))
            SerializeUInt64(flags[i], bytes);
         else
            bytes += SerializeUInt64(flags[i] | 0x8000000000000000, bytes);
      }
   }
   return (flags.size() * sizeof(std::int64_t));
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::DeserializeFeatureFlags(const void *buffer, std::uint64_t bufSize,
                                                           std::vector<std::uint64_t> &flags)
{
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);

   flags.clear();
   std::uint64_t f;
   do {
      if (bufSize < sizeof(std::uint64_t))
         return R__FAIL("feature flag buffer too short");
      bytes += DeserializeUInt64(bytes, f);
      bufSize -= sizeof(std::uint64_t);
      flags.emplace_back(f & ~0x8000000000000000);
   } while (f & 0x8000000000000000);

   return (flags.size() * sizeof(std::uint64_t));
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::SerializeLocator(const RNTupleLocator &locator, void *buffer)
{
   if (locator.GetType() > RNTupleLocator::kLastSerializableType)
      return R__FAIL("locator is not serializable");

   std::uint32_t size = 0;
   if ((locator.GetType() == RNTupleLocator::kTypeFile) &&
       (locator.GetNBytesOnStorage() <= std::numeric_limits<std::int32_t>::max())) {
      size += SerializeUInt32(locator.GetNBytesOnStorage(), buffer);
      size += SerializeUInt64(locator.GetPosition<std::uint64_t>(),
                              buffer ? reinterpret_cast<unsigned char *>(buffer) + size : nullptr);
      return size;
   }

   std::uint8_t locatorType = 0;
   auto payloadp = buffer ? reinterpret_cast<unsigned char *>(buffer) + sizeof(std::int32_t) : nullptr;
   switch (locator.GetType()) {
   case RNTupleLocator::kTypeFile:
      size += SerializeLocatorPayloadLarge(locator, payloadp);
      locatorType = 0x01;
      break;
   case RNTupleLocator::kTypeDAOS:
      size += SerializeLocatorPayloadObject64(locator, payloadp);
      locatorType = 0x02;
      break;
   default:
      if (locator.GetType() == ROOT::Internal::kTestLocatorType) {
         // For the testing locator, use the same payload as Object64. We're not gonna really read it back anyway.
         size += SerializeLocatorPayloadObject64(locator, payloadp);
         locatorType = 0x7e;
      } else {
         return R__FAIL("locator has unknown type");
      }
   }
   std::int32_t head = sizeof(std::int32_t) + size;
   head |= locator.GetReserved() << 16;
   head |= static_cast<int>(locatorType & 0x7F) << 24;
   head = -head;
   size += RNTupleSerializer::SerializeInt32(head, buffer);
   return size;
}

ROOT::RResult<std::uint32_t> ROOT::Internal::RNTupleSerializer::DeserializeLocator(const void *buffer,
                                                                                   std::uint64_t bufSize,
                                                                                   RNTupleLocator &locator)
{
   if (bufSize < sizeof(std::int32_t))
      return R__FAIL("too short locator");

   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   std::int32_t head;

   bytes += DeserializeInt32(bytes, head);
   bufSize -= sizeof(std::int32_t);
   if (head < 0) {
      head = -head;
      const int type = head >> 24;
      const std::uint32_t payloadSize = (static_cast<std::uint32_t>(head) & 0x0000FFFF) - sizeof(std::int32_t);
      if (bufSize < payloadSize)
         return R__FAIL("too short locator");

      locator.SetReserved(static_cast<std::uint32_t>(head >> 16) & 0xFF);
      switch (type) {
      case 0x01:
         locator.SetType(RNTupleLocator::kTypeFile);
         DeserializeLocatorPayloadLarge(bytes, locator);
         break;
      case 0x02:
         locator.SetType(RNTupleLocator::kTypeDAOS);
         DeserializeLocatorPayloadObject64(bytes, payloadSize, locator);
         break;
      default: locator.SetType(RNTupleLocator::kTypeUnknown);
      }
      bytes += payloadSize;
   } else {
      if (bufSize < sizeof(std::uint64_t))
         return R__FAIL("too short locator");
      std::uint64_t offset;
      bytes += DeserializeUInt64(bytes, offset);
      locator.SetType(RNTupleLocator::kTypeFile);
      locator.SetNBytesOnStorage(head);
      locator.SetPosition(offset);
   }

   return bytes - reinterpret_cast<const unsigned char *>(buffer);
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::SerializeEnvelopeLink(const REnvelopeLink &envelopeLink, void *buffer)
{
   auto size = SerializeUInt64(envelopeLink.fLength, buffer);
   auto res =
      SerializeLocator(envelopeLink.fLocator, buffer ? reinterpret_cast<unsigned char *>(buffer) + size : nullptr);
   if (res)
      size += res.Unwrap();
   else
      return R__FORWARD_ERROR(res);
   return size;
}

ROOT::RResult<std::uint32_t> ROOT::Internal::RNTupleSerializer::DeserializeEnvelopeLink(const void *buffer,
                                                                                        std::uint64_t bufSize,
                                                                                        REnvelopeLink &envelopeLink)
{
   if (bufSize < sizeof(std::int64_t))
      return R__FAIL("too short envelope link");

   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   bytes += DeserializeUInt64(bytes, envelopeLink.fLength);
   bufSize -= sizeof(std::uint64_t);
   if (auto res = DeserializeLocator(bytes, bufSize, envelopeLink.fLocator)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   return bytes - reinterpret_cast<const unsigned char *>(buffer);
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::SerializeClusterSummary(const RClusterSummary &clusterSummary, void *buffer)
{
   if (clusterSummary.fNEntries >= (static_cast<std::uint64_t>(1) << 56)) {
      return R__FAIL("number of entries in cluster exceeds maximum of 2^56");
   }

   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   auto frame = pos;
   pos += SerializeRecordFramePreamble(*where);
   pos += SerializeUInt64(clusterSummary.fFirstEntry, *where);
   const std::uint64_t nEntriesAndFlags =
      (static_cast<std::uint64_t>(clusterSummary.fFlags) << 56) | clusterSummary.fNEntries;
   pos += SerializeUInt64(nEntriesAndFlags, *where);

   auto size = pos - frame;
   if (auto res = SerializeFramePostscript(frame, size)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   return size;
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::DeserializeClusterSummary(const void *buffer, std::uint64_t bufSize,
                                                             RClusterSummary &clusterSummary)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint64_t frameSize;
   if (auto res = DeserializeFrameHeader(bytes, bufSize, frameSize)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - base); };
   if (fnFrameSizeLeft() < 2 * sizeof(std::uint64_t))
      return R__FAIL("too short cluster summary");

   bytes += DeserializeUInt64(bytes, clusterSummary.fFirstEntry);
   std::uint64_t nEntriesAndFlags;
   bytes += DeserializeUInt64(bytes, nEntriesAndFlags);

   const std::uint64_t nEntries = (nEntriesAndFlags << 8) >> 8;
   const std::uint8_t flags = nEntriesAndFlags >> 56;

   if (flags & 0x01) {
      return R__FAIL("sharded cluster flag set in cluster summary; sharded clusters are currently unsupported.");
   }

   clusterSummary.fNEntries = nEntries;
   clusterSummary.fFlags = flags;

   return frameSize;
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::SerializeClusterGroup(const RClusterGroup &clusterGroup, void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   auto frame = pos;
   pos += SerializeRecordFramePreamble(*where);
   pos += SerializeUInt64(clusterGroup.fMinEntry, *where);
   pos += SerializeUInt64(clusterGroup.fEntrySpan, *where);
   pos += SerializeUInt32(clusterGroup.fNClusters, *where);
   if (auto res = SerializeEnvelopeLink(clusterGroup.fPageListEnvelopeLink, *where)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   auto size = pos - frame;
   if (auto res = SerializeFramePostscript(frame, size)) {
      return size;
   } else {
      return R__FORWARD_ERROR(res);
   }
}

ROOT::RResult<std::uint32_t> ROOT::Internal::RNTupleSerializer::DeserializeClusterGroup(const void *buffer,
                                                                                        std::uint64_t bufSize,
                                                                                        RClusterGroup &clusterGroup)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;

   std::uint64_t frameSize;
   if (auto res = DeserializeFrameHeader(bytes, bufSize, frameSize)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - base); };
   if (fnFrameSizeLeft() < sizeof(std::uint32_t) + 2 * sizeof(std::uint64_t))
      return R__FAIL("too short cluster group");

   bytes += DeserializeUInt64(bytes, clusterGroup.fMinEntry);
   bytes += DeserializeUInt64(bytes, clusterGroup.fEntrySpan);
   bytes += DeserializeUInt32(bytes, clusterGroup.fNClusters);
   if (auto res = DeserializeEnvelopeLink(bytes, fnFrameSizeLeft(), clusterGroup.fPageListEnvelopeLink)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   return frameSize;
}

void ROOT::Internal::RNTupleSerializer::RContext::MapSchema(const ROOT::RNTupleDescriptor &desc,
                                                            bool forHeaderExtension)
{
   auto fieldZeroId = desc.GetFieldZeroId();
   auto depthFirstTraversal = [&](std::span<ROOT::DescriptorId_t> fieldTrees, auto doForEachField) {
      std::deque<ROOT::DescriptorId_t> idQueue{fieldTrees.begin(), fieldTrees.end()};
      while (!idQueue.empty()) {
         auto fieldId = idQueue.front();
         idQueue.pop_front();
         // Field zero has no physical representation nor columns of its own; recurse over its subfields only
         if (fieldId != fieldZeroId)
            doForEachField(fieldId);
         unsigned i = 0;
         for (const auto &f : desc.GetFieldIterable(fieldId))
            idQueue.insert(idQueue.begin() + i++, f.GetId());
      }
   };

   R__ASSERT(desc.GetNFields() > 0); // we must have at least a zero field

   std::vector<ROOT::DescriptorId_t> fieldTrees;
   if (!forHeaderExtension) {
      fieldTrees.emplace_back(fieldZeroId);
   } else if (auto xHeader = desc.GetHeaderExtension()) {
      fieldTrees = xHeader->GetTopLevelFields(desc);
   }
   depthFirstTraversal(fieldTrees, [&](ROOT::DescriptorId_t fieldId) { MapFieldId(fieldId); });
   depthFirstTraversal(fieldTrees, [&](ROOT::DescriptorId_t fieldId) {
      for (const auto &c : desc.GetColumnIterable(fieldId)) {
         if (!c.IsAliasColumn()) {
            MapPhysicalColumnId(c.GetPhysicalId());
         }
      }
   });

   if (forHeaderExtension) {
      // Create physical IDs for column representations that extend fields of the regular header.
      // First the physical columns then the alias columns.
      for (auto memId : desc.GetHeaderExtension()->GetExtendedColumnRepresentations()) {
         const auto &columnDesc = desc.GetColumnDescriptor(memId);
         if (!columnDesc.IsAliasColumn()) {
            MapPhysicalColumnId(columnDesc.GetPhysicalId());
         }
      }
   }
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::SerializeSchemaDescription(void *buffer, const ROOT::RNTupleDescriptor &desc,
                                                              const RContext &context, bool forHeaderExtension)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   std::size_t nFields = 0, nColumns = 0, nAliasColumns = 0, fieldListOffset = 0;
   // Columns in the extension header that are attached to a field of the regular header
   std::vector<std::reference_wrapper<const ROOT::RColumnDescriptor>> extraColumns;
   if (forHeaderExtension) {
      // A call to `RNTupleDescriptorBuilder::BeginHeaderExtension()` is not strictly required after serializing the
      // header, which may happen, e.g., in unit tests.  Ensure an empty schema extension is serialized in this case
      if (auto xHeader = desc.GetHeaderExtension()) {
         nFields = xHeader->GetNFields();
         nColumns = xHeader->GetNPhysicalColumns();
         nAliasColumns = xHeader->GetNLogicalColumns() - xHeader->GetNPhysicalColumns();
         fieldListOffset = desc.GetNFields() - nFields - 1;

         extraColumns.reserve(xHeader->GetExtendedColumnRepresentations().size());
         for (auto columnId : xHeader->GetExtendedColumnRepresentations()) {
            extraColumns.emplace_back(desc.GetColumnDescriptor(columnId));
         }
      }
   } else {
      if (auto xHeader = desc.GetHeaderExtension()) {
         nFields = desc.GetNFields() - xHeader->GetNFields() - 1;
         nColumns = desc.GetNPhysicalColumns() - xHeader->GetNPhysicalColumns();
         nAliasColumns = desc.GetNLogicalColumns() - desc.GetNPhysicalColumns() -
                         (xHeader->GetNLogicalColumns() - xHeader->GetNPhysicalColumns());
      } else {
         nFields = desc.GetNFields() - 1;
         nColumns = desc.GetNPhysicalColumns();
         nAliasColumns = desc.GetNLogicalColumns() - desc.GetNPhysicalColumns();
      }
   }
   const auto nExtraTypeInfos = desc.GetNExtraTypeInfos();
   const auto &onDiskFields = context.GetOnDiskFieldList();
   R__ASSERT(onDiskFields.size() >= fieldListOffset);
   std::span<const ROOT::DescriptorId_t> fieldList{onDiskFields.data() + fieldListOffset, nFields};

   auto frame = pos;
   pos += SerializeListFramePreamble(nFields, *where);
   if (auto res = SerializeFieldList(desc, fieldList, /*firstOnDiskId=*/fieldListOffset, context, *where)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   if (auto res = SerializeFramePostscript(buffer ? frame : nullptr, pos - frame)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   frame = pos;
   pos += SerializeListFramePreamble(nColumns, *where);
   if (auto res = SerializeColumnsOfFields(desc, fieldList, context, *where, forHeaderExtension)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   for (const auto &c : extraColumns) {
      if (!c.get().IsAliasColumn()) {
         if (auto res = SerializePhysicalColumn(c.get(), context, *where)) {
            pos += res.Unwrap();
         } else {
            return R__FORWARD_ERROR(res);
         }
      }
   }
   if (auto res = SerializeFramePostscript(buffer ? frame : nullptr, pos - frame)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   frame = pos;
   pos += SerializeListFramePreamble(nAliasColumns, *where);
   pos += SerializeAliasColumnsOfFields(desc, fieldList, context, *where, forHeaderExtension);
   for (const auto &c : extraColumns) {
      if (c.get().IsAliasColumn()) {
         pos += SerializeAliasColumn(c.get(), context, *where);
      }
   }
   if (auto res = SerializeFramePostscript(buffer ? frame : nullptr, pos - frame)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   frame = pos;
   // We only serialize the extra type info list in the header extension.
   if (forHeaderExtension) {
      pos += SerializeListFramePreamble(nExtraTypeInfos, *where);
      if (auto res = SerializeExtraTypeInfoList(desc, *where)) {
         pos += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }
   } else {
      pos += SerializeListFramePreamble(0, *where);
   }
   if (auto res = SerializeFramePostscript(buffer ? frame : nullptr, pos - frame)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   return static_cast<std::uint32_t>(pos - base);
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::DeserializeSchemaDescription(const void *buffer, std::uint64_t bufSize,
                                                                RNTupleDescriptorBuilder &descBuilder)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   auto fnBufSizeLeft = [&]() { return bufSize - (bytes - base); };

   std::uint64_t frameSize;
   auto frame = bytes;
   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - frame); };

   std::uint32_t nFields;
   if (auto res = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize, nFields)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   // The zero field is always added before `DeserializeSchemaDescription()` is called
   const std::uint32_t fieldIdRangeBegin = descBuilder.GetDescriptor().GetNFields() - 1;
   for (unsigned i = 0; i < nFields; ++i) {
      std::uint32_t fieldId = fieldIdRangeBegin + i;
      RFieldDescriptorBuilder fieldBuilder;
      if (auto res = DeserializeField(bytes, fnFrameSizeLeft(), fieldBuilder)) {
         bytes += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }
      if (fieldId == fieldBuilder.GetParentId())
         fieldBuilder.ParentId(kZeroFieldId);
      auto fieldDesc = fieldBuilder.FieldId(fieldId).MakeDescriptor();
      if (!fieldDesc)
         return R__FORWARD_ERROR(fieldDesc);
      const auto parentId = fieldDesc.Inspect().GetParentId();
      const auto projectionSourceId = fieldDesc.Inspect().GetProjectionSourceId();
      descBuilder.AddField(fieldDesc.Unwrap());
      auto resVoid = descBuilder.AddFieldLink(parentId, fieldId);
      if (!resVoid)
         return R__FORWARD_ERROR(resVoid);
      if (projectionSourceId != ROOT::kInvalidDescriptorId) {
         resVoid = descBuilder.AddFieldProjection(projectionSourceId, fieldId);
         if (!resVoid)
            return R__FORWARD_ERROR(resVoid);
      }
   }
   bytes = frame + frameSize;

   // As columns are added in order of representation index and column index, determine the column index
   // for the currently deserialized column from the columns already added.
   auto fnNextColumnIndex = [&descBuilder](ROOT::DescriptorId_t fieldId,
                                           std::uint16_t representationIndex) -> std::uint32_t {
      const auto &existingColumns = descBuilder.GetDescriptor().GetFieldDescriptor(fieldId).GetLogicalColumnIds();
      if (existingColumns.empty())
         return 0;
      const auto &lastColumnDesc = descBuilder.GetDescriptor().GetColumnDescriptor(existingColumns.back());
      return (representationIndex == lastColumnDesc.GetRepresentationIndex()) ? (lastColumnDesc.GetIndex() + 1) : 0;
   };

   std::uint32_t nColumns;
   frame = bytes;
   if (auto res = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize, nColumns)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   if (descBuilder.GetDescriptor().GetNLogicalColumns() > descBuilder.GetDescriptor().GetNPhysicalColumns())
      descBuilder.ShiftAliasColumns(nColumns);

   const std::uint32_t columnIdRangeBegin = descBuilder.GetDescriptor().GetNPhysicalColumns();
   for (unsigned i = 0; i < nColumns; ++i) {
      std::uint32_t columnId = columnIdRangeBegin + i;
      RColumnDescriptorBuilder columnBuilder;
      if (auto res = DeserializeColumn(bytes, fnFrameSizeLeft(), columnBuilder)) {
         bytes += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }

      columnBuilder.Index(fnNextColumnIndex(columnBuilder.GetFieldId(), columnBuilder.GetRepresentationIndex()));
      columnBuilder.LogicalColumnId(columnId);
      columnBuilder.PhysicalColumnId(columnId);
      auto columnDesc = columnBuilder.MakeDescriptor();
      if (!columnDesc)
         return R__FORWARD_ERROR(columnDesc);
      auto resVoid = descBuilder.AddColumn(columnDesc.Unwrap());
      if (!resVoid)
         return R__FORWARD_ERROR(resVoid);
   }
   bytes = frame + frameSize;

   std::uint32_t nAliasColumns;
   frame = bytes;
   if (auto res = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize, nAliasColumns)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   const std::uint32_t aliasColumnIdRangeBegin = descBuilder.GetDescriptor().GetNLogicalColumns();
   for (unsigned i = 0; i < nAliasColumns; ++i) {
      std::uint32_t physicalId;
      std::uint32_t fieldId;
      if (auto res = DeserializeAliasColumn(bytes, fnFrameSizeLeft(), physicalId, fieldId)) {
         bytes += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }

      RColumnDescriptorBuilder columnBuilder;
      columnBuilder.LogicalColumnId(aliasColumnIdRangeBegin + i).PhysicalColumnId(physicalId).FieldId(fieldId);
      const auto &physicalColumnDesc = descBuilder.GetDescriptor().GetColumnDescriptor(physicalId);
      columnBuilder.BitsOnStorage(physicalColumnDesc.GetBitsOnStorage());
      columnBuilder.ValueRange(physicalColumnDesc.GetValueRange());
      columnBuilder.Type(physicalColumnDesc.GetType());
      columnBuilder.RepresentationIndex(physicalColumnDesc.GetRepresentationIndex());
      columnBuilder.Index(fnNextColumnIndex(columnBuilder.GetFieldId(), columnBuilder.GetRepresentationIndex()));

      auto aliasColumnDesc = columnBuilder.MakeDescriptor();
      if (!aliasColumnDesc)
         return R__FORWARD_ERROR(aliasColumnDesc);
      auto resVoid = descBuilder.AddColumn(aliasColumnDesc.Unwrap());
      if (!resVoid)
         return R__FORWARD_ERROR(resVoid);
   }
   bytes = frame + frameSize;

   std::uint32_t nExtraTypeInfos;
   frame = bytes;
   if (auto res = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize, nExtraTypeInfos)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   for (unsigned i = 0; i < nExtraTypeInfos; ++i) {
      RExtraTypeInfoDescriptorBuilder extraTypeInfoBuilder;
      if (auto res = DeserializeExtraTypeInfo(bytes, fnFrameSizeLeft(), extraTypeInfoBuilder)) {
         bytes += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }

      auto extraTypeInfoDesc = extraTypeInfoBuilder.MoveDescriptor();
      // We ignore unknown extra type information
      if (extraTypeInfoDesc)
         descBuilder.AddExtraTypeInfo(extraTypeInfoDesc.Unwrap());
   }
   bytes = frame + frameSize;

   return bytes - base;
}

ROOT::RResult<ROOT::Internal::RNTupleSerializer::RContext>
ROOT::Internal::RNTupleSerializer::SerializeHeader(void *buffer, const ROOT::RNTupleDescriptor &desc)
{
   RContext context;

   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   pos += SerializeEnvelopePreamble(kEnvelopeTypeHeader, *where);
   // So far we don't make use of feature flags
   if (auto res = SerializeFeatureFlags(desc.GetFeatureFlags(), *where)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   pos += SerializeString(desc.GetName(), *where);
   pos += SerializeString(desc.GetDescription(), *where);
   pos += SerializeString(std::string("ROOT v") + ROOT_RELEASE, *where);

   context.MapSchema(desc, /*forHeaderExtension=*/false);

   if (auto res = SerializeSchemaDescription(*where, desc, context)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   std::uint64_t size = pos - base;
   std::uint64_t xxhash3 = 0;
   if (auto res = SerializeEnvelopePostscript(base, size, xxhash3)) {
      size += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   context.SetHeaderSize(size);
   context.SetHeaderXxHash3(xxhash3);
   return context;
}

ROOT::RResult<std::uint32_t>
ROOT::Internal::RNTupleSerializer::SerializePageList(void *buffer, const ROOT::RNTupleDescriptor &desc,
                                                     std::span<ROOT::DescriptorId_t> physClusterIDs,
                                                     const RContext &context)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   pos += SerializeEnvelopePreamble(kEnvelopeTypePageList, *where);

   pos += SerializeUInt64(context.GetHeaderXxHash3(), *where);

   // Cluster summaries
   const auto nClusters = physClusterIDs.size();
   auto clusterSummaryFrame = pos;
   pos += SerializeListFramePreamble(nClusters, *where);
   for (auto clusterId : physClusterIDs) {
      const auto &clusterDesc = desc.GetClusterDescriptor(context.GetMemClusterId(clusterId));
      RClusterSummary summary{clusterDesc.GetFirstEntryIndex(), clusterDesc.GetNEntries(), 0};
      if (auto res = SerializeClusterSummary(summary, *where)) {
         pos += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }
   }
   if (auto res = SerializeFramePostscript(buffer ? clusterSummaryFrame : nullptr, pos - clusterSummaryFrame)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   // Page locations
   auto topMostFrame = pos;
   pos += SerializeListFramePreamble(nClusters, *where);

   for (auto clusterId : physClusterIDs) {
      const auto &clusterDesc = desc.GetClusterDescriptor(context.GetMemClusterId(clusterId));
      // Get an ordered set of physical column ids
      std::set<ROOT::DescriptorId_t> onDiskColumnIds;
      for (const auto &columnRange : clusterDesc.GetColumnRangeIterable())
         onDiskColumnIds.insert(context.GetOnDiskColumnId(columnRange.GetPhysicalColumnId()));

      auto outerFrame = pos;
      pos += SerializeListFramePreamble(onDiskColumnIds.size(), *where);
      for (auto onDiskId : onDiskColumnIds) {
         auto memId = context.GetMemColumnId(onDiskId);
         const auto &columnRange = clusterDesc.GetColumnRange(memId);

         auto innerFrame = pos;
         if (columnRange.IsSuppressed()) {
            // Empty page range
            pos += SerializeListFramePreamble(0, *where);
            pos += SerializeInt64(kSuppressedColumnMarker, *where);
         } else {
            const auto &pageRange = clusterDesc.GetPageRange(memId);
            pos += SerializeListFramePreamble(pageRange.GetPageInfos().size(), *where);

            for (const auto &pi : pageRange.GetPageInfos()) {
               std::int32_t nElements =
                  pi.HasChecksum() ? -static_cast<std::int32_t>(pi.GetNElements()) : pi.GetNElements();
               pos += SerializeUInt32(nElements, *where);
               if (auto res = SerializeLocator(pi.GetLocator(), *where)) {
                  pos += res.Unwrap();
               } else {
                  return R__FORWARD_ERROR(res);
               }
            }
            pos += SerializeInt64(columnRange.GetFirstElementIndex(), *where);
            pos += SerializeUInt32(columnRange.GetCompressionSettings().value(), *where);
         }

         if (auto res = SerializeFramePostscript(buffer ? innerFrame : nullptr, pos - innerFrame)) {
            pos += res.Unwrap();
         } else {
            return R__FORWARD_ERROR(res);
         }
      }
      if (auto res = SerializeFramePostscript(buffer ? outerFrame : nullptr, pos - outerFrame)) {
         pos += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }
   }

   if (auto res = SerializeFramePostscript(buffer ? topMostFrame : nullptr, pos - topMostFrame)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   std::uint64_t size = pos - base;
   if (auto res = SerializeEnvelopePostscript(base, size)) {
      size += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   return size;
}

ROOT::RResult<std::uint32_t> ROOT::Internal::RNTupleSerializer::SerializeFooter(void *buffer,
                                                                                const ROOT::RNTupleDescriptor &desc,
                                                                                const RContext &context)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   pos += SerializeEnvelopePreamble(kEnvelopeTypeFooter, *where);

   // So far we don't make use of footer feature flags
   if (auto res = SerializeFeatureFlags(std::vector<std::uint64_t>(), *where)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   pos += SerializeUInt64(context.GetHeaderXxHash3(), *where);

   // Schema extension, i.e. incremental changes with respect to the header
   auto frame = pos;
   pos += SerializeRecordFramePreamble(*where);
   if (auto res = SerializeSchemaDescription(*where, desc, context, /*forHeaderExtension=*/true)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   if (auto res = SerializeFramePostscript(buffer ? frame : nullptr, pos - frame)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   // Cluster groups
   frame = pos;
   const auto nClusterGroups = desc.GetNClusterGroups();
   pos += SerializeListFramePreamble(nClusterGroups, *where);
   for (unsigned int i = 0; i < nClusterGroups; ++i) {
      const auto &cgDesc = desc.GetClusterGroupDescriptor(context.GetMemClusterGroupId(i));
      RClusterGroup clusterGroup;
      clusterGroup.fMinEntry = cgDesc.GetMinEntry();
      clusterGroup.fEntrySpan = cgDesc.GetEntrySpan();
      clusterGroup.fNClusters = cgDesc.GetNClusters();
      clusterGroup.fPageListEnvelopeLink.fLength = cgDesc.GetPageListLength();
      clusterGroup.fPageListEnvelopeLink.fLocator = cgDesc.GetPageListLocator();
      if (auto res = SerializeClusterGroup(clusterGroup, *where)) {
         pos += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }
   }
   if (auto res = SerializeFramePostscript(buffer ? frame : nullptr, pos - frame)) {
      pos += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   std::uint32_t size = pos - base;
   if (auto res = SerializeEnvelopePostscript(base, size)) {
      size += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   return size;
}

ROOT::RResult<void> ROOT::Internal::RNTupleSerializer::DeserializeHeader(const void *buffer, std::uint64_t bufSize,
                                                                         RNTupleDescriptorBuilder &descBuilder)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   auto fnBufSizeLeft = [&]() { return bufSize - (bytes - base); };

   std::uint64_t xxhash3{0};
   if (auto res = DeserializeEnvelope(bytes, fnBufSizeLeft(), kEnvelopeTypeHeader, xxhash3)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   descBuilder.SetOnDiskHeaderXxHash3(xxhash3);

   std::vector<std::uint64_t> featureFlags;
   if (auto res = DeserializeFeatureFlags(bytes, fnBufSizeLeft(), featureFlags)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   for (std::size_t i = 0; i < featureFlags.size(); ++i) {
      if (!featureFlags[i])
         continue;
      unsigned int bit = 0;
      while (!(featureFlags[i] & (static_cast<uint64_t>(1) << bit)))
         bit++;
      return R__FAIL("unsupported format feature: " + std::to_string(i * 64 + bit));
   }

   std::string name;
   std::string description;
   std::string writer;
   if (auto res = DeserializeString(bytes, fnBufSizeLeft(), name)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   if (auto res = DeserializeString(bytes, fnBufSizeLeft(), description)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   if (auto res = DeserializeString(bytes, fnBufSizeLeft(), writer)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   descBuilder.SetNTuple(name, description);

   // Zero field
   descBuilder.AddField(RFieldDescriptorBuilder()
                           .FieldId(kZeroFieldId)
                           .Structure(ROOT::ENTupleStructure::kRecord)
                           .MakeDescriptor()
                           .Unwrap());
   if (auto res = DeserializeSchemaDescription(bytes, fnBufSizeLeft(), descBuilder)) {
      return RResult<void>::Success();
   } else {
      return R__FORWARD_ERROR(res);
   }
}

ROOT::RResult<void> ROOT::Internal::RNTupleSerializer::DeserializeFooter(const void *buffer, std::uint64_t bufSize,
                                                                         RNTupleDescriptorBuilder &descBuilder)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   auto fnBufSizeLeft = [&]() { return bufSize - (bytes - base); };
   if (auto res = DeserializeEnvelope(bytes, fnBufSizeLeft(), kEnvelopeTypeFooter)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   std::vector<std::uint64_t> featureFlags;
   if (auto res = DeserializeFeatureFlags(bytes, fnBufSizeLeft(), featureFlags)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   for (auto f : featureFlags) {
      if (f)
         R__LOG_WARNING(ROOT::Internal::NTupleLog()) << "Unsupported feature flag! " << f;
   }

   std::uint64_t xxhash3{0};
   if (fnBufSizeLeft() < static_cast<int>(sizeof(std::uint64_t)))
      return R__FAIL("footer too short");
   bytes += DeserializeUInt64(bytes, xxhash3);
   if (xxhash3 != descBuilder.GetDescriptor().GetOnDiskHeaderXxHash3())
      return R__FAIL("XxHash-3 mismatch between header and footer");

   std::uint64_t frameSize;
   auto frame = bytes;
   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - frame); };

   if (auto res = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   if (fnFrameSizeLeft() > 0) {
      descBuilder.BeginHeaderExtension();
      if (auto res = DeserializeSchemaDescription(bytes, fnFrameSizeLeft(), descBuilder); !res) {
         return R__FORWARD_ERROR(res);
      }
   }
   bytes = frame + frameSize;

   std::uint32_t nClusterGroups;
   frame = bytes;
   if (auto res = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize, nClusterGroups)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   for (std::uint32_t groupId = 0; groupId < nClusterGroups; ++groupId) {
      RClusterGroup clusterGroup;
      if (auto res = DeserializeClusterGroup(bytes, fnFrameSizeLeft(), clusterGroup)) {
         bytes += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }

      descBuilder.AddToOnDiskFooterSize(clusterGroup.fPageListEnvelopeLink.fLocator.GetNBytesOnStorage());
      RClusterGroupDescriptorBuilder clusterGroupBuilder;
      clusterGroupBuilder.ClusterGroupId(groupId)
         .PageListLocator(clusterGroup.fPageListEnvelopeLink.fLocator)
         .PageListLength(clusterGroup.fPageListEnvelopeLink.fLength)
         .MinEntry(clusterGroup.fMinEntry)
         .EntrySpan(clusterGroup.fEntrySpan)
         .NClusters(clusterGroup.fNClusters);
      descBuilder.AddClusterGroup(clusterGroupBuilder.MoveDescriptor().Unwrap());
   }
   bytes = frame + frameSize;

   return RResult<void>::Success();
}

ROOT::RResult<std::vector<ROOT::Internal::RClusterDescriptorBuilder>>
ROOT::Internal::RNTupleSerializer::DeserializePageListRaw(const void *buffer, std::uint64_t bufSize,
                                                          ROOT::DescriptorId_t clusterGroupId,
                                                          const ROOT::RNTupleDescriptor &desc)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   auto fnBufSizeLeft = [&]() { return bufSize - (bytes - base); };

   if (auto res = DeserializeEnvelope(bytes, fnBufSizeLeft(), kEnvelopeTypePageList)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   std::uint64_t xxhash3{0};
   if (fnBufSizeLeft() < static_cast<int>(sizeof(std::uint64_t)))
      return R__FAIL("page list too short");
   bytes += DeserializeUInt64(bytes, xxhash3);
   if (xxhash3 != desc.GetOnDiskHeaderXxHash3())
      return R__FAIL("XxHash-3 mismatch between header and page list");

   std::vector<RClusterDescriptorBuilder> clusterBuilders;
   ROOT::DescriptorId_t firstClusterId{0};
   for (ROOT::DescriptorId_t i = 0; i < clusterGroupId; ++i) {
      firstClusterId = firstClusterId + desc.GetClusterGroupDescriptor(i).GetNClusters();
   }

   std::uint64_t clusterSummaryFrameSize;
   auto clusterSummaryFrame = bytes;
   auto fnClusterSummaryFrameSizeLeft = [&]() { return clusterSummaryFrameSize - (bytes - clusterSummaryFrame); };

   std::uint32_t nClusterSummaries;
   if (auto res = DeserializeFrameHeader(bytes, fnBufSizeLeft(), clusterSummaryFrameSize, nClusterSummaries)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }
   for (auto clusterId = firstClusterId; clusterId < firstClusterId + nClusterSummaries; ++clusterId) {
      RClusterSummary clusterSummary;
      if (auto res = DeserializeClusterSummary(bytes, fnClusterSummaryFrameSizeLeft(), clusterSummary)) {
         bytes += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }

      RClusterDescriptorBuilder builder;
      builder.ClusterId(clusterId).FirstEntryIndex(clusterSummary.fFirstEntry).NEntries(clusterSummary.fNEntries);
      clusterBuilders.emplace_back(std::move(builder));
   }
   bytes = clusterSummaryFrame + clusterSummaryFrameSize;

   std::uint64_t topMostFrameSize;
   auto topMostFrame = bytes;
   auto fnTopMostFrameSizeLeft = [&]() { return topMostFrameSize - (bytes - topMostFrame); };

   std::uint32_t nClusters;
   if (auto res = DeserializeFrameHeader(bytes, fnBufSizeLeft(), topMostFrameSize, nClusters)) {
      bytes += res.Unwrap();
   } else {
      return R__FORWARD_ERROR(res);
   }

   if (nClusters != nClusterSummaries)
      return R__FAIL("mismatch between number of clusters and number of cluster summaries");

   for (std::uint32_t i = 0; i < nClusters; ++i) {
      std::uint64_t outerFrameSize;
      auto outerFrame = bytes;
      auto fnOuterFrameSizeLeft = [&]() { return outerFrameSize - (bytes - outerFrame); };

      std::uint32_t nColumns;
      if (auto res = DeserializeFrameHeader(bytes, fnTopMostFrameSizeLeft(), outerFrameSize, nColumns)) {
         bytes += res.Unwrap();
      } else {
         return R__FORWARD_ERROR(res);
      }

      for (std::uint32_t j = 0; j < nColumns; ++j) {
         std::uint64_t innerFrameSize;
         auto innerFrame = bytes;
         auto fnInnerFrameSizeLeft = [&]() { return innerFrameSize - (bytes - innerFrame); };

         std::uint32_t nPages;
         if (auto res = DeserializeFrameHeader(bytes, fnOuterFrameSizeLeft(), innerFrameSize, nPages)) {
            bytes += res.Unwrap();
         } else {
            return R__FORWARD_ERROR(res);
         }

         ROOT::RClusterDescriptor::RPageRange pageRange;
         pageRange.SetPhysicalColumnId(j);
         for (std::uint32_t k = 0; k < nPages; ++k) {
            if (fnInnerFrameSizeLeft() < static_cast<int>(sizeof(std::uint32_t)))
               return R__FAIL("inner frame too short");
            std::int32_t nElements;
            bool hasChecksum = false;
            RNTupleLocator locator;
            bytes += DeserializeInt32(bytes, nElements);
            if (nElements < 0) {
               nElements = -nElements;
               hasChecksum = true;
            }
            if (auto res = DeserializeLocator(bytes, fnInnerFrameSizeLeft(), locator)) {
               bytes += res.Unwrap();
            } else {
               return R__FORWARD_ERROR(res);
            }
            pageRange.GetPageInfos().push_back({static_cast<std::uint32_t>(nElements), locator, hasChecksum});
         }

         if (fnInnerFrameSizeLeft() < static_cast<int>(sizeof(std::int64_t)))
            return R__FAIL("page list frame too short");
         std::int64_t columnOffset;
         bytes += DeserializeInt64(bytes, columnOffset);
         if (columnOffset < 0) {
            if (nPages > 0)
               return R__FAIL("unexpected non-empty page list");
            clusterBuilders[i].MarkSuppressedColumnRange(j);
         } else {
            if (fnInnerFrameSizeLeft() < static_cast<int>(sizeof(std::uint32_t)))
               return R__FAIL("page list frame too short");
            std::uint32_t compressionSettings;
            bytes += DeserializeUInt32(bytes, compressionSettings);
            clusterBuilders[i].CommitColumnRange(j, columnOffset, compressionSettings, pageRange);
         }

         bytes = innerFrame + innerFrameSize;
      } // loop over columns

      bytes = outerFrame + outerFrameSize;
   } // loop over clusters

   return clusterBuilders;
}

ROOT::RResult<void> ROOT::Internal::RNTupleSerializer::DeserializePageList(const void *buffer, std::uint64_t bufSize,
                                                                           ROOT::DescriptorId_t clusterGroupId,
                                                                           ROOT::RNTupleDescriptor &desc,
                                                                           EDescriptorDeserializeMode mode)
{
   auto clusterBuildersRes = RNTupleSerializer::DeserializePageListRaw(buffer, bufSize, clusterGroupId, desc);
   if (!clusterBuildersRes)
      return R__FORWARD_ERROR(clusterBuildersRes);

   auto clusterBuilders = clusterBuildersRes.Unwrap();

   std::vector<ROOT::RClusterDescriptor> clusters;
   clusters.reserve(clusterBuilders.size());

   // Conditionally fixup the clusters depending on the attach purpose
   switch (mode) {
   case EDescriptorDeserializeMode::kForReading:
      for (auto &builder : clusterBuilders) {
         if (auto res = builder.CommitSuppressedColumnRanges(desc); !res)
            return R__FORWARD_RESULT(res);
         builder.AddExtendedColumnRanges(desc);
         clusters.emplace_back(builder.MoveDescriptor().Unwrap());
      }
      break;
   case EDescriptorDeserializeMode::kForWriting:
      for (auto &builder : clusterBuilders) {
         if (auto res = builder.CommitSuppressedColumnRanges(desc); !res)
            return R__FORWARD_RESULT(res);
         clusters.emplace_back(builder.MoveDescriptor().Unwrap());
      }
      break;
   case EDescriptorDeserializeMode::kRaw:
      for (auto &builder : clusterBuilders)
         clusters.emplace_back(builder.MoveDescriptor().Unwrap());
      break;
   }

   desc.AddClusterGroupDetails(clusterGroupId, clusters);

   return RResult<void>::Success();
}

std::string ROOT::Internal::RNTupleSerializer::SerializeStreamerInfos(const StreamerInfoMap_t &infos)
{
   TList streamerInfos;
   for (auto si : infos) {
      assert(si.first == si.second->GetNumber());
      streamerInfos.Add(si.second);
   }
   TBufferFile buffer(TBuffer::kWrite);
   buffer.WriteObject(&streamerInfos);
   assert(buffer.Length() > 0);
   return std::string{buffer.Buffer(), static_cast<UInt_t>(buffer.Length())};
}

ROOT::RResult<ROOT::Internal::RNTupleSerializer::StreamerInfoMap_t>
ROOT::Internal::RNTupleSerializer::DeserializeStreamerInfos(const std::string &extraTypeInfoContent)
{
   StreamerInfoMap_t infoMap;

   TBufferFile buffer(TBuffer::kRead, extraTypeInfoContent.length(), const_cast<char *>(extraTypeInfoContent.data()),
                      false /* adopt */);
   auto infoList = reinterpret_cast<TList *>(buffer.ReadObject(TList::Class()));

   TObjLink *lnk = infoList->FirstLink();
   while (lnk) {
      auto info = reinterpret_cast<TStreamerInfo *>(lnk->GetObject());
      info->BuildCheck();
      infoMap[info->GetNumber()] = info->GetClass()->GetStreamerInfo(info->GetClassVersion());
      assert(info->GetNumber() == infoMap[info->GetNumber()]->GetNumber());
      lnk = lnk->Next();
   }

   delete infoList;

   return infoMap;
}
