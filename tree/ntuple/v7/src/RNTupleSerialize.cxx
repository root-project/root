/// \file RNTupleSerialize.cxx
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

#include <ROOT/RColumnElement.hxx>
#include <ROOT/RColumnModel.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleSerialize.hxx>

#include <RVersion.h>
#include <xxhash.h>

#include <cstring> // for memcpy
#include <deque>
#include <set>
#include <unordered_map>

template <typename T>
using RResult = ROOT::Experimental::RResult<T>;


namespace {
using RNTupleSerializer = ROOT::Experimental::Internal::RNTupleSerializer;

std::uint32_t SerializeField(const ROOT::Experimental::RFieldDescriptor &fieldDesc,
                             ROOT::Experimental::DescriptorId_t onDiskParentId, void *buffer)
{

   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += RNTupleSerializer::SerializeRecordFramePreamble(*where);

   pos += RNTupleSerializer::SerializeUInt32(fieldDesc.GetFieldVersion(), *where);
   pos += RNTupleSerializer::SerializeUInt32(fieldDesc.GetTypeVersion(), *where);
   pos += RNTupleSerializer::SerializeUInt32(onDiskParentId, *where);
   pos += RNTupleSerializer::SerializeFieldStructure(fieldDesc.GetStructure(), *where);
   if (fieldDesc.GetNRepetitions() > 0) {
      pos += RNTupleSerializer::SerializeUInt16(RNTupleSerializer::kFlagRepetitiveField, *where);
      pos += RNTupleSerializer::SerializeUInt64(fieldDesc.GetNRepetitions(), *where);
   } else {
      pos += RNTupleSerializer::SerializeUInt16(0, *where);
   }
   pos += RNTupleSerializer::SerializeString(fieldDesc.GetFieldName(), *where);
   pos += RNTupleSerializer::SerializeString(fieldDesc.GetTypeName(), *where);
   pos += RNTupleSerializer::SerializeString(fieldDesc.GetTypeAlias(), *where);
   pos += RNTupleSerializer::SerializeString(fieldDesc.GetFieldDescription(), *where);

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
std::uint32_t SerializeFieldList(const ROOT::Experimental::RNTupleDescriptor &desc,
                                 std::span<const ROOT::Experimental::DescriptorId_t> fieldList,
                                 std::size_t firstOnDiskId,
                                 const ROOT::Experimental::Internal::RNTupleSerializer::RContext &context, void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   auto fieldZeroId = desc.GetFieldZeroId();
   ROOT::Experimental::DescriptorId_t onDiskFieldId = firstOnDiskId;
   for (auto fieldId : fieldList) {
      const auto &f = desc.GetFieldDescriptor(fieldId);
      auto onDiskParentId =
         (f.GetParentId() == fieldZeroId) ? onDiskFieldId : context.GetOnDiskFieldId(f.GetParentId());
      pos += SerializeField(f, onDiskParentId, *where);
      ++onDiskFieldId;
   }

   return pos - base;
}

RResult<std::uint32_t> DeserializeField(const void *buffer, std::uint64_t bufSize,
                                        ROOT::Experimental::Internal::RFieldDescriptorBuilder &fieldDesc)
{
   using ENTupleStructure = ROOT::Experimental::ENTupleStructure;

   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint64_t frameSize;
   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - base); };
   auto result = RNTupleSerializer::DeserializeFrameHeader(bytes, bufSize, frameSize);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   std::uint32_t fieldVersion;
   std::uint32_t typeVersion;
   std::uint32_t parentId;
   // initialize properly for call to SerializeFieldStructure()
   ENTupleStructure structure{ENTupleStructure::kLeaf};
   std::uint16_t flags;
   if (fnFrameSizeLeft() < 3 * sizeof(std::uint32_t) +
                           RNTupleSerializer::SerializeFieldStructure(structure, nullptr) +
                           sizeof(std::uint16_t))
   {
      return R__FAIL("field record frame too short");
   }
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fieldVersion);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, typeVersion);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, parentId);
   auto res16 = RNTupleSerializer::DeserializeFieldStructure(bytes, structure);
   if (!res16)
      return R__FORWARD_ERROR(res16);
   bytes += res16.Unwrap();
   bytes += RNTupleSerializer::DeserializeUInt16(bytes, flags);
   fieldDesc.FieldVersion(fieldVersion).TypeVersion(typeVersion).ParentId(parentId).Structure(structure);

   if (flags & RNTupleSerializer::kFlagRepetitiveField) {
      if (fnFrameSizeLeft() < sizeof(std::uint64_t))
         return R__FAIL("field record frame too short");
      std::uint64_t nRepetitions;
      bytes += RNTupleSerializer::DeserializeUInt64(bytes, nRepetitions);
      fieldDesc.NRepetitions(nRepetitions);
   }

   std::string fieldName;
   std::string typeName;
   std::string aliasName;
   std::string description;
   result = RNTupleSerializer::DeserializeString(bytes, fnFrameSizeLeft(), fieldName).Unwrap();
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   result = RNTupleSerializer::DeserializeString(bytes, fnFrameSizeLeft(), typeName).Unwrap();
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   result = RNTupleSerializer::DeserializeString(bytes, fnFrameSizeLeft(), aliasName).Unwrap();
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   result = RNTupleSerializer::DeserializeString(bytes, fnFrameSizeLeft(), description).Unwrap();
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   fieldDesc.FieldName(fieldName).TypeName(typeName).TypeAlias(aliasName).FieldDescription(description);

   return frameSize;
}

std::uint32_t SerializeColumnList(const ROOT::Experimental::RNTupleDescriptor &desc,
                                  std::span<const ROOT::Experimental::DescriptorId_t> fieldList,
                                  const ROOT::Experimental::Internal::RNTupleSerializer::RContext &context,
                                  void *buffer)
{
   using RColumnElementBase = ROOT::Experimental::Internal::RColumnElementBase;

   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   for (auto parentId : fieldList) {
      for (const auto &c : desc.GetColumnIterable(parentId)) {
         if (c.IsAliasColumn())
            continue;

         auto frame = pos;
         pos += RNTupleSerializer::SerializeRecordFramePreamble(*where);

         auto type = c.GetModel().GetType();
         pos += RNTupleSerializer::SerializeColumnType(type, *where);
         pos += RNTupleSerializer::SerializeUInt16(RColumnElementBase::GetBitsOnStorage(type), *where);
         pos += RNTupleSerializer::SerializeUInt32(context.GetOnDiskFieldId(c.GetFieldId()), *where);
         std::uint32_t flags = 0;
         // TODO(jblomer): add support for descending columns in the column model
         if (c.GetModel().GetIsSorted())
            flags |= RNTupleSerializer::kFlagSortAscColumn;
         // TODO(jblomer): fix for unsigned integer types
         if (type == ROOT::Experimental::EColumnType::kIndex32)
            flags |= RNTupleSerializer::kFlagNonNegativeColumn;
         const std::uint64_t firstElementIdx = c.GetFirstElementIndex();
         if (firstElementIdx > 0)
            flags |= RNTupleSerializer::kFlagDeferredColumn;
         pos += RNTupleSerializer::SerializeUInt32(flags, *where);
         if (flags & RNTupleSerializer::kFlagDeferredColumn)
            pos += RNTupleSerializer::SerializeUInt64(firstElementIdx, *where);

         pos += RNTupleSerializer::SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);
      }
   }

   return pos - base;
}

RResult<std::uint32_t> DeserializeColumn(const void *buffer, std::uint64_t bufSize,
                                         ROOT::Experimental::Internal::RColumnDescriptorBuilder &columnDesc)
{
   using EColumnType = ROOT::Experimental::EColumnType;

   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint64_t frameSize;
   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - base); };
   auto result = RNTupleSerializer::DeserializeFrameHeader(bytes, bufSize, frameSize);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   // Initialize properly for SerializeColumnType
   EColumnType type{EColumnType::kIndex32};
   std::uint16_t bitsOnStorage;
   std::uint32_t fieldId;
   std::uint32_t flags;
   std::uint64_t firstElementIdx = 0;
   if (fnFrameSizeLeft() < RNTupleSerializer::SerializeColumnType(type, nullptr) +
                           sizeof(std::uint16_t) + 2 * sizeof(std::uint32_t))
   {
      return R__FAIL("column record frame too short");
   }
   auto res16 = RNTupleSerializer::DeserializeColumnType(bytes, type);
   if (!res16)
      return R__FORWARD_ERROR(res16);
   bytes += res16.Unwrap();
   bytes += RNTupleSerializer::DeserializeUInt16(bytes, bitsOnStorage);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fieldId);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, flags);
   if (flags & RNTupleSerializer::kFlagDeferredColumn) {
      if (fnFrameSizeLeft() < sizeof(std::uint64_t))
         return R__FAIL("column record frame too short");
      bytes += RNTupleSerializer::DeserializeUInt64(bytes, firstElementIdx);
   }

   if (ROOT::Experimental::Internal::RColumnElementBase::GetBitsOnStorage(type) != bitsOnStorage)
      return R__FAIL("column element size mismatch");

   const bool isSorted = (flags & (RNTupleSerializer::kFlagSortAscColumn | RNTupleSerializer::kFlagSortDesColumn));
   columnDesc.FieldId(fieldId).Model({type, isSorted}).FirstElementIndex(firstElementIdx);

   return frameSize;
}

std::uint32_t SerializeLocatorPayloadURI(const ROOT::Experimental::RNTupleLocator &locator, unsigned char *buffer)
{
   const auto &uri = locator.GetPosition<std::string>();
   if (uri.length() >= (1 << 16))
      throw ROOT::Experimental::RException(R__FAIL("locator too large"));
   if (buffer)
      memcpy(buffer, uri.data(), uri.length());
   return uri.length();
}

void DeserializeLocatorPayloadURI(const unsigned char *buffer, std::uint32_t payloadSize,
                                  ROOT::Experimental::RNTupleLocator &locator)
{
   locator.fBytesOnStorage = 0;
   auto &uri = locator.fPosition.emplace<std::string>();
   uri.resize(payloadSize);
   memcpy(uri.data(), buffer, payloadSize);
}

std::uint32_t SerializeLocatorPayloadObject64(const ROOT::Experimental::RNTupleLocator &locator, unsigned char *buffer)
{
   const auto &data = locator.GetPosition<ROOT::Experimental::RNTupleLocatorObject64>();
   if (buffer) {
      RNTupleSerializer::SerializeUInt32(locator.fBytesOnStorage, buffer);
      RNTupleSerializer::SerializeUInt64(data.fLocation, buffer + sizeof(std::uint32_t));
   }
   return sizeof(std::uint32_t) + sizeof(std::uint64_t);
}

void DeserializeLocatorPayloadObject64(const unsigned char *buffer, ROOT::Experimental::RNTupleLocator &locator)
{
   auto &data = locator.fPosition.emplace<ROOT::Experimental::RNTupleLocatorObject64>();
   RNTupleSerializer::DeserializeUInt32(buffer, locator.fBytesOnStorage);
   RNTupleSerializer::DeserializeUInt64(buffer + sizeof(std::uint32_t), data.fLocation);
}

std::uint32_t SerializeAliasColumnList(const ROOT::Experimental::RNTupleDescriptor &desc,
                                       std::span<const ROOT::Experimental::DescriptorId_t> fieldList,
                                       const ROOT::Experimental::Internal::RNTupleSerializer::RContext &context,
                                       void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   for (auto parentId : fieldList) {
      for (const auto &c : desc.GetColumnIterable(parentId)) {
         if (!c.IsAliasColumn())
            continue;

         auto frame = pos;
         pos += RNTupleSerializer::SerializeRecordFramePreamble(*where);

         pos += RNTupleSerializer::SerializeUInt32(context.GetOnDiskColumnId(c.GetPhysicalId()), *where);
         pos += RNTupleSerializer::SerializeUInt32(context.GetOnDiskFieldId(c.GetFieldId()), *where);

         pos += RNTupleSerializer::SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);
      }
   }

   return pos - base;
}

RResult<std::uint32_t> DeserializeAliasColumn(const void *buffer, std::uint64_t bufSize,
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

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeXxHash3(const unsigned char *data,
                                                                                std::uint64_t length,
                                                                                std::uint64_t &xxhash3, void *buffer)
{
   if (buffer != nullptr) {
      xxhash3 = XXH3_64bits(data, length);
      SerializeUInt64(xxhash3, buffer);
   }
   return 8;
}

RResult<void> ROOT::Experimental::Internal::RNTupleSerializer::VerifyXxHash3(const unsigned char *data,
                                                                             std::uint64_t length,
                                                                             std::uint64_t &xxhash3)
{
   auto checksumReal = XXH3_64bits(data, length);
   DeserializeUInt64(data + length, xxhash3);
   if (xxhash3 != checksumReal)
      return R__FAIL("XxHash-3 checksum mismatch");
   return RResult<void>::Success();
}

RResult<void>
ROOT::Experimental::Internal::RNTupleSerializer::VerifyXxHash3(const unsigned char *data, std::uint64_t length)
{
   std::uint64_t xxhash3;
   return R__FORWARD_RESULT(VerifyXxHash3(data, length, xxhash3));
}


std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeInt16(std::int16_t val, void *buffer)
{
   if (buffer != nullptr) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);
      bytes[0] = (val & 0x00FF);
      bytes[1] = (val & 0xFF00) >> 8;
   }
   return 2;
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::DeserializeInt16(const void *buffer, std::int16_t &val)
{
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   val = std::int16_t(bytes[0]) + (std::int16_t(bytes[1]) << 8);
   return 2;
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeUInt16(std::uint16_t val, void *buffer)
{
   return SerializeInt16(val, buffer);
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::DeserializeUInt16(const void *buffer, std::uint16_t &val)
{
   return DeserializeInt16(buffer, *reinterpret_cast<std::int16_t *>(&val));
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeInt32(std::int32_t val, void *buffer)
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

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::DeserializeInt32(const void *buffer, std::int32_t &val)
{
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   val = std::int32_t(bytes[0]) + (std::int32_t(bytes[1]) << 8) +
         (std::int32_t(bytes[2]) << 16) + (std::int32_t(bytes[3]) << 24);
   return 4;
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeUInt32(std::uint32_t val, void *buffer)
{
   return SerializeInt32(val, buffer);
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::DeserializeUInt32(const void *buffer, std::uint32_t &val)
{
   return DeserializeInt32(buffer, *reinterpret_cast<std::int32_t *>(&val));
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeInt64(std::int64_t val, void *buffer)
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

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::DeserializeInt64(const void *buffer, std::int64_t &val)
{
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   val = std::int64_t(bytes[0]) + (std::int64_t(bytes[1]) << 8) +
         (std::int64_t(bytes[2]) << 16) + (std::int64_t(bytes[3]) << 24) +
         (std::int64_t(bytes[4]) << 32) + (std::int64_t(bytes[5]) << 40) +
         (std::int64_t(bytes[6]) << 48) + (std::int64_t(bytes[7]) << 56);
   return 8;
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeUInt64(std::uint64_t val, void *buffer)
{
   return SerializeInt64(val, buffer);
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::DeserializeUInt64(const void *buffer, std::uint64_t &val)
{
   return DeserializeInt64(buffer, *reinterpret_cast<std::int64_t *>(&val));
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeString(const std::string &val, void *buffer)
{
   if (buffer) {
      auto pos = reinterpret_cast<unsigned char *>(buffer);
      pos += SerializeUInt32(val.length(), pos);
      memcpy(pos, val.data(), val.length());
   }
   return sizeof(std::uint32_t) + val.length();
}

RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeString(const void *buffer,
                                                                                          std::uint64_t bufSize,
                                                                                          std::string &val)
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


std::uint16_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeColumnType(
   ROOT::Experimental::EColumnType type, void *buffer)
{
   using EColumnType = ROOT::Experimental::EColumnType;
   switch (type) {
   case EColumnType::kIndex64: return SerializeUInt16(0x01, buffer);
   case EColumnType::kIndex32: return SerializeUInt16(0x02, buffer);
   case EColumnType::kSwitch: return SerializeUInt16(0x03, buffer);
   case EColumnType::kByte: return SerializeUInt16(0x04, buffer);
   case EColumnType::kChar: return SerializeUInt16(0x05, buffer);
   case EColumnType::kBit: return SerializeUInt16(0x06, buffer);
   case EColumnType::kReal64: return SerializeUInt16(0x07, buffer);
   case EColumnType::kReal32: return SerializeUInt16(0x08, buffer);
   case EColumnType::kReal16: return SerializeUInt16(0x09, buffer);
   case EColumnType::kInt64: return SerializeUInt16(0x16, buffer);
   case EColumnType::kUInt64: return SerializeUInt16(0x0A, buffer);
   case EColumnType::kInt32: return SerializeUInt16(0x17, buffer);
   case EColumnType::kUInt32: return SerializeUInt16(0x0B, buffer);
   case EColumnType::kInt16: return SerializeUInt16(0x18, buffer);
   case EColumnType::kUInt16: return SerializeUInt16(0x0C, buffer);
   case EColumnType::kInt8: return SerializeUInt16(0x19, buffer);
   case EColumnType::kUInt8: return SerializeUInt16(0x0D, buffer);
   case EColumnType::kSplitIndex64: return SerializeUInt16(0x0E, buffer);
   case EColumnType::kSplitIndex32: return SerializeUInt16(0x0F, buffer);
   case EColumnType::kSplitReal64: return SerializeUInt16(0x10, buffer);
   case EColumnType::kSplitReal32: return SerializeUInt16(0x11, buffer);
   case EColumnType::kSplitInt64: return SerializeUInt16(0x1A, buffer);
   case EColumnType::kSplitUInt64: return SerializeUInt16(0x13, buffer);
   case EColumnType::kSplitInt32: return SerializeUInt16(0x1B, buffer);
   case EColumnType::kSplitUInt32: return SerializeUInt16(0x14, buffer);
   case EColumnType::kSplitInt16: return SerializeUInt16(0x1C, buffer);
   case EColumnType::kSplitUInt16: return SerializeUInt16(0x15, buffer);
   default: throw RException(R__FAIL("ROOT bug: unexpected column type"));
   }
}


RResult<std::uint16_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeColumnType(
   const void *buffer, ROOT::Experimental::EColumnType &type)
{
   using EColumnType = ROOT::Experimental::EColumnType;
   std::uint16_t onDiskType;
   auto result = DeserializeUInt16(buffer, onDiskType);
   switch (onDiskType) {
   case 0x01: type = EColumnType::kIndex64; break;
   case 0x02: type = EColumnType::kIndex32; break;
   case 0x03: type = EColumnType::kSwitch; break;
   case 0x04: type = EColumnType::kByte; break;
   case 0x05: type = EColumnType::kChar; break;
   case 0x06: type = EColumnType::kBit; break;
   case 0x07: type = EColumnType::kReal64; break;
   case 0x08: type = EColumnType::kReal32; break;
   case 0x09: type = EColumnType::kReal16; break;
   case 0x16: type = EColumnType::kInt64; break;
   case 0x0A: type = EColumnType::kUInt64; break;
   case 0x17: type = EColumnType::kInt32; break;
   case 0x0B: type = EColumnType::kUInt32; break;
   case 0x18: type = EColumnType::kInt16; break;
   case 0x0C: type = EColumnType::kUInt16; break;
   case 0x19: type = EColumnType::kInt8; break;
   case 0x0D: type = EColumnType::kUInt8; break;
   case 0x0E: type = EColumnType::kSplitIndex64; break;
   case 0x0F: type = EColumnType::kSplitIndex32; break;
   case 0x10: type = EColumnType::kSplitReal64; break;
   case 0x11: type = EColumnType::kSplitReal32; break;
   case 0x1A: type = EColumnType::kSplitInt64; break;
   case 0x13: type = EColumnType::kSplitUInt64; break;
   case 0x1B: type = EColumnType::kSplitInt32; break;
   case 0x14: type = EColumnType::kSplitUInt32; break;
   case 0x1C: type = EColumnType::kSplitInt16; break;
   case 0x15: type = EColumnType::kSplitUInt16; break;
   default: return R__FAIL("unexpected on-disk column type");
   }
   return result;
}


std::uint16_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeFieldStructure(
   ROOT::Experimental::ENTupleStructure structure, void *buffer)
{
   using ENTupleStructure = ROOT::Experimental::ENTupleStructure;
   switch (structure) {
      case ENTupleStructure::kLeaf:
         return SerializeUInt16(0x00, buffer);
      case ENTupleStructure::kCollection:
         return SerializeUInt16(0x01, buffer);
      case ENTupleStructure::kRecord:
         return SerializeUInt16(0x02, buffer);
      case ENTupleStructure::kVariant:
         return SerializeUInt16(0x03, buffer);
      case ENTupleStructure::kReference:
         return SerializeUInt16(0x04, buffer);
      default:
         throw RException(R__FAIL("ROOT bug: unexpected field structure type"));
   }
}


RResult<std::uint16_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeFieldStructure(
   const void *buffer, ROOT::Experimental::ENTupleStructure &structure)
{
   using ENTupleStructure = ROOT::Experimental::ENTupleStructure;
   std::uint16_t onDiskValue;
   auto result = DeserializeUInt16(buffer, onDiskValue);
   switch (onDiskValue) {
      case 0x00:
         structure = ENTupleStructure::kLeaf;
         break;
      case 0x01:
         structure = ENTupleStructure::kCollection;
         break;
      case 0x02:
         structure = ENTupleStructure::kRecord;
         break;
      case 0x03:
         structure = ENTupleStructure::kVariant;
         break;
      case 0x04:
         structure = ENTupleStructure::kReference;
         break;
      default:
         return R__FAIL("unexpected on-disk field structure value");
   }
   return result;
}

std::uint32_t
ROOT::Experimental::Internal::RNTupleSerializer::SerializeEnvelopePreamble(std::uint16_t envelopeType, void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += SerializeUInt64(envelopeType, *where);
   // The 48bits size information is filled in the postscript
   return pos - base;
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeEnvelopePostscript(unsigned char *envelope,
                                                                                           std::uint64_t size,
                                                                                           std::uint64_t &xxhash3)
{
   if (size < sizeof(std::uint64_t))
      throw RException(R__FAIL("envelope size too small"));
   if (size >= static_cast<uint64_t>(1) << 48)
      throw RException(R__FAIL("envelope size too big"));
   if (envelope) {
      std::uint64_t typeAndSize;
      DeserializeUInt64(envelope, typeAndSize);
      typeAndSize |= (size + 8) << 16;
      SerializeUInt64(typeAndSize, envelope);
   }
   return SerializeXxHash3(envelope, size, xxhash3, envelope ? (envelope + size) : nullptr);
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeEnvelopePostscript(unsigned char *envelope,
                                                                                           std::uint64_t size)
{
   std::uint64_t xxhash3;
   return SerializeEnvelopePostscript(envelope, size, xxhash3);
}

RResult<std::uint32_t>
ROOT::Experimental::Internal::RNTupleSerializer::DeserializeEnvelope(const void *buffer, std::uint64_t bufSize,
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

RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeEnvelope(const void *buffer,
                                                                                            std::uint64_t bufSize,
                                                                                            std::uint16_t expectedType)
{
   std::uint64_t xxhash3;
   return R__FORWARD_RESULT(DeserializeEnvelope(buffer, bufSize, expectedType, xxhash3));
}


std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeRecordFramePreamble(void *buffer)
{
   // Marker: multiply the final size with 1
   return SerializeInt64(1, buffer);
}


std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeListFramePreamble(
   std::uint32_t nitems, void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   // Marker: multiply the final size with -1
   pos += SerializeInt64(-1, *where);
   pos += SerializeUInt32(nitems, *where);
   return pos - base;
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeFramePostscript(void *frame, std::uint64_t size)
{
   auto preambleSize = sizeof(std::int64_t);
   if (size < preambleSize)
      throw RException(R__FAIL("frame too short: " + std::to_string(size)));
   if (frame) {
      std::int64_t marker;
      DeserializeInt64(frame, marker);
      if ((marker < 0) && (size < (sizeof(std::uint32_t) + preambleSize)))
         throw RException(R__FAIL("frame too short: " + std::to_string(size)));
      SerializeInt64(marker * static_cast<int64_t>(size), frame);
   }
   return 0;
}

ROOT::Experimental::RResult<std::uint32_t>
ROOT::Experimental::Internal::RNTupleSerializer::DeserializeFrameHeader(const void *buffer, std::uint64_t bufSize,
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

ROOT::Experimental::RResult<std::uint32_t>
ROOT::Experimental::Internal::RNTupleSerializer::DeserializeFrameHeader(const void *buffer, std::uint64_t bufSize,
                                                                        std::uint64_t &frameSize)
{
   std::uint32_t nitems;
   return R__FORWARD_RESULT(DeserializeFrameHeader(buffer, bufSize, frameSize, nitems));
}

std::uint32_t
ROOT::Experimental::Internal::RNTupleSerializer::SerializeFeatureFlags(const std::vector<std::uint64_t> &flags,
                                                                       void *buffer)
{
   if (flags.empty())
      return SerializeUInt64(0, buffer);

   if (buffer) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);

      for (unsigned i = 0; i < flags.size(); ++i) {
         if (flags[i] & 0x8000000000000000)
            throw RException(R__FAIL("feature flag out of bounds"));

         // The MSb indicates that another Int64 follows; set this bit to 1 for all except the last element
         if (i == (flags.size() - 1))
            SerializeUInt64(flags[i], bytes);
         else
            bytes += SerializeUInt64(flags[i] | 0x8000000000000000, bytes);
      }
   }
   return (flags.size() * sizeof(std::int64_t));
}

RResult<std::uint32_t>
ROOT::Experimental::Internal::RNTupleSerializer::DeserializeFeatureFlags(const void *buffer, std::uint64_t bufSize,
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

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeLocator(
   const RNTupleLocator &locator, void *buffer)
{
   if (locator.fType > RNTupleLocator::kLastSerializableType)
      throw RException(R__FAIL("locator is not serializable"));

   std::uint32_t size = 0;
   if (locator.fType == RNTupleLocator::kTypeFile) {
      if (static_cast<std::int32_t>(locator.fBytesOnStorage) < 0)
         throw RException(R__FAIL("locator too large"));
      size += SerializeUInt32(locator.fBytesOnStorage, buffer);
      size += SerializeUInt64(locator.GetPosition<std::uint64_t>(),
                              buffer ? reinterpret_cast<unsigned char *>(buffer) + size : nullptr);
      return size;
   }

   auto payloadp = buffer ? reinterpret_cast<unsigned char *>(buffer) + sizeof(std::int32_t) : nullptr;
   switch (locator.fType) {
   case RNTupleLocator::kTypeURI: size += SerializeLocatorPayloadURI(locator, payloadp); break;
   case RNTupleLocator::kTypeDAOS: size += SerializeLocatorPayloadObject64(locator, payloadp); break;
   default: throw RException(R__FAIL("locator has unknown type"));
   }
   std::int32_t head = sizeof(std::int32_t) + size;
   head |= locator.fReserved << 16;
   head |= static_cast<int>(locator.fType & 0x7F) << 24;
   head = -head;
   size += RNTupleSerializer::SerializeInt32(head, buffer);
   return size;
}

RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeLocator(const void *buffer,
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
      locator.fType = static_cast<RNTupleLocator::ELocatorType>(type);
      locator.fReserved = static_cast<std::uint32_t>(head >> 16) & 0xFF;
      switch (type) {
      case RNTupleLocator::kTypeURI: DeserializeLocatorPayloadURI(bytes, payloadSize, locator); break;
      case RNTupleLocator::kTypeDAOS: DeserializeLocatorPayloadObject64(bytes, locator); break;
      default: return R__FAIL("unsupported locator type: " + std::to_string(type));
      }
      bytes += payloadSize;
   } else {
      if (bufSize < sizeof(std::uint64_t))
         return R__FAIL("too short locator");
      auto &offset = locator.fPosition.emplace<std::uint64_t>();
      locator.fType = RNTupleLocator::kTypeFile;
      bytes += DeserializeUInt64(bytes, offset);
      locator.fBytesOnStorage = head;
   }

   return bytes - reinterpret_cast<const unsigned char *>(buffer);
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeEnvelopeLink(
   const REnvelopeLink &envelopeLink, void *buffer)
{
   auto size = SerializeUInt64(envelopeLink.fLength, buffer);
   size += SerializeLocator(envelopeLink.fLocator,
                            buffer ? reinterpret_cast<unsigned char *>(buffer) + size : nullptr);
   return size;
}

RResult<std::uint32_t>
ROOT::Experimental::Internal::RNTupleSerializer::DeserializeEnvelopeLink(const void *buffer, std::uint64_t bufSize,
                                                                         REnvelopeLink &envelopeLink)
{
   if (bufSize < sizeof(std::int64_t))
      return R__FAIL("too short envelope link");

   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   bytes += DeserializeUInt64(bytes, envelopeLink.fLength);
   bufSize -= sizeof(std::uint64_t);
   auto result = DeserializeLocator(bytes, bufSize, envelopeLink.fLocator);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   return bytes - reinterpret_cast<const unsigned char *>(buffer);
}


std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeClusterSummary(
   const RClusterSummary &clusterSummary, void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   auto frame = pos;
   pos += SerializeRecordFramePreamble(*where);
   pos += SerializeUInt64(clusterSummary.fFirstEntry, *where);
   if (clusterSummary.fColumnGroupID >= 0) {
      pos += SerializeInt64(-static_cast<int64_t>(clusterSummary.fNEntries), *where);
      pos += SerializeUInt32(clusterSummary.fColumnGroupID, *where);
   } else {
      pos += SerializeInt64(static_cast<int64_t>(clusterSummary.fNEntries), *where);
   }
   auto size = pos - frame;
   pos += SerializeFramePostscript(frame, size);
   return size;
}

RResult<std::uint32_t>
ROOT::Experimental::Internal::RNTupleSerializer::DeserializeClusterSummary(const void *buffer, std::uint64_t bufSize,
                                                                           RClusterSummary &clusterSummary)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint64_t frameSize;
   auto result = DeserializeFrameHeader(bytes, bufSize, frameSize);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - base); };
   if (fnFrameSizeLeft() < 2 * sizeof(std::uint64_t))
      return R__FAIL("too short cluster summary");

   bytes += DeserializeUInt64(bytes, clusterSummary.fFirstEntry);
   std::int64_t nEntries;
   bytes += DeserializeInt64(bytes, nEntries);

   if (nEntries < 0) {
      if (fnFrameSizeLeft() < sizeof(std::uint32_t))
         return R__FAIL("too short cluster summary");
      clusterSummary.fNEntries = -nEntries;
      std::uint32_t columnGroupID;
      bytes += DeserializeUInt32(bytes, columnGroupID);
      clusterSummary.fColumnGroupID = columnGroupID;
   } else {
      clusterSummary.fNEntries = nEntries;
      clusterSummary.fColumnGroupID = -1;
   }

   return frameSize;
}


std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeClusterGroup(
   const RClusterGroup &clusterGroup, void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   auto frame = pos;
   pos += SerializeRecordFramePreamble(*where);
   pos += SerializeUInt64(clusterGroup.fMinEntry, *where);
   pos += SerializeUInt64(clusterGroup.fEntrySpan, *where);
   pos += SerializeUInt32(clusterGroup.fNClusters, *where);
   pos += SerializeEnvelopeLink(clusterGroup.fPageListEnvelopeLink, *where);
   auto size = pos - frame;
   pos += SerializeFramePostscript(frame, size);
   return size;
}

RResult<std::uint32_t>
ROOT::Experimental::Internal::RNTupleSerializer::DeserializeClusterGroup(const void *buffer, std::uint64_t bufSize,
                                                                         RClusterGroup &clusterGroup)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;

   std::uint64_t frameSize;
   auto result = DeserializeFrameHeader(bytes, bufSize, frameSize);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - base); };
   if (fnFrameSizeLeft() < sizeof(std::uint32_t) + 2 * sizeof(std::uint64_t))
      return R__FAIL("too short cluster group");

   bytes += DeserializeUInt64(bytes, clusterGroup.fMinEntry);
   bytes += DeserializeUInt64(bytes, clusterGroup.fEntrySpan);
   bytes += DeserializeUInt32(bytes, clusterGroup.fNClusters);
   result = DeserializeEnvelopeLink(bytes, fnFrameSizeLeft(), clusterGroup.fPageListEnvelopeLink);
   if (!result)
      return R__FORWARD_ERROR(result);

   return frameSize;
}

void ROOT::Experimental::Internal::RNTupleSerializer::RContext::MapSchema(const RNTupleDescriptor &desc,
                                                                          bool forHeaderExtension)
{
   auto fieldZeroId = desc.GetFieldZeroId();
   auto depthFirstTraversal = [&](std::span<DescriptorId_t> fieldTrees, auto doForEachField) {
      std::deque<DescriptorId_t> idQueue{fieldTrees.begin(), fieldTrees.end()};
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
   if (!forHeaderExtension)
      R__ASSERT(GetHeaderExtensionOffset() == -1U);

   std::vector<DescriptorId_t> fieldTrees;
   if (!forHeaderExtension) {
      fieldTrees.emplace_back(fieldZeroId);
   } else if (auto xHeader = desc.GetHeaderExtension()) {
      fieldTrees = xHeader->GetTopLevelFields(desc);
   }
   depthFirstTraversal(fieldTrees, [&](DescriptorId_t fieldId) { MapFieldId(fieldId); });
   depthFirstTraversal(fieldTrees, [&](DescriptorId_t fieldId) {
      for (const auto &c : desc.GetColumnIterable(fieldId))
         if (!c.IsAliasColumn())
            MapColumnId(c.GetLogicalId());
   });
   depthFirstTraversal(fieldTrees, [&](DescriptorId_t fieldId) {
      for (const auto &c : desc.GetColumnIterable(fieldId))
         if (c.IsAliasColumn())
            MapColumnId(c.GetLogicalId());
   });

   // Anything added after this point is accounted for the header extension
   if (!forHeaderExtension)
      BeginHeaderExtension();
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeSchemaDescription(void *buffer,
                                                                                          const RNTupleDescriptor &desc,
                                                                                          const RContext &context,
                                                                                          bool forHeaderExtension)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   std::size_t nFields = 0, nColumns = 0, nAliasColumns = 0, fieldListOffset = 0;
   if (forHeaderExtension) {
      // A call to `RNTupleDescriptorBuilder::BeginHeaderExtension()` is not strictly required after serializing the
      // header, which may happen, e.g., in unit tests.  Ensure an empty schema extension is serialized in this case
      if (auto xHeader = desc.GetHeaderExtension()) {
         nFields = xHeader->GetNFields();
         nColumns = xHeader->GetNPhysicalColumns();
         nAliasColumns = xHeader->GetNLogicalColumns() - xHeader->GetNPhysicalColumns();
         fieldListOffset = context.GetHeaderExtensionOffset();
      }
   } else {
      nFields = desc.GetNFields() - 1;
      nColumns = desc.GetNPhysicalColumns();
      nAliasColumns = desc.GetNLogicalColumns() - desc.GetNPhysicalColumns();
   }
   const auto &onDiskFields = context.GetOnDiskFieldList();
   R__ASSERT(onDiskFields.size() >= fieldListOffset);
   std::span<const DescriptorId_t> fieldList{onDiskFields.data() + fieldListOffset,
                                             onDiskFields.size() - fieldListOffset};

   auto frame = pos;
   pos += SerializeListFramePreamble(nFields, *where);
   pos += SerializeFieldList(desc, fieldList, /*firstOnDiskId=*/fieldListOffset, context, *where);
   pos += SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);

   frame = pos;
   pos += SerializeListFramePreamble(nColumns, *where);
   pos += SerializeColumnList(desc, fieldList, context, *where);
   pos += SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);

   frame = pos;
   pos += SerializeListFramePreamble(nAliasColumns, *where);
   pos += SerializeAliasColumnList(desc, fieldList, context, *where);
   pos += SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);

   // We don't use extra type information yet
   frame = pos;
   pos += SerializeListFramePreamble(0, *where);
   pos += SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);
   return static_cast<std::uint32_t>(pos - base);
}

RResult<std::uint32_t>
ROOT::Experimental::Internal::RNTupleSerializer::DeserializeSchemaDescription(const void *buffer, std::uint64_t bufSize,
                                                                              RNTupleDescriptorBuilder &descBuilder)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   auto fnBufSizeLeft = [&]() { return bufSize - (bytes - base); };
   RResult<std::uint32_t> result{0};

   std::uint64_t frameSize;
   auto frame = bytes;
   auto fnFrameSizeLeft = [&]() { return frameSize - (bytes - frame); };

   std::uint32_t nFields;
   result = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize, nFields);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   // The zero field is always added before `DeserializeSchemaDescription()` is called
   const std::uint32_t fieldIdRangeBegin = descBuilder.GetDescriptor().GetNFields() - 1;
   for (unsigned i = 0; i < nFields; ++i) {
      std::uint32_t fieldId = fieldIdRangeBegin + i;
      RFieldDescriptorBuilder fieldBuilder;
      result = DeserializeField(bytes, fnFrameSizeLeft(), fieldBuilder);
      if (!result)
         return R__FORWARD_ERROR(result);
      bytes += result.Unwrap();
      if (fieldId == fieldBuilder.GetParentId())
         fieldBuilder.ParentId(kZeroFieldId);
      auto fieldDesc = fieldBuilder.FieldId(fieldId).MakeDescriptor();
      if (!fieldDesc)
         return R__FORWARD_ERROR(fieldDesc);
      auto parentId = fieldDesc.Inspect().GetParentId();
      descBuilder.AddField(fieldDesc.Unwrap());
      auto resVoid = descBuilder.AddFieldLink(parentId, fieldId);
      if (!resVoid)
         return R__FORWARD_ERROR(resVoid);
   }
   bytes = frame + frameSize;

   std::uint32_t nColumns;
   frame = bytes;
   result = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize, nColumns);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   const std::uint32_t columnIdRangeBegin = descBuilder.GetDescriptor().GetNLogicalColumns();
   std::unordered_map<DescriptorId_t, std::uint32_t> maxIndexes;
   for (unsigned i = 0; i < nColumns; ++i) {
      std::uint32_t columnId = columnIdRangeBegin + i;
      RColumnDescriptorBuilder columnBuilder;
      result = DeserializeColumn(bytes, fnFrameSizeLeft(), columnBuilder);
      if (!result)
         return R__FORWARD_ERROR(result);
      bytes += result.Unwrap();

      std::uint32_t idx = 0;
      const auto fieldId = columnBuilder.GetFieldId();
      auto maxIdx = maxIndexes.find(fieldId);
      if (maxIdx != maxIndexes.end())
         idx = maxIdx->second + 1;
      maxIndexes[fieldId] = idx;

      auto columnDesc = columnBuilder.Index(idx).LogicalColumnId(columnId).PhysicalColumnId(columnId).MakeDescriptor();
      if (!columnDesc)
         return R__FORWARD_ERROR(columnDesc);
      auto resVoid = descBuilder.AddColumn(columnDesc.Unwrap());
      if (!resVoid)
         return R__FORWARD_ERROR(resVoid);
   }
   bytes = frame + frameSize;

   std::uint32_t nAliasColumns;
   frame = bytes;
   result = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize, nAliasColumns);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   const std::uint32_t aliasColumnIdRangeBegin = columnIdRangeBegin + nColumns;
   for (unsigned i = 0; i < nAliasColumns; ++i) {
      std::uint32_t physicalId;
      std::uint32_t fieldId;
      result = DeserializeAliasColumn(bytes, fnFrameSizeLeft(), physicalId, fieldId);
      if (!result)
         return R__FORWARD_ERROR(result);
      bytes += result.Unwrap();

      RColumnDescriptorBuilder columnBuilder;
      columnBuilder.LogicalColumnId(aliasColumnIdRangeBegin + i).PhysicalColumnId(physicalId).FieldId(fieldId);
      columnBuilder.Model(descBuilder.GetDescriptor().GetColumnDescriptor(physicalId).GetModel());

      std::uint32_t idx = 0;
      auto maxIdx = maxIndexes.find(fieldId);
      if (maxIdx != maxIndexes.end())
         idx = maxIdx->second + 1;
      maxIndexes[fieldId] = idx;

      auto aliasColumnDesc = columnBuilder.Index(idx).MakeDescriptor();
      if (!aliasColumnDesc)
         return R__FORWARD_ERROR(aliasColumnDesc);
      auto resVoid = descBuilder.AddColumn(aliasColumnDesc.Unwrap());
      if (!resVoid)
         return R__FORWARD_ERROR(resVoid);
   }
   bytes = frame + frameSize;

   std::uint32_t nTypeInfo;
   frame = bytes;
   result = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize, nTypeInfo);
   if (!result)
      return R__FORWARD_ERROR(result);
   if (nTypeInfo > 0)
      R__LOG_WARNING(NTupleLog()) << "Extra type information is still unsupported! ";
   bytes = frame + frameSize;

   return bytes - base;
}

ROOT::Experimental::Internal::RNTupleSerializer::RContext
ROOT::Experimental::Internal::RNTupleSerializer::SerializeHeader(void *buffer,
                                                                 const ROOT::Experimental::RNTupleDescriptor &desc)
{
   RContext context;

   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void **where = (buffer == nullptr) ? &buffer : reinterpret_cast<void **>(&pos);

   pos += SerializeEnvelopePreamble(kEnvelopeTypeHeader, *where);
   // So far we don't make use of feature flags
   pos += SerializeFeatureFlags(desc.GetFeatureFlags(), *where);
   pos += SerializeString(desc.GetName(), *where);
   pos += SerializeString(desc.GetDescription(), *where);
   pos += SerializeString(std::string("ROOT v") + ROOT_RELEASE, *where);

   context.MapSchema(desc, /*forHeaderExtension=*/false);
   pos += SerializeSchemaDescription(*where, desc, context);

   std::uint64_t size = pos - base;
   std::uint64_t xxhash3 = 0;
   size += SerializeEnvelopePostscript(base, size, xxhash3);

   context.SetHeaderSize(size);
   context.SetHeaderXxHash3(xxhash3);
   return context;
}

std::uint32_t
ROOT::Experimental::Internal::RNTupleSerializer::SerializePageList(void *buffer, const RNTupleDescriptor &desc,
                                                                   std::span<DescriptorId_t> physClusterIDs,
                                                                   const RContext &context)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += SerializeEnvelopePreamble(kEnvelopeTypePageList, *where);

   pos += SerializeUInt64(context.GetHeaderXxHash3(), *where);

   // Cluster summaries
   const auto nClusters = physClusterIDs.size();
   auto clusterSummaryFrame = pos;
   pos += SerializeListFramePreamble(nClusters, *where);
   for (auto clusterId : physClusterIDs) {
      const auto &clusterDesc = desc.GetClusterDescriptor(context.GetMemClusterId(clusterId));
      RClusterSummary summary{clusterDesc.GetFirstEntryIndex(), clusterDesc.GetNEntries(), -1};
      pos += SerializeClusterSummary(summary, *where);
   }
   pos += SerializeFramePostscript(buffer ? clusterSummaryFrame : nullptr, pos - clusterSummaryFrame);

   // Page locations
   auto topMostFrame = pos;
   pos += SerializeListFramePreamble(nClusters, *where);

   for (auto clusterId : physClusterIDs) {
      const auto &clusterDesc = desc.GetClusterDescriptor(context.GetMemClusterId(clusterId));
      // Get an ordered set of physical column ids
      std::set<DescriptorId_t> onDiskColumnIds;
      for (auto column : clusterDesc.GetColumnIds())
         onDiskColumnIds.insert(context.GetOnDiskColumnId(column));

      auto outerFrame = pos;
      pos += SerializeListFramePreamble(onDiskColumnIds.size(), *where);
      for (auto onDiskId : onDiskColumnIds) {
         auto memId = context.GetMemColumnId(onDiskId);
         const auto &columnRange = clusterDesc.GetColumnRange(memId);
         const auto &pageRange = clusterDesc.GetPageRange(memId);

         auto innerFrame = pos;
         pos += SerializeListFramePreamble(pageRange.fPageInfos.size(), *where);

         for (const auto &pi : pageRange.fPageInfos) {
            pos += SerializeUInt32(pi.fNElements, *where);
            pos += SerializeLocator(pi.fLocator, *where);
         }
         pos += SerializeUInt64(columnRange.fFirstElementIndex, *where);
         pos += SerializeUInt32(columnRange.fCompressionSettings, *where);

         pos += SerializeFramePostscript(buffer ? innerFrame : nullptr, pos - innerFrame);
      }
      pos += SerializeFramePostscript(buffer ? outerFrame : nullptr, pos - outerFrame);
   }

   pos += SerializeFramePostscript(buffer ? topMostFrame : nullptr, pos - topMostFrame);
   std::uint64_t size = pos - base;
   size += SerializeEnvelopePostscript(base, size);
   return size;
}

std::uint32_t
ROOT::Experimental::Internal::RNTupleSerializer::SerializeFooter(void *buffer,
                                                                 const ROOT::Experimental::RNTupleDescriptor &desc,
                                                                 const RContext &context)
{
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += SerializeEnvelopePreamble(kEnvelopeTypeFooter, *where);

   // So far we don't make use of footer feature flags
   pos += SerializeFeatureFlags(std::vector<std::uint64_t>(), *where);
   pos += SerializeUInt64(context.GetHeaderXxHash3(), *where);

   // Schema extension, i.e. incremental changes with respect to the header
   auto frame = pos;
   pos += SerializeRecordFramePreamble(*where);
   pos += SerializeSchemaDescription(*where, desc, context, /*forHeaderExtension=*/true);
   pos += SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);

   // So far no support for shared clusters (no column groups)
   frame = pos;
   pos += SerializeListFramePreamble(0, *where);
   pos += SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);

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
      pos += SerializeClusterGroup(clusterGroup, *where);
   }
   pos += SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);

   // So far no support for meta-data
   frame = pos;
   pos += SerializeListFramePreamble(0, *where);
   pos += SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);

   std::uint32_t size = pos - base;
   size += SerializeEnvelopePostscript(base, size);
   return size;
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::Internal::RNTupleSerializer::DeserializeHeader(const void *buffer, std::uint64_t bufSize,
                                                                   RNTupleDescriptorBuilder &descBuilder)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   auto fnBufSizeLeft = [&]() { return bufSize - (bytes - base); };
   RResult<std::uint32_t> result{0};

   std::uint64_t xxhash3{0};
   result = DeserializeEnvelope(bytes, fnBufSizeLeft(), kEnvelopeTypeHeader, xxhash3);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   descBuilder.SetOnDiskHeaderXxHash3(xxhash3);

   std::vector<std::uint64_t> featureFlags;
   result = DeserializeFeatureFlags(bytes, fnBufSizeLeft(), featureFlags);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
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
   result = DeserializeString(bytes, fnBufSizeLeft(), name);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   result = DeserializeString(bytes, fnBufSizeLeft(), description);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   result = DeserializeString(bytes, fnBufSizeLeft(), writer);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   descBuilder.SetNTuple(name, description);

   // Zero field
   descBuilder.AddField(
      RFieldDescriptorBuilder().FieldId(kZeroFieldId).Structure(ENTupleStructure::kRecord).MakeDescriptor().Unwrap());
   result = DeserializeSchemaDescription(bytes, fnBufSizeLeft(), descBuilder);
   if (!result)
      return R__FORWARD_ERROR(result);

   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::Internal::RNTupleSerializer::DeserializeFooter(const void *buffer, std::uint64_t bufSize,
                                                                   RNTupleDescriptorBuilder &descBuilder)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   auto fnBufSizeLeft = [&]() { return bufSize - (bytes - base); };
   RResult<std::uint32_t> result{0};

   result = DeserializeEnvelope(bytes, fnBufSizeLeft(), kEnvelopeTypeFooter);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   std::vector<std::uint64_t> featureFlags;
   result = DeserializeFeatureFlags(bytes, fnBufSizeLeft(), featureFlags);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   for (auto f: featureFlags) {
      if (f)
         R__LOG_WARNING(NTupleLog()) << "Unsupported feature flag! " << f;
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

   result = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   if (fnFrameSizeLeft() > 0) {
      descBuilder.BeginHeaderExtension();
      result = DeserializeSchemaDescription(bytes, fnFrameSizeLeft(), descBuilder);
      if (!result)
         return R__FORWARD_ERROR(result);
   }
   bytes = frame + frameSize;

   std::uint32_t nColumnGroups;
   frame = bytes;
   result = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize, nColumnGroups);
   if (!result)
      return R__FORWARD_ERROR(result);
   if (nColumnGroups > 0)
      return R__FAIL("sharded clusters are still unsupported");
   bytes = frame + frameSize;

   std::uint32_t nClusterGroups;
   frame = bytes;
   result = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize, nClusterGroups);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   for (std::uint32_t groupId = 0; groupId < nClusterGroups; ++groupId) {
      RClusterGroup clusterGroup;
      result = DeserializeClusterGroup(bytes, fnFrameSizeLeft(), clusterGroup);
      if (!result)
         return R__FORWARD_ERROR(result);
      bytes += result.Unwrap();

      descBuilder.AddToOnDiskFooterSize(clusterGroup.fPageListEnvelopeLink.fLocator.fBytesOnStorage);
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

   std::uint32_t nMDBlocks;
   frame = bytes;
   result = DeserializeFrameHeader(bytes, fnBufSizeLeft(), frameSize, nMDBlocks);
   if (!result)
      return R__FORWARD_ERROR(result);
   if (nMDBlocks > 0)
      R__LOG_WARNING(NTupleLog()) << "meta-data blocks are still unsupported";
   bytes = frame + frameSize;

   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::Internal::RNTupleSerializer::DeserializePageList(const void *buffer, std::uint64_t bufSize,
                                                                     DescriptorId_t clusterGroupId,
                                                                     RNTupleDescriptor &desc)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   auto fnBufSizeLeft = [&]() { return bufSize - (bytes - base); };
   RResult<std::uint32_t> result{0};

   result = DeserializeEnvelope(bytes, fnBufSizeLeft(), kEnvelopeTypePageList);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   std::uint64_t xxhash3{0};
   if (fnBufSizeLeft() < static_cast<int>(sizeof(std::uint64_t)))
      return R__FAIL("page list too short");
   bytes += DeserializeUInt64(bytes, xxhash3);
   if (xxhash3 != desc.GetOnDiskHeaderXxHash3())
      return R__FAIL("XxHash-3 mismatch between header and page list");

   std::vector<RClusterDescriptorBuilder> clusterBuilders;
   DescriptorId_t firstClusterId{0};
   for (DescriptorId_t i = 0; i < clusterGroupId; ++i) {
      firstClusterId = firstClusterId + desc.GetClusterGroupDescriptor(i).GetNClusters();
   }

   std::uint64_t clusterSummaryFrameSize;
   auto clusterSummaryFrame = bytes;
   auto fnClusterSummaryFrameSizeLeft = [&]() { return clusterSummaryFrameSize - (bytes - clusterSummaryFrame); };

   std::uint32_t nClusterSummaries;
   result = DeserializeFrameHeader(bytes, fnBufSizeLeft(), clusterSummaryFrameSize, nClusterSummaries);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   for (auto clusterId = firstClusterId; clusterId < firstClusterId + nClusterSummaries; ++clusterId) {
      RClusterSummary clusterSummary;
      result = DeserializeClusterSummary(bytes, fnClusterSummaryFrameSizeLeft(), clusterSummary);
      if (!result)
         return R__FORWARD_ERROR(result);
      bytes += result.Unwrap();
      if (clusterSummary.fColumnGroupID >= 0)
         return R__FAIL("sharded clusters are still unsupported");

      RClusterDescriptorBuilder builder;
      builder.ClusterId(clusterId).FirstEntryIndex(clusterSummary.fFirstEntry).NEntries(clusterSummary.fNEntries);
      clusterBuilders.emplace_back(std::move(builder));
   }
   bytes = clusterSummaryFrame + clusterSummaryFrameSize;

   std::uint64_t topMostFrameSize;
   auto topMostFrame = bytes;
   auto fnTopMostFrameSizeLeft = [&]() { return topMostFrameSize - (bytes - topMostFrame); };

   std::uint32_t nClusters;
   result = DeserializeFrameHeader(bytes, fnBufSizeLeft(), topMostFrameSize, nClusters);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   if (nClusters != nClusterSummaries)
      return R__FAIL("mismatch between number of clusters and number of cluster summaries");

   std::vector<RClusterDescriptor> clusters;
   for (std::uint32_t i = 0; i < nClusters; ++i) {
      std::uint64_t outerFrameSize;
      auto outerFrame = bytes;
      auto fnOuterFrameSizeLeft = [&]() { return outerFrameSize - (bytes - outerFrame); };

      std::uint32_t nColumns;
      result = DeserializeFrameHeader(bytes, fnTopMostFrameSizeLeft(), outerFrameSize, nColumns);
      if (!result)
         return R__FORWARD_ERROR(result);
      bytes += result.Unwrap();

      for (std::uint32_t j = 0; j < nColumns; ++j) {
         std::uint64_t innerFrameSize;
         auto innerFrame = bytes;
         auto fnInnerFrameSizeLeft = [&]() { return innerFrameSize - (bytes - innerFrame); };

         std::uint32_t nPages;
         result = DeserializeFrameHeader(bytes, fnOuterFrameSizeLeft(), innerFrameSize, nPages);
         if (!result)
            return R__FORWARD_ERROR(result);
         bytes += result.Unwrap();

         RClusterDescriptor::RPageRange pageRange;
         pageRange.fPhysicalColumnId = j;
         for (std::uint32_t k = 0; k < nPages; ++k) {
            if (fnInnerFrameSizeLeft() < static_cast<int>(sizeof(std::uint32_t)))
               return R__FAIL("inner frame too short");
            std::int32_t nElements;
            RNTupleLocator locator;
            bytes += DeserializeInt32(bytes, nElements);
            if (nElements < 0) {
               // TODO(jblomer): page with checksum
               nElements = -nElements;
            }
            result = DeserializeLocator(bytes, fnInnerFrameSizeLeft(), locator);
            if (!result)
               return R__FORWARD_ERROR(result);
            pageRange.fPageInfos.push_back({static_cast<std::uint32_t>(nElements), locator});
            bytes += result.Unwrap();
         }

         if (fnInnerFrameSizeLeft() < static_cast<int>(sizeof(std::uint32_t) + sizeof(std::uint64_t)))
            return R__FAIL("page list frame too short");
         std::uint64_t columnOffset;
         bytes += DeserializeUInt64(bytes, columnOffset);
         std::uint32_t compressionSettings;
         bytes += DeserializeUInt32(bytes, compressionSettings);

         clusterBuilders[i].CommitColumnRange(j, columnOffset, compressionSettings, pageRange);
         bytes = innerFrame + innerFrameSize;
      } // loop over columns

      bytes = outerFrame + outerFrameSize;

      clusterBuilders[i].AddDeferredColumnRanges(desc);
      clusters.emplace_back(clusterBuilders[i].MoveDescriptor().Unwrap());
   } // loop over clusters
   desc.AddClusterGroupDetails(clusterGroupId, clusters);

   return RResult<void>::Success();
}
