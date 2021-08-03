/// \file RNTupleSerialize.cxx
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

#include <ROOT/RColumnElement.hxx>
#include <ROOT/RColumnModel.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleSerialize.hxx>

#include <RZip.h> // for R__crc32

#include <cstring> // for memcpy
#include <deque>
#include <set>

template <typename T>
using RResult = ROOT::Experimental::RResult<T>;


namespace {

std::uint32_t SerializeFieldsV1(
   const ROOT::Experimental::RNTupleDescriptor &desc,
   ROOT::Experimental::Internal::RNTupleSerializer::RContext &context,
   void *buffer)
{
   using RNTupleSerializer = ROOT::Experimental::Internal::RNTupleSerializer;

   std::deque<ROOT::Experimental::DescriptorId_t> idQueue{desc.GetFieldZeroId()};
   std::uint32_t size = 0;

   while (!idQueue.empty()) {
      auto parentId = idQueue.front();
      idQueue.pop_front();
      auto physParentId = context.MapFieldId(parentId);

      for (const auto &f : desc.GetFieldIterable(parentId)) {
         auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
         auto pos = base;
         void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

         pos += RNTupleSerializer::SerializeRecordFramePreamble(*where);

         pos += RNTupleSerializer::SerializeUInt32(f.GetFieldVersion().GetVersionUse(), *where);
         pos += RNTupleSerializer::SerializeUInt32(f.GetTypeVersion().GetVersionUse(), *where);
         pos += RNTupleSerializer::SerializeUInt32(physParentId, *where);
         pos += RNTupleSerializer::SerializeFieldStructure(f.GetStructure(), *where);
         if (f.GetNRepetitions() > 0) {
            pos += RNTupleSerializer::SerializeUInt16(RNTupleSerializer::kFlagRepetitiveField, *where);
            pos += RNTupleSerializer::SerializeUInt64(f.GetNRepetitions(), *where);
         } else {
            pos += RNTupleSerializer::SerializeUInt16(0, *where);
         }
         pos += RNTupleSerializer::SerializeString(f.GetFieldName(), *where);
         pos += RNTupleSerializer::SerializeString(f.GetTypeName(), *where);
         pos += RNTupleSerializer::SerializeString(""
          /* type alias */, *where);
         pos += RNTupleSerializer::SerializeString(f.GetFieldDescription(), *where);

         size += pos - base;
         pos += RNTupleSerializer::SerializeFramePostscript(base, pos - base);

         idQueue.push_back(f.GetId());
      }
   }

   return size;
}

RResult<std::uint32_t> DeserializeFieldV1(
   const void *buffer,
   std::uint32_t bufSize,
   ROOT::Experimental::RDanglingFieldDescriptor &fieldDesc)
{
   using RNTupleSerializer = ROOT::Experimental::Internal::RNTupleSerializer;

   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint32_t frameSize;
   auto fnFrameSize = [&]() { return frameSize - static_cast<std::uint32_t>(bytes - base); };
   auto result = RNTupleSerializer::DeserializeFrame(bytes, bufSize, frameSize);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   std::uint32_t fieldVersion;
   std::uint32_t typeVersion;
   std::uint32_t parentId;
   ROOT::Experimental::ENTupleStructure structure;
   std::uint16_t flags;
   if (fnFrameSize() < 3 * sizeof(std::uint32_t) +
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
   fieldDesc.FieldVersion(ROOT::Experimental::RNTupleVersion(fieldVersion, fieldVersion))
            .TypeVersion(ROOT::Experimental::RNTupleVersion(typeVersion, typeVersion))
            .ParentId(parentId)
            .Structure(structure);

   if (flags & RNTupleSerializer::kFlagRepetitiveField) {
      if (fnFrameSize() < sizeof(std::uint64_t))
         return R__FAIL("field record frame too short");
      std::uint64_t nRepetitions;
      bytes += RNTupleSerializer::DeserializeUInt64(bytes, nRepetitions);
      fieldDesc.NRepetitions(nRepetitions);
   }

   std::string fieldName;
   std::string typeName;
   std::string aliasName; // so far unused
   std::string description;
   result = RNTupleSerializer::DeserializeString(bytes, fnFrameSize(), fieldName).Unwrap();
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   result = RNTupleSerializer::DeserializeString(bytes, fnFrameSize(), typeName).Unwrap();
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   result = RNTupleSerializer::DeserializeString(bytes, fnFrameSize(), aliasName).Unwrap();
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   result = RNTupleSerializer::DeserializeString(bytes, fnFrameSize(), description).Unwrap();
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   fieldDesc.FieldName(fieldName).TypeName(typeName).FieldDescription(description);

   return frameSize;
}

std::uint32_t SerializeColumnsV1(
   const ROOT::Experimental::RNTupleDescriptor &desc,
   ROOT::Experimental::Internal::RNTupleSerializer::RContext &context,
   void *buffer)
{
   using RNTupleSerializer = ROOT::Experimental::Internal::RNTupleSerializer;
   using RColumnElementBase = ROOT::Experimental::Detail::RColumnElementBase;

   std::deque<ROOT::Experimental::DescriptorId_t> idQueue{desc.GetFieldZeroId()};
   std::uint32_t size = 0;

   while (!idQueue.empty()) {
      auto parentId = idQueue.front();
      idQueue.pop_front();

      for (const auto &c : desc.GetColumnIterable(parentId)) {
         auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
         auto pos = base;
         void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

         pos += RNTupleSerializer::SerializeRecordFramePreamble(*where);

         auto type = c.GetModel().GetType();
         pos += RNTupleSerializer::SerializeColumnType(type, *where);
         pos += RNTupleSerializer::SerializeUInt16(RColumnElementBase::GetBitsOnStorage(type), *where);
         pos += RNTupleSerializer::SerializeUInt32(context.GetPhysColumnId(c.GetFieldId()), *where);
         std::uint32_t flags = 0;
         // TODO(jblomer): add support for descending columns in the column model
         if (c.GetModel().GetIsSorted())
            flags |= RNTupleSerializer::kFlagSortAscColumn;
         // TODO(jblomer): fix for unsigned integer types
         if (type == ROOT::Experimental::EColumnType::kIndex)
            flags |= RNTupleSerializer::kFlagNonNegativeColumn;
         pos += RNTupleSerializer::SerializeUInt32(flags, *where);

         size += pos - base;
         pos += RNTupleSerializer::SerializeFramePostscript(base, size);

         context.MapColumnId(c.GetId());
      }

      for (const auto &f : desc.GetFieldIterable(parentId))
         idQueue.push_back(f.GetId());
   }

   return size;
}

std::uint32_t DeserializeColumnV1(
   const void *buffer,
   std::uint32_t bufSize)
{
   using RNTupleSerializer = ROOT::Experimental::Internal::RNTupleSerializer;

   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint32_t frameSize;
   auto fnFrameSize = [&]() { return frameSize - static_cast<std::uint32_t>(bytes - base); };
   bytes += RNTupleSerializer::DeserializeFrame(bytes, bufSize, frameSize).Unwrap();

   ROOT::Experimental::EColumnType type;
   std::uint16_t bitsOnStorage;
   std::uint32_t fieldId;
   std::uint32_t flags;
   if (fnFrameSize() < RNTupleSerializer::SerializeColumnType(type, nullptr) +
                       sizeof(std::uint16_t) + 2 * sizeof(std::uint32_t))
   {
      throw ROOT::Experimental::RException(R__FAIL("column record frame too short"));
   }
   bytes += RNTupleSerializer::DeserializeColumnType(bytes, type).Unwrap();
   bytes += RNTupleSerializer::DeserializeUInt16(bytes, bitsOnStorage);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, fieldId);
   bytes += RNTupleSerializer::DeserializeUInt32(bytes, flags);

   return frameSize;
}

} // anonymous namespace


std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeCRC32(
   const unsigned char *data, std::uint32_t length, std::uint32_t &crc32, void *buffer)
{
   if (buffer != nullptr) {
      crc32 = R__crc32(0, nullptr, 0);
      crc32 = R__crc32(crc32, data, length);
      SerializeUInt32(crc32, buffer);
   }
   return 4;
}

RResult<void> ROOT::Experimental::Internal::RNTupleSerializer::VerifyCRC32(
   const unsigned char *data, std::uint32_t length, std::uint32_t &crc32)
{
   auto checksumReal = R__crc32(0, nullptr, 0);
   checksumReal = R__crc32(checksumReal, data, length);
   DeserializeUInt32(data + length, crc32);
   if (crc32 != checksumReal)
      return R__FAIL("CRC32 checksum mismatch");
   return RResult<void>::Success();
}


RResult<void> ROOT::Experimental::Internal::RNTupleSerializer::VerifyCRC32(
   const unsigned char *data, std::uint32_t length)
{
   std::uint32_t crc32;
   return R__FORWARD_RESULT(VerifyCRC32(data, length, crc32));
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

RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeString(
   const void *buffer, std::uint32_t bufSize, std::string &val)
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
      case EColumnType::kIndex:
         return SerializeUInt16(0x02, buffer);
      case EColumnType::kSwitch:
         return SerializeUInt16(0x03, buffer);
      case EColumnType::kByte:
         return SerializeUInt16(0x0D, buffer);
      case EColumnType::kBit:
         return SerializeUInt16(0x06, buffer);
      case EColumnType::kReal64:
         return SerializeUInt16(0x07, buffer);
      case EColumnType::kReal32:
         return SerializeUInt16(0x08, buffer);
      case EColumnType::kReal16:
         return SerializeUInt16(0x09, buffer);
      case EColumnType::kInt64:
         return SerializeUInt16(0x0A, buffer);
      case EColumnType::kInt32:
         return SerializeUInt16(0x0B, buffer);
      case EColumnType::kInt16:
         return SerializeUInt16(0x0C, buffer);
      default:
         throw RException(R__FAIL("ROOT bug: unexpected column type"));
   }
}


RResult<std::uint16_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeColumnType(
   const void *buffer, ROOT::Experimental::EColumnType &type)
{
   using EColumnType = ROOT::Experimental::EColumnType;
   std::uint16_t onDiskType;
   auto result = DeserializeUInt16(buffer, onDiskType);
   switch (onDiskType) {
      case 0x02:
         type = EColumnType::kIndex;
         break;
      case 0x03:
         type = EColumnType::kSwitch;
         break;
      case 0x06:
         type = EColumnType::kBit;
         break;
      case 0x07:
         type = EColumnType::kReal64;
         break;
      case 0x08:
         type = EColumnType::kReal32;
         break;
      case 0x09:
         type = EColumnType::kReal16;
         break;
      case 0x0A:
         type = EColumnType::kInt64;
         break;
      case 0x0B:
         type = EColumnType::kInt32;
         break;
      case 0x0C:
         type = EColumnType::kInt16;
         break;
      case 0x0D:
         type = EColumnType::kByte;
         break;
      default:
         return R__FAIL("unexpected on-disk column type");
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


/// Currently all enevelopes have the same version number (1). At a later point, different envelope types
/// may have different version numbers
std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeEnvelopePreamble(void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += SerializeUInt16(kEnvelopeCurrentVersion, *where);
   pos += SerializeUInt16(kEnvelopeMinVersion, *where);
   return pos - base;
}


std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeEnvelopePostscript(
   const unsigned char *envelope, std::uint32_t size, std::uint32_t &crc32, void *buffer)
{
   return SerializeCRC32(envelope, size, crc32, buffer);
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeEnvelopePostscript(
   const unsigned char *envelope, std::uint32_t size, void *buffer)
{
   std::uint32_t crc32;
   return SerializeEnvelopePostscript(envelope, size, crc32, buffer);
}

/// Currently all enevelopes have the same version number (1). At a later point, different envelope types
/// may have different version numbers
RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeEnvelope(
   const void *buffer, std::uint32_t bufSize, std::uint32_t &crc32)
{
   if (bufSize < (2 * sizeof(std::uint16_t) + sizeof(std::uint32_t)))
      return R__FAIL("invalid envelope, too short");

   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   auto result = VerifyCRC32(bytes, bufSize - 4, crc32);
   if (!result)
      return R__FORWARD_ERROR(result);

   std::uint16_t protocolVersionAtWrite;
   std::uint16_t protocolVersionMinRequired;
   bytes += DeserializeUInt16(bytes, protocolVersionAtWrite);
   // RNTuple compatible back to version 1 (but not to version 0)
   if (protocolVersionAtWrite < 1)
      return R__FAIL("The RNTuple format is too old (version 0)");

   bytes += DeserializeUInt16(bytes, protocolVersionMinRequired);
   if (protocolVersionMinRequired > kEnvelopeCurrentVersion) {
      return R__FAIL(std::string("The RNTuple format is too new (version ") +
                                 std::to_string(protocolVersionMinRequired) + ")");
   }

   return sizeof(protocolVersionAtWrite) + sizeof(protocolVersionMinRequired);
}


RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeEnvelope(
   const void *buffer, std::uint32_t bufSize)
{
   std::uint32_t crc32;
   return R__FORWARD_RESULT(DeserializeEnvelope(buffer, bufSize, crc32));
}


std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeRecordFramePreamble(void *buffer)
{
   // Marker: multiply the final size with 1
   return SerializeInt32(1, buffer);
}


std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeListFramePreamble(
   std::uint32_t nitems, void *buffer)
{
   if (nitems >= (1 << 28))
      throw RException(R__FAIL("list frame too large: " + std::to_string(nitems)));

   auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   // Marker: multiply the final size with -1
   pos += RNTupleSerializer::SerializeInt32(-1, *where);
   pos += SerializeUInt32(nitems, *where);
   return pos - base;
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeFramePostscript(
   void *frame, std::int32_t size)
{
   if (size < 0)
      throw RException(R__FAIL("frame too large: " + std::to_string(size)));
   if (size < static_cast<std::int32_t>(sizeof(std::int32_t)))
      throw RException(R__FAIL("frame too short: " + std::to_string(size)));
   if (frame) {
      std::int32_t marker;
      DeserializeInt32(frame, marker);
      if ((marker < 0) && (size < static_cast<std::int32_t>(2 * sizeof(std::int32_t))))
         throw RException(R__FAIL("frame too short: " + std::to_string(size)));

      SerializeInt32(marker * size, frame);
   }
   return 0;
}

ROOT::Experimental::RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeFrame(
   const void *buffer, std::uint32_t bufSize, std::uint32_t &frameSize, std::uint32_t &nitems)
{
   if (bufSize < sizeof(std::int32_t))
      return R__FAIL("frame too short");

   std::int32_t *ssize = reinterpret_cast<std::int32_t *>(&frameSize);
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   bytes += DeserializeInt32(bytes, *ssize);
   if (*ssize >= 0) {
      // Record frame
      nitems = 1;
      if (frameSize < sizeof(std::int32_t))
         return R__FAIL("corrupt frame size");
   } else {
      // List frame
      if (bufSize < 2 * sizeof(std::int32_t))
         return R__FAIL("frame too short");
      bytes += DeserializeUInt32(bytes, nitems);
      nitems &= (2 << 28) - 1;
      *ssize = -(*ssize);
      if (frameSize < 2 * sizeof(std::int32_t))
         return R__FAIL("corrupt frame size");
   }

   if (bufSize < frameSize)
      return R__FAIL("frame too short");

   return bytes - reinterpret_cast<const unsigned char *>(buffer);
}

ROOT::Experimental::RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeFrame(
   const void *buffer, std::uint32_t bufSize, std::uint32_t &frameSize)
{
   std::uint32_t nitems;
   return R__FORWARD_RESULT(DeserializeFrame(buffer, bufSize, frameSize, nitems));
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeFeatureFlags(
   const std::vector<std::int64_t> &flags, void *buffer)
{
   if (flags.empty())
      return SerializeInt64(0, buffer);

   if (buffer) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);

      for (unsigned i = 0; i < flags.size(); ++i) {
         if (flags[i] < 0)
            throw RException(R__FAIL("feature flag out of bounds"));

         if (i == (flags.size() - 1))
            SerializeInt64(flags[i], bytes);
         else
            bytes += SerializeInt64(-flags[i], bytes);
      }
   }
   return (flags.size() * sizeof(std::int64_t));
}

RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeFeatureFlags(
   const void *buffer, std::uint32_t bufSize, std::vector<std::int64_t> &flags)
{
   auto bytes = reinterpret_cast<const unsigned char *>(buffer);

   flags.clear();
   std::int64_t f;
   do {
      if (bufSize < sizeof(std::int64_t))
         return R__FAIL("feature flag buffer too short");
      bytes += DeserializeInt64(bytes, f);
      bufSize -= sizeof(std::int64_t);
      flags.emplace_back(abs(f));
   } while (f < 0);

   return (flags.size() * sizeof(std::int64_t));
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeLocator(
   const RNTupleLocator &locator, void *buffer)
{
   std::uint32_t size = 0;
   if (!locator.fUrl.empty()) {
      if (locator.fUrl.length() >= (1 << 24))
         throw RException(R__FAIL("locator too large"));
      std::int32_t head = locator.fUrl.length();
      head |= 0x02 << 24;
      head = -head;
      size += SerializeInt32(head, buffer);
      if (buffer)
         memcpy(reinterpret_cast<unsigned char *>(buffer) + size, locator.fUrl.data(), locator.fUrl.length());
      size += locator.fUrl.length();
      return size;
   }

   if (static_cast<std::int32_t>(locator.fBytesOnStorage) < 0)
      throw RException(R__FAIL("locator too large"));
   size += SerializeUInt32(locator.fBytesOnStorage, buffer);
   size += SerializeUInt64(locator.fPosition, buffer ? reinterpret_cast<unsigned char *>(buffer) + size : nullptr);
   return size;
}

RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeLocator(
   const void *buffer, std::uint32_t bufSize, RNTupleLocator &locator)
{
   if (bufSize < sizeof(std::int32_t))
      return R__FAIL("too short locator");

   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   std::int32_t head;

   bytes += DeserializeInt32(bytes, head);
   bufSize -= sizeof(std::int32_t);
   if (head < 0) {
      head = -head;
      int type = head >> 24;
      if (type != 0x02)
         return R__FAIL("unsupported locator type: " + std::to_string(type));
      std::uint32_t locatorSize = static_cast<std::uint32_t>(head) & 0x00FFFFFF;
      if (bufSize < locatorSize)
         return R__FAIL("too short locator");
      locator.fBytesOnStorage = 0;
      locator.fPosition = 0;
      locator.fUrl.resize(locatorSize);
      memcpy(&locator.fUrl[0], bytes, locatorSize);
      bytes += locatorSize;
   } else {
      if (bufSize < sizeof(std::uint64_t))
         return R__FAIL("too short locator");
      std::uint64_t offset;
      bytes += DeserializeUInt64(bytes, offset);
      locator.fUrl.clear();
      locator.fBytesOnStorage = head;
      locator.fPosition = offset;
   }

   return bytes - reinterpret_cast<const unsigned char *>(buffer);
}

std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeEnvelopeLink(
   const REnvelopeLink &envelopeLink, void *buffer)
{
   auto size = SerializeUInt32(envelopeLink.fUnzippedSize, buffer);
   size += SerializeLocator(envelopeLink.fLocator,
                            buffer ? reinterpret_cast<unsigned char *>(buffer) + size : nullptr);
   return size;
}

RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeEnvelopeLink(
   const void *buffer, std::uint32_t bufSize, REnvelopeLink &envelopeLink)
{
   if (bufSize < sizeof(std::int32_t))
      return R__FAIL("too short envelope link");

   auto bytes = reinterpret_cast<const unsigned char *>(buffer);
   bytes += DeserializeUInt32(bytes, envelopeLink.fUnzippedSize);
   bufSize -= sizeof(std::uint32_t);
   auto result = DeserializeLocator(bytes, bufSize, envelopeLink.fLocator);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   return bytes - reinterpret_cast<const unsigned char *>(buffer);
}


std::uint32_t ROOT::Experimental::Internal::RNTupleSerializer::SerializeClusterSummary(
   const RClusterSummary &clusterSummary, void *buffer)
{
   auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
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


RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeClusterSummary(
   const void *buffer, std::uint32_t bufSize, RClusterSummary &clusterSummary)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   std::uint32_t frameSize;
   auto result = DeserializeFrame(bytes, bufSize, frameSize);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   auto fnBufSize = [&]() { return frameSize - static_cast<std::uint32_t>(bytes - base); };
   if (fnBufSize() < 2 * sizeof(std::uint64_t))
      return R__FAIL("too short cluster summary");

   bytes += DeserializeUInt64(bytes, clusterSummary.fFirstEntry);
   std::int64_t nEntries;
   bytes += DeserializeInt64(bytes, nEntries);

   if (nEntries < 0) {
      if (fnBufSize() < sizeof(std::uint32_t))
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
   auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   auto frame = pos;
   pos += SerializeRecordFramePreamble(*where);
   pos += SerializeUInt32(clusterGroup.fNClusters, *where);
   pos += SerializeEnvelopeLink(clusterGroup.fPageListEnvelopeLink, *where);
   auto size = pos - frame;
   pos += SerializeFramePostscript(frame, size);
   return size;
}


RResult<std::uint32_t> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeClusterGroup(
   const void *buffer, std::uint32_t bufSize, RClusterGroup &clusterGroup)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;

   std::uint32_t frameSize;
   auto result = DeserializeFrame(bytes, bufSize, frameSize);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();

   auto fnFrameSize = [&]() { return frameSize - static_cast<std::uint32_t>(bytes - base); };
   if (fnFrameSize() < sizeof(std::uint32_t))
      return R__FAIL("too short cluster group");

   bytes += DeserializeUInt32(bytes, clusterGroup.fNClusters);
   result = DeserializeEnvelopeLink(bytes, fnFrameSize(), clusterGroup.fPageListEnvelopeLink);
   if (!result)
      return R__FORWARD_ERROR(result);

   return frameSize;
}


ROOT::Experimental::Internal::RNTupleSerializer::RContext
ROOT::Experimental::Internal::RNTupleSerializer::SerializeHeaderV1(
   const ROOT::Experimental::RNTupleDescriptor &desc, void *buffer)
{
   RContext context;

   auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += SerializeEnvelopePreamble(*where);
   // So far we don't make use of feature flags
   pos += SerializeFeatureFlags(std::vector<std::int64_t>(), *where);
   pos += SerializeString(desc.GetName(), *where);
   pos += SerializeString(desc.GetDescription(), *where);

   auto frame = pos;
   R__ASSERT(desc.GetNFields() > 0); // we must have a zero field, which we don't serialize
   pos += SerializeListFramePreamble(desc.GetNFields() - 1, *where);
   pos += SerializeFieldsV1(desc, context, *where);
   pos += SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);

   frame = pos;
   pos += SerializeListFramePreamble(desc.GetNColumns(), *where);
   pos += SerializeColumnsV1(desc, context, *where);
   pos += SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);

   // We don't use alias columns yet
   frame = pos;
   pos += SerializeListFramePreamble(0, *where);
   pos += SerializeFramePostscript(buffer ? frame : nullptr, pos - frame);

   std::uint32_t size = pos - base;
   std::uint32_t crc32 = 0;
   size += SerializeEnvelopePostscript(base, size, crc32, *where);

   context.SetHeaderSize(size);
   context.SetHeaderCRC32(crc32);
   return context;
}

void ROOT::Experimental::Internal::RNTupleSerializer::SerializePageListV1(
   void *buffer, const RNTupleDescriptor &desc, std::span<DescriptorId_t> physClusterIDs, const RContext &context)
{
   auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += SerializeEnvelopePreamble(*where);
   auto topMostFrame = pos;
   pos += SerializeListFramePreamble(physClusterIDs.size(), *where);

   for (auto clusterId : physClusterIDs) {
      const auto &clusterDesc = desc.GetClusterDescriptor(context.GetMemClusterId(clusterId));
      // Get an ordered set of physical column ids
      std::set<DescriptorId_t> physColumnIds;
      for (auto column : clusterDesc.GetColumnIds())
         physColumnIds.insert(context.GetPhysClusterId(column));

      auto outerFrame = pos;
      pos += SerializeListFramePreamble(physColumnIds.size(), *where);
      for (auto physId : physColumnIds) {
         auto memId = context.GetMemClusterId(physId);

         auto innerFrame = pos;
         const auto &pageRange = clusterDesc.GetPageRange(memId);
         pos += SerializeListFramePreamble(pageRange.fPageInfos.size(), *where);
         for (const auto &pi : pageRange.fPageInfos) {
            pos += SerializeUInt32(pi.fNElements, *where);
            pos += SerializeLocator(pi.fLocator, *where);
         }
         pos += SerializeFramePostscript(innerFrame, pos - innerFrame);
      }
      pos += SerializeFramePostscript(outerFrame, pos - outerFrame);
   }

   pos += SerializeFramePostscript(topMostFrame, pos - topMostFrame);
   std::uint32_t size = pos - base;
   pos += SerializeEnvelopePostscript(base, size, *where);
}

void ROOT::Experimental::Internal::RNTupleSerializer::SerializeClusterV1(
   void *buffer, const ROOT::Experimental::RClusterDescriptor &cluster, const RContext &context)
{
   auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += SerializeEnvelopePreamble(*where);
   auto frame = pos;
   pos += SerializeListFramePreamble(0, *where);

   // Get an ordered set of physical column ids
   std::set<DescriptorId_t> physColumnIds;
   for (auto id : cluster.GetColumnIds())
      physColumnIds.insert(context.GetPhysClusterId(id));
   for (auto physId : physColumnIds) {
      auto id = context.GetMemClusterId(physId);

      auto innerFrame = pos;
      pos += SerializeListFramePreamble(0, *where);
      for (const auto &pi : cluster.GetPageRange(id).fPageInfos) {
         pos += SerializeUInt32(pi.fNElements, *where);
         pos += SerializeLocator(pi.fLocator, *where);
      }
      pos += SerializeFramePostscript(innerFrame, pos - innerFrame);
   }

   pos += SerializeFramePostscript(frame, pos - frame);
   std::uint32_t size = pos - base;
   pos += SerializeEnvelopePostscript(base, size, *where);
}


void ROOT::Experimental::Internal::RNTupleSerializer::SerializeFooterV1(
   void *buffer, const ROOT::Experimental::RNTupleDescriptor &desc, const RContext &context)
{
   auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += SerializeEnvelopePreamble(*where);

   // So far we don't make use of feature flags
   pos += SerializeFeatureFlags(std::vector<std::int64_t>(), *where);
   pos += SerializeUInt32(context.GetHeaderCRC32(), *where);

   // So far no support for extension headers
   auto frame = pos;
   pos += SerializeListFramePreamble(0, *where);
   pos += SerializeFramePostscript(frame, pos - frame);

   // So far no support for shared clusters (no column groups)
   frame = pos;
   pos += SerializeListFramePreamble(0, *where);
   pos += SerializeFramePostscript(frame, pos - frame);

   // Cluster summaries
   const auto nClusters = desc.GetNClusters();
   frame = pos;
   pos += SerializeListFramePreamble(nClusters, *where);
   for (unsigned int i = 0; i < nClusters; ++i) {
      const auto &clusterDesc = desc.GetClusterDescriptor(context.GetMemClusterId(i));
      RClusterSummary summary{clusterDesc.GetFirstEntryIndex(), clusterDesc.GetNEntries(), -1};
      pos += SerializeClusterSummary(summary, *where);
   }
   pos += SerializeFramePostscript(frame, pos - frame);

   // Cluster groups
   const auto &clusterGroups = context.GetClusterGroups();
   const auto nClusterGroups = clusterGroups.size();
   frame = pos;
   pos += SerializeListFramePreamble(nClusterGroups, *where);
   for (unsigned int i = 0; i < nClusterGroups; ++i) {
      pos += SerializeClusterGroup(clusterGroups[i], *where);
   }
   pos += SerializeFramePostscript(frame, pos - frame);

   // So far no support for meta-data
   frame = pos;
   pos += SerializeListFramePreamble(0, *where);
   pos += SerializeFramePostscript(frame, pos - frame);

   std::uint32_t size = pos - base;
   pos += SerializeEnvelopePostscript(base, size, *where);
}

ROOT::Experimental::RResult<void> ROOT::Experimental::Internal::RNTupleSerializer::DeserializeHeaderV1(
  const void *buffer, std::uint32_t bufSize, RNTupleDescriptorBuilder &descBuilder)
{
   auto base = reinterpret_cast<const unsigned char *>(buffer);
   auto bytes = base;
   auto fnBufSize = [&]() { return bufSize - (bytes - base); };
   RResult<std::uint32_t> result{0};

   std::uint32_t crc32{0};
   result = DeserializeEnvelope(bytes, fnBufSize(), crc32);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   descBuilder.SetHeaderCRC32(crc32);

   std::vector<std::int64_t> featureFlags;
   result = DeserializeFeatureFlags(bytes, fnBufSize(), featureFlags);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   for (auto f: featureFlags) {
      if (f)
         R__LOG_WARNING(NTupleLog()) << "Unsupported feature flag! " << f;
   }

   std::string name;
   std::string description;
   result = DeserializeString(bytes, fnBufSize(), name);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   result = DeserializeString(bytes, fnBufSize(), description);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   descBuilder.SetNTuple(name, description, "", RNTupleVersion(), RNTupleUuid());

   std::uint32_t frameSize;
   auto frame = bytes;
   auto fnFrameSize = [&]() { return frameSize - (bytes - frame); };

   std::uint32_t nFields;
   result = DeserializeFrame(bytes, fnBufSize(), frameSize, nFields);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   descBuilder.AddField(RDanglingFieldDescriptor().FieldId(0)
                                                  .ParentId(0)
                                                  .Structure(ENTupleStructure::kRecord)
                                                  .MakeDescriptor().Unwrap());
   for (std::uint32_t i = 0, fieldId = 1; i < nFields; ++i, ++fieldId) {
      RDanglingFieldDescriptor fieldBuilder;
      result = DeserializeFieldV1(bytes, fnFrameSize(), fieldBuilder);
      if (!result)
         return R__FORWARD_ERROR(result);
      bytes += result.Unwrap();
      fieldBuilder.FieldId(fieldId);
      auto fieldDesc = fieldBuilder.MakeDescriptor();
      if (!fieldDesc)
         return R__FORWARD_ERROR(fieldDesc);
      descBuilder.AddField(fieldDesc.Inspect());
      if (fieldDesc.Inspect().GetParentId() != 0)
         descBuilder.AddFieldLink(fieldDesc.Inspect().GetParentId(), fieldDesc.Inspect().GetId());
   }
   bytes = frame + frameSize;

   std::uint32_t nColumns;
   frame = bytes;
   result = DeserializeFrame(bytes, fnBufSize(), frameSize, nColumns);
   if (!result)
      return R__FORWARD_ERROR(result);
   bytes += result.Unwrap();
   std::uint32_t columnId = 0;
   for (std::uint32_t i = 0; i < nColumns; ++i) {
      result = DeserializeColumnV1(bytes, fnFrameSize());
      if (!result)
         return R__FORWARD_ERROR(result);
      bytes += result.Unwrap();
      columnId++;
   }
   bytes = frame + frameSize;

   return RResult<void>::Success();
}
