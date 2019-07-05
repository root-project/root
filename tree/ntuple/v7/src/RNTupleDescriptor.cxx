/// \file RNTupleDescriptor.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RStringView.hxx>

#include <TError.h>

#include <cstdint>
#include <cstring>
#include <iostream>

namespace ROOT {
namespace Experimental {

namespace {

std::uint32_t SerializeInt64(std::int64_t val, void* buffer) {
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


std::uint32_t SerializeUInt64(std::uint64_t val, void* buffer) {
   return SerializeInt64(val, buffer);
}


std::uint32_t DeserializeInt64(void* buffer, std::int64_t *val) {
   auto bytes = reinterpret_cast<unsigned char *>(buffer);
   *val = std::int64_t(bytes[0]) + (std::int64_t(bytes[1]) << 8) +
          (std::int64_t(bytes[2]) << 16) + (std::int64_t(bytes[3]) << 24) +
          (std::int64_t(bytes[4]) << 32) + (std::int64_t(bytes[5]) << 40) +
          (std::int64_t(bytes[6]) << 48) + (std::int64_t(bytes[7]) << 56);
   return 8;
}


std::uint32_t DeserializeUInt64(void* buffer, std::uint64_t *val) {
   return DeserializeInt64(buffer, reinterpret_cast<std::int64_t*>(val));
}


std::uint32_t SerializeInt32(std::int32_t val, void* buffer) {
   if (buffer != nullptr) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);
      bytes[0] = (val & 0x000000FF);
      bytes[1] = (val & 0x0000FF00) >> 8;
      bytes[2] = (val & 0x00FF0000) >> 16;
      bytes[3] = (val & 0xFF000000) >> 24;
   }
   return 4;
}


std::uint32_t SerializeUInt32(std::uint32_t val, void* buffer) {
   return SerializeInt32(val, buffer);
}


std::uint32_t DeserializeInt32(void* buffer, std::int32_t *val) {
   auto bytes = reinterpret_cast<unsigned char *>(buffer);
   *val = std::int32_t(bytes[0]) + (std::int32_t(bytes[1]) << 8) +
          (std::int32_t(bytes[2]) << 16) + (std::int32_t(bytes[3]) << 24);
   return 4;
}


std::uint32_t DeserializeUInt32(void* buffer, std::uint32_t *val) {
   return DeserializeInt32(buffer, reinterpret_cast<std::int32_t*>(val));
}


std::uint32_t SerializeString(const std::string &val, void* buffer) {
   if (buffer != nullptr) {
      auto pos = reinterpret_cast<unsigned char *>(buffer);
      pos += SerializeUInt32(val.length(), pos);
      memcpy(pos, val.data(), val.length());
   }
   return SerializeUInt32(val.length(), nullptr) + val.length();
}

std::uint32_t DeserializeString(void* buffer, std::string *val) {
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto bytes = base;
   std::uint32_t length;
   bytes += DeserializeUInt32(buffer, &length);
   val->resize(length);
   memcpy(&(*val)[0], bytes, length);
   return bytes + length - base;
}

std::uint32_t SerializeVersion(const RNTupleVersion &val, void* buffer) {
   if (buffer != nullptr) {
      auto pos = reinterpret_cast<unsigned char *>(buffer);
      pos += SerializeUInt32(val.GetVersionUse(), pos);
      pos += SerializeUInt32(val.GetVersionMin(), pos);
      pos += SerializeUInt64(val.GetFlags(), pos);
   }
   return 16;
}

std::uint32_t DeserializeVersion(void* buffer, RNTupleVersion *version) {
   auto bytes = reinterpret_cast<unsigned char *>(buffer);
   std::uint32_t versionUse;
   std::uint32_t versionMin;
   std::uint64_t flags;
   bytes += DeserializeUInt32(bytes, &versionUse);
   bytes += DeserializeUInt32(bytes, &versionMin);
   bytes += DeserializeUInt64(bytes, &flags);
   *version = RNTupleVersion(versionUse, versionMin, flags);
   return 16;
}

std::uint32_t SerializeField(const RFieldDescriptor &val, void* buffer) {
   auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += SerializeUInt64(val.GetId(), *where);
   pos += SerializeVersion(val.GetFieldVersion(), *where);
   pos += SerializeVersion(val.GetTypeVersion(), *where);
   pos += SerializeString(val.GetFieldName(), *where);
   pos += SerializeString(val.GetFieldDescription(), *where);
   pos += SerializeString(val.GetTypeName(), *where);
   pos += SerializeUInt32(static_cast<int>(val.GetStructure()), *where);
   pos += SerializeUInt64(val.GetParentId(), *where);
   pos += SerializeUInt32(val.GetLinkIds().size(), *where);
   for (const auto& l : val.GetLinkIds())
      pos += SerializeUInt64(l, *where);
   return pos - base;
}

std::uint32_t SerializeColumn(const RColumnDescriptor &val, void* buffer) {
   auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += SerializeUInt64(val.GetId(), *where);
   pos += SerializeVersion(val.GetVersion(), *where);
   pos += SerializeString(val.GetModel().GetName(), *where);
   pos += SerializeInt32(static_cast<int>(val.GetModel().GetType()), *where);
   pos += SerializeInt32(static_cast<int>(val.GetModel().GetIsSorted()), *where);
   pos += SerializeUInt64(val.GetFieldId(), *where);
   pos += SerializeUInt64(val.GetOffsetId(), *where);
   pos += SerializeUInt32(val.GetLinkIds().size(), *where);
   for (const auto& l : val.GetLinkIds())
      pos += SerializeUInt64(l, *where);
   return pos - base;
}

} // anonymous namespace


////////////////////////////////////////////////////////////////////////////////


bool RFieldDescriptor::operator==(const RFieldDescriptor &other) const {
   return fFieldId == other.fFieldId &&
          fFieldVersion == other.fFieldVersion &&
          fTypeVersion == other.fTypeVersion &&
          fFieldName == other.fFieldName &&
          fFieldDescription == other.fFieldDescription &&
          fTypeName == other.fTypeName &&
          fStructure == other.fStructure &&
          fParentId == other.fParentId &&
          fLinkIds == other.fLinkIds;
}


////////////////////////////////////////////////////////////////////////////////


bool RColumnDescriptor::operator==(const RColumnDescriptor &other) const {
   return fColumnId == other.fColumnId &&
          fVersion == other.fVersion &&
          fModel == other.fModel &&
          fFieldId == other.fFieldId &&
          fOffsetId == other.fOffsetId &&
          fLinkIds == other.fLinkIds;
}


////////////////////////////////////////////////////////////////////////////////


bool RClusterDescriptor::operator==(const RClusterDescriptor &other) const {
   return fClusterId == other.fClusterId &&
          fVersion == other.fVersion &&
          fFirstEntryIndex == other.fFirstEntryIndex &&
          fNEntries == other.fNEntries &&
          fColumnRanges == other.fColumnRanges;
}


////////////////////////////////////////////////////////////////////////////////


bool RNTupleDescriptor::operator==(const RNTupleDescriptor &other) const {
   return fName == other.fName &&
          fDescription == other.fDescription &&
          fVersion == other.fVersion &&
          fOwnUuid == other.fOwnUuid &&
          fGroupUuid == other.fGroupUuid &&
          fFieldDescriptors == other.fFieldDescriptors &&
          fColumnDescriptors == other.fColumnDescriptors &&
          fClusterDescriptors == other.fClusterDescriptors;
}


std::uint32_t RNTupleDescriptor::SerializeHeader(void* buffer) const
{
   auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += SerializeUInt32(kByteProtocol, *where);
   void *ptrSize = *where;
   pos += SerializeUInt32(0, *where); // placeholder for the size

   pos += SerializeString(fName, *where);
   pos += SerializeString(fDescription, *where);
   pos += SerializeVersion(fVersion, *where);
   pos += SerializeUInt32(fFieldDescriptors.size(), *where);
   for (const auto& f : fFieldDescriptors) {
      pos += SerializeField(f.second, *where);
   }
   pos += SerializeUInt32(fColumnDescriptors.size(), *where);
   for (const auto& c : fColumnDescriptors) {
      pos += SerializeColumn(c.second, *where);
   }

   pos += SerializeUInt32(0 /* TODO CRC32 */, *where);
   std::uint32_t size = pos - base;
   SerializeUInt32(size, ptrSize);
   return size;
}

std::uint32_t RNTupleDescriptor::SerializeFooter(void* buffer) const
{
   auto base = reinterpret_cast<unsigned char *>((buffer != nullptr) ? buffer : 0);
   auto pos = base;
   void** where = (buffer == nullptr) ? &buffer : reinterpret_cast<void**>(&pos);

   pos += SerializeUInt32(kByteProtocol, *where);
   void *ptrSize = *where;
   pos += SerializeUInt32(0, *where); // placeholder for the size

   // TODO UUid again
   pos += SerializeUInt64(fClusterDescriptors.size(), *where);
   for (const auto& cluster : fClusterDescriptors) {
      pos += SerializeUInt64(cluster.second.GetId(), *where);
      pos += SerializeVersion(cluster.second.GetVersion(), *where);
      pos += SerializeUInt64(cluster.second.GetFirstEntryIndex(), *where);
      pos += SerializeUInt64(cluster.second.GetNEntries(), *where);
      pos += SerializeUInt32(fColumnDescriptors.size(), *where);
      for (const auto& column : fColumnDescriptors) {
         auto range = cluster.second.GetColumnRanges(column.first);
         R__ASSERT(range.fColumnId == column.first);
         pos += SerializeUInt64(range.fColumnId, *where);
         pos += SerializeUInt64(range.fFirstElementIndex, *where);
         pos += SerializeUInt64(range.fNElements, *where);
      }
   }

   pos += SerializeUInt32(0 /* TODO CRC32 */, *where);
   std::uint32_t size = pos - base;
   SerializeUInt32(size, ptrSize);
   return size;
}


////////////////////////////////////////////////////////////////////////////////


void RNTupleDescriptorBuilder::SetFromHeader(void* headerBuffer) {
   auto pos = reinterpret_cast<unsigned char *>(headerBuffer);
   std::uint32_t byteProtocol;
   pos += DeserializeUInt32(pos, &byteProtocol);
   R__ASSERT(byteProtocol == 0);
   std::uint32_t size;
   pos += DeserializeUInt32(pos, &size);
   // TODO: verify crc32

   pos += DeserializeString(pos, &fDescriptor.fName);
   pos += DeserializeString(pos, &fDescriptor.fDescription);
   pos += DeserializeVersion(pos, &fDescriptor.fVersion);
   // TODO
   fDescriptor.fOwnUuid = RNTupleUuid();
   fDescriptor.fGroupUuid = RNTupleUuid();

   std::uint32_t nFields;
   pos += DeserializeUInt32(pos, &nFields);
   for (std::uint32_t i = 0; i < nFields; ++i) {
      RFieldDescriptor f;
      pos += DeserializeUInt64(pos, &f.fFieldId);
      pos += DeserializeVersion(pos, &f.fFieldVersion);
      pos += DeserializeVersion(pos, &f.fTypeVersion);
      pos += DeserializeString(pos, &f.fFieldName);
      pos += DeserializeString(pos, &f.fFieldDescription);
      pos += DeserializeString(pos, &f.fTypeName);
      std::int32_t structure;
      pos += DeserializeInt32(pos, &structure);
      f.fStructure = static_cast<ENTupleStructure>(structure);
      pos += DeserializeUInt64(pos, &f.fParentId);

      std::uint32_t nLinks;
      pos += DeserializeUInt32(pos, &nLinks);
      f.fLinkIds.resize(nLinks);
      for (std::uint32_t j = 0; j < nLinks; ++j) {
         pos += DeserializeUInt64(pos, &f.fLinkIds[j]);
      }

      fDescriptor.fFieldDescriptors[f.fFieldId] = f;
   }

   std::uint32_t nColumns;
   pos += DeserializeUInt32(pos, &nColumns);
   for (std::uint32_t i = 0; i < nColumns; ++i) {
      RColumnDescriptor c;
      pos += DeserializeUInt64(pos, &c.fColumnId);
      pos += DeserializeVersion(pos, &c.fVersion);
      std::string name;
      std::int32_t type;
      std::int32_t isSorted;
      pos += DeserializeString(pos, &name);
      pos += DeserializeInt32(pos, &type);
      pos += DeserializeInt32(pos, &isSorted);
      c.fModel = RColumnModel(name, static_cast<EColumnType>(type), isSorted);
      pos += DeserializeUInt64(pos, &c.fFieldId);
      pos += DeserializeUInt64(pos, &c.fOffsetId);

      std::uint32_t nLinks;
      pos += DeserializeUInt32(pos, &nLinks);
      c.fLinkIds.resize(nLinks);
      for (std::uint32_t j = 0; j < nLinks; ++j) {
         pos += DeserializeUInt64(pos, &c.fLinkIds[j]);
      }

      fDescriptor.fColumnDescriptors[c.fColumnId] = c;
   }
}


void RNTupleDescriptorBuilder::AddClustersFromFooter(void* footerBuffer) {
   auto pos = reinterpret_cast<unsigned char *>(footerBuffer);
   std::uint32_t byteProtocol;
   pos += DeserializeUInt32(pos, &byteProtocol);
   R__ASSERT(byteProtocol == 0);
   std::uint32_t size;
   pos += DeserializeUInt32(pos, &size);
   // TODO: verify crc32

   std::uint64_t nClusters;
   pos += DeserializeUInt64(pos, &nClusters);
   for (std::uint64_t i = 0; i < nClusters; ++i) {
      std::uint64_t clusterId;
      RNTupleVersion version;
      std::uint64_t firstEntry;
      std::uint64_t nEntries;
      pos += DeserializeUInt64(pos, &clusterId);
      pos += DeserializeVersion(pos, &version);
      pos += DeserializeUInt64(pos, &firstEntry);
      pos += DeserializeUInt64(pos, &nEntries);
      AddCluster(clusterId, version, firstEntry, ROOT::Experimental::ClusterSize_t(nEntries));
      std::uint32_t nColumns;
      pos += DeserializeUInt32(pos, &nColumns);
      for (std::uint32_t j = 0; j < nColumns; ++j) {
         RClusterDescriptor::RColumnRange range;
         uint64_t nElements;
         pos += DeserializeUInt64(pos, &range.fColumnId);
         pos += DeserializeUInt64(pos, &range.fFirstElementIndex);
         pos += DeserializeUInt64(pos, &nElements);
         range.fNElements = nElements;
         AddClusterColumnRange(clusterId, range);
      }
   }
}


void RNTupleDescriptorBuilder::SetNTuple(
   const std::string_view &name, const std::string_view &description, const RNTupleVersion &version,
   const RNTupleUuid &uuid)
{
   fDescriptor.fName = std::string(name);
   fDescriptor.fDescription = std::string(description);
   fDescriptor.fVersion = version;
   fDescriptor.fOwnUuid = uuid;
   fDescriptor.fGroupUuid = uuid;
}

void RNTupleDescriptorBuilder::AddField(
   DescriptorId_t fieldId, const RNTupleVersion &fieldVersion, const RNTupleVersion &typeVersion,
   std::string_view fieldName, std::string_view typeName, ENTupleStructure structure)
{
   RFieldDescriptor f;
   f.fFieldId = fieldId;
   f.fFieldVersion = fieldVersion;
   f.fTypeVersion = typeVersion;
   f.fFieldName = std::string(fieldName);
   f.fTypeName = std::string(typeName);
   f.fStructure = structure;
   fDescriptor.fFieldDescriptors[fieldId] = f;
}

void RNTupleDescriptorBuilder::SetFieldParent(DescriptorId_t fieldId, DescriptorId_t parentId)
{
   fDescriptor.fFieldDescriptors[fieldId].fParentId = parentId;
}

void RNTupleDescriptorBuilder::AddFieldLink(DescriptorId_t fieldId, DescriptorId_t linkId)
{
   fDescriptor.fFieldDescriptors[fieldId].fLinkIds.push_back(linkId);
}

void RNTupleDescriptorBuilder::AddColumn(
   DescriptorId_t columnId, DescriptorId_t fieldId, const RNTupleVersion &version, const RColumnModel &model)
{
   RColumnDescriptor c;
   c.fColumnId = columnId;
   c.fFieldId = fieldId;
   c.fVersion = version;
   c.fModel = model;
   fDescriptor.fColumnDescriptors[columnId] = c;
}

void RNTupleDescriptorBuilder::SetColumnOffset(DescriptorId_t columnId, DescriptorId_t offsetId)
{
   fDescriptor.fColumnDescriptors[columnId].fOffsetId = offsetId;
}

void RNTupleDescriptorBuilder::AddColumnLink(DescriptorId_t columnId, DescriptorId_t linkId)
{
   fDescriptor.fColumnDescriptors[columnId].fLinkIds.push_back(linkId);
}

void RNTupleDescriptorBuilder::AddCluster(
   DescriptorId_t clusterId, RNTupleVersion version, NTupleSize_t firstEntryIndex, ClusterSize_t nEntries)
{
   RClusterDescriptor c;
   c.fClusterId = clusterId;
   c.fVersion = version;
   c.fFirstEntryIndex = firstEntryIndex;
   c.fNEntries = nEntries;
   fDescriptor.fClusterDescriptors[clusterId] = c;
}

void RNTupleDescriptorBuilder::AddClusterColumnRange(
   DescriptorId_t clusterId, const RClusterDescriptor::RColumnRange &columnRange)
{
   fDescriptor.fClusterDescriptors[clusterId].fColumnRanges[columnRange.fColumnId] = columnRange;
}

} // namespace Experimental
} // namespace ROOT
