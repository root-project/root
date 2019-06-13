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

#include <cstring>

namespace ROOT {
namespace Experimental {

namespace {

std::uint32_t SerializeInt32(std::int32_t val, void* buffer) {
   if (buffer != nullptr) {
      auto bytes = reinterpret_cast<unsigned char *>(buffer);
      bytes[0] = (val & 0x000000FF);
      bytes[1] = (val & 0x0000FF00) >> 8;
      bytes[2] = (val & 0x00FF0000) >> 16;
      bytes[3] = (val & 0xFF000000) >> 24;
   }
   return sizeof(val);
}


std::uint32_t DeserializeInt32(void* buffer, std::int32_t *val) {
   auto bytes = reinterpret_cast<unsigned char *>(buffer);
   *val = bytes[0] + (bytes[1] << 8) + (bytes[2] << 16) + (bytes[3] << 24);
   return sizeof(*val);
}


std::uint32_t SerializeString(const std::string &val, void* buffer) {
   if (buffer != nullptr) {
      auto pos = reinterpret_cast<unsigned char *>(buffer);
      pos += SerializeInt32(val.length(), pos);
      memcpy(pos, val.data(), val.length());
   }
   return SerializeInt32(val.length(), nullptr) + val.length();
}

std::uint32_t DeserializeString(void* buffer, std::string *val) {
   auto base = reinterpret_cast<unsigned char *>(buffer);
   auto bytes = base;
   std::int32_t length;
   bytes += DeserializeInt32(buffer, &length);
   val->resize(length);
   memcpy(&(*val)[0], bytes, length);
   return bytes - base;
}

} // anonymous namespace


////////////////////////////////////////////////////////////////////////////////


bool RFieldDescriptor::operator==(const RFieldDescriptor &other) const {
   return fFieldId == other.fFieldId &&
          fFieldVersion == other.fFieldVersion &&
          fTypeVersion == other.fTypeVersion &&
          fFieldName == other.fFieldName &&
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
          fVersion == other.fVersion &&
          fOwnUuid == other.fOwnUuid &&
          fGroupUuid == other.fGroupUuid &&
          fFieldDescriptors == other.fFieldDescriptors &&
          fColumnDescriptors == other.fColumnDescriptors &&
          fClusterDescriptors == other.fClusterDescriptors;
}


std::uint32_t RNTupleDescriptor::SerializeHeader(void* buffer)
{
   if (buffer != nullptr) {
      auto pos = reinterpret_cast<unsigned char *>(buffer);
      pos += SerializeInt32(kByteProtocol, pos);
      pos += SerializeString(fName, pos);
   }
   return SerializeInt32(kByteProtocol, nullptr) + SerializeString(fName, nullptr);
}

std::uint32_t RNTupleDescriptor::SerializeFooter(void* /*buffer*/)
{
   return 0;
}

RNTupleDescriptor RNTupleDescriptor::Deserialize(void* headerBuffer, void* /*footer*/)
{
   auto pos = reinterpret_cast<unsigned char *>(headerBuffer);
   std::int32_t byteProtocol;
   pos += DeserializeInt32(pos, &byteProtocol);
   R__ASSERT(byteProtocol == 0);

   RNTupleDescriptorBuilder descBuilder;
   std::string name;
   pos += DeserializeString(pos, &name);
   descBuilder.SetNTuple(name, RNTupleVersion(), Uuid_t());
   return descBuilder.GetDescriptor();
}


////////////////////////////////////////////////////////////////////////////////


void RNTupleDescriptorBuilder::SetNTuple(
   const std::string_view &name, const RNTupleVersion &version, const Uuid_t &uuid)
{
   fDescriptor.fName = std::string(name);
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
