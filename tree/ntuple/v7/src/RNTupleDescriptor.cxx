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

void ROOT::Experimental::RNTupleDescriptorBuilder::SetNTuple(std::string_view name, const RNTupleVersion &version) {
   fDescriptor.fName = std::string(name);
   fDescriptor.fVersion = version;
}

void ROOT::Experimental::RNTupleDescriptorBuilder::AddField(
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

void ROOT::Experimental::RNTupleDescriptorBuilder::SetFieldParent(DescriptorId_t fieldId, DescriptorId_t parentId)
{
   fDescriptor.fFieldDescriptors[fieldId].fParentId = parentId;
}

void ROOT::Experimental::RNTupleDescriptorBuilder::AddFieldLink(DescriptorId_t fieldId, DescriptorId_t linkId)
{
   fDescriptor.fFieldDescriptors[fieldId].fLinkIds.push_back(linkId);
}

void ROOT::Experimental::RNTupleDescriptorBuilder::AddColumn(
   DescriptorId_t columnId, DescriptorId_t fieldId, const RNTupleVersion &version, const RColumnModel &model)
{
   RColumnDescriptor c;
   c.fColumnId = columnId;
   c.fFieldId = fieldId;
   c.fVersion = version;
   c.fModel = model;
   fDescriptor.fColumnDescriptors[columnId] = c;
}

void ROOT::Experimental::RNTupleDescriptorBuilder::SetColumnOffset(DescriptorId_t columnId, DescriptorId_t offsetId)
{
   fDescriptor.fColumnDescriptors[columnId].fOffsetId = offsetId;
}

void ROOT::Experimental::RNTupleDescriptorBuilder::AddColumnLink(DescriptorId_t columnId, DescriptorId_t linkId)
{
   fDescriptor.fColumnDescriptors[columnId].fLinkIds.push_back(linkId);
}

void ROOT::Experimental::RNTupleDescriptorBuilder::AddCluster(
   DescriptorId_t clusterId, RNTupleVersion version, NTupleSize_t firstEntryIndex, ClusterSize_t nEntries)
{
   RClusterDescriptor c;
   c.fClusterId = clusterId;
   c.fVersion = version;
   c.fFirstEntryIndex = firstEntryIndex;
   c.fNEntries = nEntries;
   fDescriptor.fClusterDescriptors[clusterId] = c;
}

void ROOT::Experimental::RNTupleDescriptorBuilder::AddClusterColumnInfo(
   DescriptorId_t clusterId, const RClusterDescriptor::RColumnInfo &columnInfo)
{
   fDescriptor.fClusterDescriptors[clusterId].fColumnInfos[columnInfo.fColumnId] = columnInfo;
}

