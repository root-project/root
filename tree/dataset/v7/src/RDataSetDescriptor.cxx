/// \file RDataSetDescriptor.cxx
/// \ingroup DataSet ROOT7
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

#include <ROOT/RDataSetDescriptor.hxx>
#include <ROOT/RDataSetUtil.hxx>
#include <ROOT/RStringView.hxx>

void ROOT::Experimental::RDataSetDescriptorBuilder::SetDataSet(std::string_view name, const RForestVersion &version) {
   fDescriptor.fName = name.to_string();
   fDescriptor.fVersion = version;
}

void ROOT::Experimental::RDataSetDescriptorBuilder::AddField(
   DescriptorId_t fieldId, const RForestVersion &fieldVersion, const RForestVersion &typeVersion,
   std::string_view fieldName, std::string_view typeName, EForestStructure structure)
{
   RFieldDescriptor f;
   f.fFieldId = fieldId;
   f.fFieldVersion = fieldVersion;
   f.fTypeVersion = typeVersion;
   f.fFieldName = fieldName.to_string();
   f.fTypeName = typeName.to_string();
   f.fStructure = structure;
   fDescriptor.fFieldDescriptors[fieldId] = f;
}

void ROOT::Experimental::RDataSetDescriptorBuilder::SetFieldParent(DescriptorId_t fieldId, DescriptorId_t parentId)
{
   fDescriptor.fFieldDescriptors[fieldId].fParentId = parentId;
}

void ROOT::Experimental::RDataSetDescriptorBuilder::AddFieldLink(DescriptorId_t fieldId, DescriptorId_t linkId)
{
   fDescriptor.fFieldDescriptors[fieldId].fLinkIds.push_back(linkId);
}

void ROOT::Experimental::RDataSetDescriptorBuilder::AddColumn(
   DescriptorId_t columnId, DescriptorId_t fieldId, const RForestVersion &version, const RColumnModel &model)
{
   RColumnDescriptor c;
   c.fColumnId = columnId;
   c.fFieldId = fieldId;
   c.fVersion = version;
   c.fModel = model;
   fDescriptor.fColumnDescriptors[columnId] = c;
}

void ROOT::Experimental::RDataSetDescriptorBuilder::SetColumnOffset(DescriptorId_t columnId, DescriptorId_t offsetId)
{
   fDescriptor.fColumnDescriptors[columnId].fOffsetId = offsetId;
}

void ROOT::Experimental::RDataSetDescriptorBuilder::AddColumnLink(DescriptorId_t columnId, DescriptorId_t linkId)
{
   fDescriptor.fColumnDescriptors[columnId].fLinkIds.push_back(linkId);
}

void ROOT::Experimental::RDataSetDescriptorBuilder::AddCluster(
   DescriptorId_t clusterId, RForestVersion version, ForestSize_t firstEntryIndex, ClusterSize_t nEntries)
{
   RClusterDescriptor c;
   c.fClusterId = clusterId;
   c.fVersion = version;
   c.fFirstEntryIndex = firstEntryIndex;
   c.fNEntries = nEntries;
   fDescriptor.fClusterDescriptors[clusterId] = c;
}

void ROOT::Experimental::RDataSetDescriptorBuilder::AddClusterColumnInfo(
   DescriptorId_t clusterId, const RClusterDescriptor::RColumnInfo &columnInfo)
{
   fDescriptor.fClusterDescriptors[clusterId].fColumnInfos[columnInfo.fColumnId] = columnInfo;
}

