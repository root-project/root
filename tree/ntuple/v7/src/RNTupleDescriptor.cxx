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

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RStringView.hxx>

#include <RZip.h>
#include <TError.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <iostream>
#include <set>
#include <utility>


bool ROOT::Experimental::RFieldDescriptor::operator==(const RFieldDescriptor &other) const
{
   return fFieldId == other.fFieldId &&
          fFieldVersion == other.fFieldVersion &&
          fTypeVersion == other.fTypeVersion &&
          fFieldName == other.fFieldName &&
          fFieldDescription == other.fFieldDescription &&
          fTypeName == other.fTypeName &&
          fNRepetitions == other.fNRepetitions &&
          fStructure == other.fStructure &&
          fParentId == other.fParentId &&
          fLinkIds == other.fLinkIds;
}

ROOT::Experimental::RFieldDescriptor
ROOT::Experimental::RFieldDescriptor::Clone() const
{
   RFieldDescriptor clone;
   clone.fFieldId = fFieldId;
   clone.fFieldVersion = fFieldVersion;
   clone.fTypeVersion = fTypeVersion;
   clone.fFieldName = fFieldName;
   clone.fFieldDescription = fFieldDescription;
   clone.fTypeName = fTypeName;
   clone.fNRepetitions = fNRepetitions;
   clone.fStructure = fStructure;
   clone.fParentId = fParentId;
   clone.fLinkIds = fLinkIds;
   return clone;
}

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RFieldDescriptor::CreateField(const RNTupleDescriptor &ntplDesc) const
{
   if (GetTypeName().empty() && GetStructure() == ENTupleStructure::kCollection) {
      // For untyped collections, we have no class available to collect all the sub fields.
      // Therefore, we create an untyped record field as an artifical binder for the collection items.
      std::vector<std::unique_ptr<Detail::RFieldBase>> memberFields;
      for (auto id : fLinkIds) {
         const auto &memberDesc = ntplDesc.GetFieldDescriptor(id);
         memberFields.emplace_back(memberDesc.CreateField(ntplDesc));
      }
      auto recordField = std::make_unique<RRecordField>("_0", memberFields);
      auto collectionField = std::make_unique<RVectorField>(GetFieldName(), std::move(recordField));
      collectionField->SetOnDiskId(fFieldId);
      return collectionField;
   }

   auto field = Detail::RFieldBase::Create(GetFieldName(), GetTypeName()).Unwrap();
   field->SetOnDiskId(fFieldId);
   for (auto &f : *field)
      f.SetOnDiskId(ntplDesc.FindFieldId(f.GetName(), f.GetParent()->GetOnDiskId()));
   return field;
}


////////////////////////////////////////////////////////////////////////////////


bool ROOT::Experimental::RColumnDescriptor::operator==(const RColumnDescriptor &other) const
{
   return fColumnId == other.fColumnId &&
          fModel == other.fModel &&
          fFieldId == other.fFieldId &&
          fIndex == other.fIndex;
}


ROOT::Experimental::RColumnDescriptor
ROOT::Experimental::RColumnDescriptor::Clone() const
{
   RColumnDescriptor clone;
   clone.fColumnId = fColumnId;
   clone.fModel = fModel;
   clone.fFieldId = fFieldId;
   clone.fIndex = fIndex;
   return clone;
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::RClusterDescriptor::RPageRange::RPageInfoExtended
ROOT::Experimental::RClusterDescriptor::RPageRange::Find(ROOT::Experimental::RClusterSize::ValueType idxInCluster) const
{
   // TODO(jblomer): binary search
   RPageInfo pageInfo;
   decltype(idxInCluster) firstInPage = 0;
   NTupleSize_t pageNo = 0;
   for (const auto &pi : fPageInfos) {
      if (firstInPage + pi.fNElements > idxInCluster) {
         pageInfo = pi;
         break;
      }
      firstInPage += pi.fNElements;
      ++pageNo;
   }
   R__ASSERT(firstInPage <= idxInCluster);
   R__ASSERT((firstInPage + pageInfo.fNElements) > idxInCluster);
   return RPageInfoExtended{pageInfo, firstInPage, pageNo};
}


bool ROOT::Experimental::RClusterDescriptor::operator==(const RClusterDescriptor &other) const
{
   return fClusterId == other.fClusterId && fFirstEntryIndex == other.fFirstEntryIndex &&
          fNEntries == other.fNEntries && fHasPageLocations == other.fHasPageLocations &&
          fColumnRanges == other.fColumnRanges && fPageRanges == other.fPageRanges;
}


std::unordered_set<ROOT::Experimental::DescriptorId_t> ROOT::Experimental::RClusterDescriptor::GetColumnIds() const
{
   EnsureHasPageLocations();
   std::unordered_set<DescriptorId_t> result;
   for (const auto &x : fColumnRanges)
      result.emplace(x.first);
   return result;
}


bool ROOT::Experimental::RClusterDescriptor::ContainsColumn(DescriptorId_t columnId) const
{
   EnsureHasPageLocations();
   return fColumnRanges.find(columnId) != fColumnRanges.end();
}


std::uint64_t ROOT::Experimental::RClusterDescriptor::GetBytesOnStorage() const
{
   EnsureHasPageLocations();
   std::uint64_t nbytes = 0;
   for (const auto &pr : fPageRanges) {
      for (const auto &pi : pr.second.fPageInfos) {
         nbytes += pi.fLocator.fBytesOnStorage;
      }
   }
   return nbytes;
}

void ROOT::Experimental::RClusterDescriptor::EnsureHasPageLocations() const
{
   if (!fHasPageLocations)
      throw RException(R__FAIL("invalid attempt to access page locations of summary-only cluster descriptor"));
}

////////////////////////////////////////////////////////////////////////////////


bool ROOT::Experimental::RNTupleDescriptor::operator==(const RNTupleDescriptor &other) const
{
   return fName == other.fName &&
          fDescription == other.fDescription &&
          fFieldDescriptors == other.fFieldDescriptors &&
          fColumnDescriptors == other.fColumnDescriptors &&
          fClusterDescriptors == other.fClusterDescriptors;
}


ROOT::Experimental::NTupleSize_t ROOT::Experimental::RNTupleDescriptor::GetNElements(DescriptorId_t columnId) const
{
   NTupleSize_t result = 0;
   for (const auto &cd : fClusterDescriptors) {
      if (!cd.second.ContainsColumn(columnId))
         continue;
      auto columnRange = cd.second.GetColumnRange(columnId);
      result = std::max(result, columnRange.fFirstElementIndex + columnRange.fNElements);
   }
   return result;
}


ROOT::Experimental::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::FindFieldId(std::string_view fieldName, DescriptorId_t parentId) const
{
   std::string leafName(fieldName);
   auto posDot = leafName.find_last_of('.');
   if (posDot != std::string::npos) {
      auto parentName = leafName.substr(0, posDot);
      leafName = leafName.substr(posDot + 1);
      parentId = FindFieldId(parentName, parentId);
   }
   for (const auto &fd : fFieldDescriptors) {
      if (fd.second.GetParentId() == parentId && fd.second.GetFieldName() == leafName)
         return fd.second.GetId();
   }
   return kInvalidDescriptorId;
}


std::string ROOT::Experimental::RNTupleDescriptor::GetQualifiedFieldName(DescriptorId_t fieldId) const
{
   if (fieldId == kInvalidDescriptorId)
      return "";

   const auto &fieldDescriptor = fFieldDescriptors.at(fieldId);
   auto prefix = GetQualifiedFieldName(fieldDescriptor.GetParentId());
   if (prefix.empty())
      return fieldDescriptor.GetFieldName();
   return prefix + "." + fieldDescriptor.GetFieldName();
}


ROOT::Experimental::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::GetFieldZeroId() const
{
   return FindFieldId("", kInvalidDescriptorId);
}


ROOT::Experimental::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::FindFieldId(std::string_view fieldName) const
{
   return FindFieldId(fieldName, GetFieldZeroId());
}


ROOT::Experimental::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::FindColumnId(DescriptorId_t fieldId, std::uint32_t columnIndex) const
{
   for (const auto &cd : fColumnDescriptors) {
      if (cd.second.GetFieldId() == fieldId && cd.second.GetIndex() == columnIndex)
        return cd.second.GetId();
   }
   return kInvalidDescriptorId;
}

ROOT::Experimental::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::FindClusterId(DescriptorId_t columnId, NTupleSize_t index) const
{
   // TODO(jblomer): binary search?
   for (const auto &cd : fClusterDescriptors) {
      if (!cd.second.ContainsColumn(columnId))
         continue;
      auto columnRange = cd.second.GetColumnRange(columnId);
      if (columnRange.Contains(index))
         return cd.second.GetId();
   }
   return kInvalidDescriptorId;
}


// TODO(jblomer): fix for cases of sharded clasters
ROOT::Experimental::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::FindNextClusterId(DescriptorId_t clusterId) const
{
   const auto &clusterDesc = GetClusterDescriptor(clusterId);
   auto firstEntryInNextCluster = clusterDesc.GetFirstEntryIndex() + clusterDesc.GetNEntries();
   // TODO(jblomer): binary search?
   for (const auto &cd : fClusterDescriptors) {
      if (cd.second.GetFirstEntryIndex() == firstEntryInNextCluster)
         return cd.second.GetId();
   }
   return kInvalidDescriptorId;
}


// TODO(jblomer): fix for cases of sharded clasters
ROOT::Experimental::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::FindPrevClusterId(DescriptorId_t clusterId) const
{
   const auto &clusterDesc = GetClusterDescriptor(clusterId);
   // TODO(jblomer): binary search?
   for (const auto &cd : fClusterDescriptors) {
      if (cd.second.GetFirstEntryIndex() + cd.second.GetNEntries() == clusterDesc.GetFirstEntryIndex())
         return cd.second.GetId();
   }
   return kInvalidDescriptorId;
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleDescriptor::AddClusterDetails(RClusterDescriptor &&clusterDesc)
{
   auto iter = fClusterDescriptors.find(clusterDesc.GetId());
   if (iter == fClusterDescriptors.end())
      return R__FAIL("invalid attempt to add cluster details without known cluster summary");
   if (iter->second.HasPageLocations())
      return R__FAIL("invalid attempt to re-populate page list");
   if (!clusterDesc.HasPageLocations())
      return R__FAIL("provided cluster descriptor does not contain page locations");
   iter->second = std::move(clusterDesc);
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void> ROOT::Experimental::RNTupleDescriptor::DropClusterDetails(DescriptorId_t clusterId)
{
   auto iter = fClusterDescriptors.find(clusterId);
   if (iter == fClusterDescriptors.end())
      return R__FAIL("invalid attempt to drop cluster details of unknown cluster");
   if (!iter->second.HasPageLocations())
      return R__FAIL("invalid attempt to drop details of cluster summary");
   iter->second = RClusterDescriptor(clusterId, iter->second.GetFirstEntryIndex(), iter->second.GetNEntries());
   return RResult<void>::Success();
}

std::unique_ptr<ROOT::Experimental::RNTupleModel> ROOT::Experimental::RNTupleDescriptor::GenerateModel() const
{
   auto model = RNTupleModel::Create();
   model->GetFieldZero()->SetOnDiskId(GetFieldZeroId());
   for (const auto &topDesc : GetTopLevelFields())
      model->AddField(topDesc.CreateField(*this));
   model->Freeze();
   return model;
}


////////////////////////////////////////////////////////////////////////////////

bool ROOT::Experimental::RColumnGroupDescriptor::operator==(const RColumnGroupDescriptor &other) const
{
   return fColumnGroupId == other.fColumnGroupId && fColumnIds == other.fColumnIds;
}

////////////////////////////////////////////////////////////////////////////////

bool ROOT::Experimental::RClusterGroupDescriptor::operator==(const RClusterGroupDescriptor &other) const
{
   return fClusterGroupId == other.fClusterGroupId && fClusterIds == other.fClusterIds;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::RResult<void>
ROOT::Experimental::RClusterDescriptorBuilder::CommitColumnRange(
   DescriptorId_t columnId, std::uint64_t firstElementIndex, std::uint32_t compressionSettings,
   const RClusterDescriptor::RPageRange &pageRange)
{
   if (columnId != pageRange.fColumnId)
      return R__FAIL("column ID mismatch");
   if (fCluster.fPageRanges.count(columnId) > 0)
      return R__FAIL("column ID conflict");
   RClusterDescriptor::RColumnRange columnRange{columnId, firstElementIndex, RClusterSize(0)};
   columnRange.fCompressionSettings = compressionSettings;
   for (const auto &pi : pageRange.fPageInfos) {
      columnRange.fNElements += pi.fNElements;
   }
   fCluster.fPageRanges[columnId] = pageRange.Clone();
   fCluster.fColumnRanges[columnId] = columnRange;
   return RResult<void>::Success();
}


ROOT::Experimental::RResult<ROOT::Experimental::RClusterDescriptor>
ROOT::Experimental::RClusterDescriptorBuilder::MoveDescriptor()
{
   if (fCluster.fClusterId == kInvalidDescriptorId)
      return R__FAIL("unset cluster ID");
   if (fCluster.fNEntries == 0)
      return R__FAIL("empty cluster");
   for (const auto &pr : fCluster.fPageRanges) {
      if (fCluster.fColumnRanges.count(pr.first) == 0) {
         return R__FAIL("missing column range");
      }
   }
   fCluster.fHasPageLocations = true;
   RClusterDescriptor result;
   std::swap(result, fCluster);
   return result;
}

std::vector<ROOT::Experimental::RClusterDescriptorBuilder>
ROOT::Experimental::RClusterGroupDescriptorBuilder::GetClusterSummaries(const RNTupleDescriptor &ntplDesc,
                                                                        DescriptorId_t clusterGroupId)
{
   const auto &clusterGroupDesc = ntplDesc.GetClusterGroupDescriptor(clusterGroupId);
   std::vector<RClusterDescriptorBuilder> result;
   for (auto clusterId : clusterGroupDesc.fClusterIds) {
      const auto &cluster = ntplDesc.GetClusterDescriptor(clusterId);
      result.emplace_back(RClusterDescriptorBuilder(clusterId, cluster.GetFirstEntryIndex(), cluster.GetNEntries()));
   }
   return result;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::RResult<ROOT::Experimental::RClusterGroupDescriptor>
ROOT::Experimental::RClusterGroupDescriptorBuilder::MoveDescriptor()
{
   if (fClusterGroup.fClusterGroupId == kInvalidDescriptorId)
      return R__FAIL("unset cluster group ID");
   RClusterGroupDescriptor result;
   std::swap(result, fClusterGroup);
   return result;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::RResult<ROOT::Experimental::RColumnGroupDescriptor>
ROOT::Experimental::RColumnGroupDescriptorBuilder::MoveDescriptor()
{
   if (fColumnGroup.fColumnGroupId == kInvalidDescriptorId)
      return R__FAIL("unset column group ID");
   RColumnGroupDescriptor result;
   std::swap(result, fColumnGroup);
   return result;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleDescriptorBuilder::EnsureFieldExists(DescriptorId_t fieldId) const {
   if (fDescriptor.fFieldDescriptors.count(fieldId) == 0)
      return R__FAIL("field with id '" + std::to_string(fieldId) + "' doesn't exist");
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleDescriptorBuilder::EnsureValidDescriptor() const {
   // Reuse field name validity check
   auto validName = Detail::RFieldBase::EnsureValidFieldName(fDescriptor.GetName());
   if (!validName) {
      return R__FORWARD_ERROR(validName);
   }
   // open-ended list of invariant checks
   for (const auto& key_val: fDescriptor.fFieldDescriptors) {
      const auto& id = key_val.first;
      const auto& desc = key_val.second;
      // parent not properly set
      if (id != DescriptorId_t(0) && desc.GetParentId() == kInvalidDescriptorId) {
         return R__FAIL("field with id '" + std::to_string(id) + "' has an invalid parent id");
      }
   }
   return RResult<void>::Success();
}

ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::RNTupleDescriptorBuilder::MoveDescriptor()
{
   RNTupleDescriptor result;
   std::swap(result, fDescriptor);
   return result;
}

void ROOT::Experimental::RNTupleDescriptorBuilder::SetNTuple(const std::string_view name,
                                                             const std::string_view description)
{
   fDescriptor.fName = std::string(name);
   fDescriptor.fDescription = std::string(description);
}

ROOT::Experimental::RResult<ROOT::Experimental::RColumnDescriptor>
ROOT::Experimental::RColumnDescriptorBuilder::MakeDescriptor() const
{
   if (fColumn.GetId() == kInvalidDescriptorId)
      return R__FAIL("invalid column id");
   if (fColumn.GetModel().GetType() == EColumnType::kUnknown)
      return R__FAIL("invalid column model");
   if (fColumn.GetFieldId() == kInvalidDescriptorId)
      return R__FAIL("invalid field id, dangling column");
   return fColumn.Clone();
}

ROOT::Experimental::RFieldDescriptorBuilder::RFieldDescriptorBuilder(
   const RFieldDescriptor& fieldDesc) : fField(fieldDesc.Clone())
{
   fField.fParentId = kInvalidDescriptorId;
   fField.fLinkIds = {};
}

ROOT::Experimental::RFieldDescriptorBuilder
ROOT::Experimental::RFieldDescriptorBuilder::FromField(const Detail::RFieldBase& field) {
   RFieldDescriptorBuilder fieldDesc;
   fieldDesc.FieldVersion(field.GetFieldVersion())
      .TypeVersion(field.GetTypeVersion())
      .FieldName(field.GetName())
      .FieldDescription(field.GetDescription())
      .TypeName(field.GetType())
      .Structure(field.GetStructure())
      .NRepetitions(field.GetNRepetitions());
   return fieldDesc;
}

ROOT::Experimental::RResult<ROOT::Experimental::RFieldDescriptor>
ROOT::Experimental::RFieldDescriptorBuilder::MakeDescriptor() const {
   if (fField.GetId() == kInvalidDescriptorId) {
      return R__FAIL("invalid field id");
   }
   if (fField.GetStructure() == ENTupleStructure::kInvalid) {
      return R__FAIL("invalid field structure");
   }
   // FieldZero is usually named "" and would be a false positive here
   if (fField.GetParentId() != kInvalidDescriptorId) {
      auto validName = Detail::RFieldBase::EnsureValidFieldName(fField.GetFieldName());
      if (!validName) {
         return R__FORWARD_ERROR(validName);
      }
   }
   return fField.Clone();
}

void ROOT::Experimental::RNTupleDescriptorBuilder::AddField(const RFieldDescriptor& fieldDesc) {
   fDescriptor.fFieldDescriptors.emplace(fieldDesc.GetId(), fieldDesc.Clone());
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleDescriptorBuilder::AddFieldLink(DescriptorId_t fieldId, DescriptorId_t linkId)
{
   auto fieldExists = RResult<void>::Success();
   if (!(fieldExists = EnsureFieldExists(fieldId)))
      return R__FORWARD_ERROR(fieldExists);
   if (!(fieldExists = EnsureFieldExists(linkId)))
      return  R__FAIL("child field with id '" + std::to_string(linkId) + "' doesn't exist in NTuple");

   if (linkId == fDescriptor.GetFieldZeroId()) {
      return R__FAIL("cannot make FieldZero a child field");
   }
   // fail if field already has another valid parent
   auto parentId = fDescriptor.fFieldDescriptors.at(linkId).GetParentId();
   if ((parentId != kInvalidDescriptorId) && (parentId != fieldId)) {
      return R__FAIL("field '" + std::to_string(linkId) + "' already has a parent ('" +
         std::to_string(parentId) + ")");
   }
   if (fieldId == linkId) {
      return R__FAIL("cannot make field '" + std::to_string(fieldId) + "' a child of itself");
   }
   fDescriptor.fFieldDescriptors.at(linkId).fParentId = fieldId;
   fDescriptor.fFieldDescriptors.at(fieldId).fLinkIds.push_back(linkId);
   return RResult<void>::Success();
}

void ROOT::Experimental::RNTupleDescriptorBuilder::AddColumn(DescriptorId_t columnId, DescriptorId_t fieldId,
                                                             const RColumnModel &model, std::uint32_t index)
{
   RColumnDescriptor c;
   c.fColumnId = columnId;
   c.fFieldId = fieldId;
   c.fModel = model;
   c.fIndex = index;
   fDescriptor.fColumnDescriptors.emplace(columnId, std::move(c));
}


ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleDescriptorBuilder::AddColumn(RColumnDescriptor &&columnDesc)
{
   const auto fieldId = columnDesc.GetFieldId();
   const auto index = columnDesc.GetIndex();

   auto fieldExists = EnsureFieldExists(fieldId);
   if (!fieldExists)
      return R__FORWARD_ERROR(fieldExists);
   if (fDescriptor.FindColumnId(fieldId, index) != kInvalidDescriptorId) {
      return R__FAIL("column index clash");
   }
   if (index > 0) {
      if (fDescriptor.FindColumnId(fieldId, index - 1) == kInvalidDescriptorId)
         return R__FAIL("out of bounds column index");
   }

   auto columnId = columnDesc.GetId();
   fDescriptor.fColumnDescriptors.emplace(columnId, std::move(columnDesc));

   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleDescriptorBuilder::AddClusterSummary(DescriptorId_t clusterId, std::uint64_t firstEntry,
                                                                std::uint64_t nEntries)
{
   if (fDescriptor.fClusterDescriptors.count(clusterId) > 0)
      return R__FAIL("cluster id clash while adding cluster summary");
   fDescriptor.fNEntries = std::max(fDescriptor.fNEntries, firstEntry + nEntries);
   fDescriptor.fClusterDescriptors.emplace(clusterId, RClusterDescriptor(clusterId, firstEntry, nEntries));
   return RResult<void>::Success();
}

void ROOT::Experimental::RNTupleDescriptorBuilder::AddClusterGroup(RClusterGroupDescriptorBuilder &&clusterGroup)
{
   auto id = clusterGroup.GetId();
   fDescriptor.fClusterGroupDescriptors.emplace(id, clusterGroup.MoveDescriptor().Unwrap());
}

void ROOT::Experimental::RNTupleDescriptorBuilder::Reset()
{
   fDescriptor.fName = "";
   fDescriptor.fDescription = "";
   fDescriptor.fFieldDescriptors.clear();
   fDescriptor.fColumnDescriptors.clear();
   fDescriptor.fClusterDescriptors.clear();
   fDescriptor.fClusterGroupDescriptors.clear();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleDescriptorBuilder::AddClusterWithDetails(RClusterDescriptor &&clusterDesc)
{
   auto clusterId = clusterDesc.GetId();
   if (fDescriptor.fClusterDescriptors.count(clusterId) > 0)
      return R__FAIL("cluster id clash");
   fDescriptor.fNEntries =
      std::max(fDescriptor.fNEntries, clusterDesc.GetFirstEntryIndex() + clusterDesc.GetNEntries());
   fDescriptor.fClusterDescriptors.emplace(clusterId, std::move(clusterDesc));
   return RResult<void>::Success();
}
