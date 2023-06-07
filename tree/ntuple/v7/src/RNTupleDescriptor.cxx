/// \file RNTupleDescriptor.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Javier Lopez-Gomez <javier.lopez.gomez@cern.ch>
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
#include <ROOT/RPage.hxx>
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
   return fLogicalColumnId == other.fLogicalColumnId && fPhysicalColumnId == other.fPhysicalColumnId &&
          fModel == other.fModel && fFieldId == other.fFieldId && fIndex == other.fIndex;
}


ROOT::Experimental::RColumnDescriptor
ROOT::Experimental::RColumnDescriptor::Clone() const
{
   RColumnDescriptor clone;
   clone.fLogicalColumnId = fLogicalColumnId;
   clone.fPhysicalColumnId = fPhysicalColumnId;
   clone.fModel = fModel;
   clone.fFieldId = fFieldId;
   clone.fIndex = fIndex;
   clone.fFirstElementIndex = fFirstElementIndex;
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

std::size_t
ROOT::Experimental::RClusterDescriptor::RPageRange::ExtendToFitColumnRange(const RColumnRange &columnRange,
                                                                           const Detail::RColumnElementBase &element,
                                                                           std::size_t pageSize)
{
   R__ASSERT(fPhysicalColumnId == columnRange.fPhysicalColumnId);

   const auto nElements = std::accumulate(fPageInfos.begin(), fPageInfos.end(), 0U,
                                          [](std::size_t n, const auto &PI) { return n + PI.fNElements; });
   const auto nElementsRequired = static_cast<std::uint64_t>(columnRange.fNElements);

   if (nElementsRequired == nElements)
      return 0U;
   R__ASSERT((nElementsRequired > nElements) && "invalid attempt to shrink RPageRange");

   std::vector<RPageInfo> pageInfos;
   // Synthesize new `RPageInfo`s as needed
   const std::uint64_t nElementsPerPage = pageSize / element.GetSize();
   R__ASSERT(nElementsPerPage > 0);
   for (auto nRemainingElements = nElementsRequired - nElements; nRemainingElements > 0;) {
      RPageInfo PI;
      PI.fNElements = std::min(nElementsPerPage, nRemainingElements);
      PI.fLocator.fType = RNTupleLocator::kTypePageZero;
      PI.fLocator.fBytesOnStorage = element.GetPackedSize(PI.fNElements);
      pageInfos.emplace_back(PI);
      nRemainingElements -= PI.fNElements;
   }

   pageInfos.insert(pageInfos.end(), std::make_move_iterator(fPageInfos.begin()),
                    std::make_move_iterator(fPageInfos.end()));
   std::swap(fPageInfos, pageInfos);
   return nElementsRequired - nElements;
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

bool ROOT::Experimental::RClusterDescriptor::ContainsColumn(DescriptorId_t physicalId) const
{
   EnsureHasPageLocations();
   return fColumnRanges.find(physicalId) != fColumnRanges.end();
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

ROOT::Experimental::RClusterDescriptor ROOT::Experimental::RClusterDescriptor::Clone() const
{
   RClusterDescriptor clone;
   clone.fClusterId = fClusterId;
   clone.fFirstEntryIndex = fFirstEntryIndex;
   clone.fNEntries = fNEntries;
   clone.fHasPageLocations = fHasPageLocations;
   clone.fColumnRanges = fColumnRanges;
   for (const auto &d : fPageRanges)
      clone.fPageRanges.emplace(d.first, d.second.Clone());
   return clone;
}

////////////////////////////////////////////////////////////////////////////////


bool ROOT::Experimental::RNTupleDescriptor::operator==(const RNTupleDescriptor &other) const
{
   return fName == other.fName && fDescription == other.fDescription && fNEntries == other.fNEntries &&
          fGeneration == other.fGeneration && fFieldDescriptors == other.fFieldDescriptors &&
          fColumnDescriptors == other.fColumnDescriptors &&
          fClusterGroupDescriptors == other.fClusterGroupDescriptors &&
          fClusterDescriptors == other.fClusterDescriptors;
}

ROOT::Experimental::NTupleSize_t
ROOT::Experimental::RNTupleDescriptor::GetNElements(DescriptorId_t physicalColumnId) const
{
   NTupleSize_t result = 0;
   for (const auto &cd : fClusterDescriptors) {
      if (!cd.second.ContainsColumn(physicalColumnId))
         continue;
      auto columnRange = cd.second.GetColumnRange(physicalColumnId);
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
ROOT::Experimental::RNTupleDescriptor::FindLogicalColumnId(DescriptorId_t fieldId, std::uint32_t columnIndex) const
{
   for (const auto &cd : fColumnDescriptors) {
      if (cd.second.GetFieldId() == fieldId && cd.second.GetIndex() == columnIndex)
         return cd.second.GetLogicalId();
   }
   return kInvalidDescriptorId;
}

ROOT::Experimental::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::FindPhysicalColumnId(DescriptorId_t fieldId, std::uint32_t columnIndex) const
{
   auto logicalId = FindLogicalColumnId(fieldId, columnIndex);
   if (logicalId == kInvalidDescriptorId)
      return kInvalidDescriptorId;
   return GetColumnDescriptor(logicalId).GetPhysicalId();
}

ROOT::Experimental::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::FindClusterId(DescriptorId_t physicalColumnId, NTupleSize_t index) const
{
   // TODO(jblomer): binary search?
   for (const auto &cd : fClusterDescriptors) {
      if (!cd.second.ContainsColumn(physicalColumnId))
         continue;
      auto columnRange = cd.second.GetColumnRange(physicalColumnId);
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

std::vector<ROOT::Experimental::DescriptorId_t>
ROOT::Experimental::RNTupleDescriptor::RHeaderExtension::GetTopLevelFields(const RNTupleDescriptor &desc) const
{
   auto fieldZeroId = desc.GetFieldZeroId();

   std::vector<DescriptorId_t> fields;
   for (const DescriptorId_t fieldId : fFields) {
      if (desc.GetFieldDescriptor(fieldId).GetParentId() == fieldZeroId)
         fields.emplace_back(fieldId);
   }
   return fields;
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

std::unique_ptr<ROOT::Experimental::RNTupleDescriptor> ROOT::Experimental::RNTupleDescriptor::Clone() const
{
   auto clone = std::make_unique<RNTupleDescriptor>();
   clone->fName = fName;
   clone->fDescription = fDescription;
   clone->fOnDiskHeaderSize = fOnDiskHeaderSize;
   clone->fOnDiskFooterSize = fOnDiskFooterSize;
   clone->fNEntries = fNEntries;
   clone->fNPhysicalColumns = fNPhysicalColumns;
   clone->fGeneration = fGeneration;
   for (const auto &d : fFieldDescriptors)
      clone->fFieldDescriptors.emplace(d.first, d.second.Clone());
   for (const auto &d : fColumnDescriptors)
      clone->fColumnDescriptors.emplace(d.first, d.second.Clone());
   for (const auto &d : fClusterGroupDescriptors)
      clone->fClusterGroupDescriptors.emplace(d.first, d.second.Clone());
   for (const auto &d : fClusterDescriptors)
      clone->fClusterDescriptors.emplace(d.first, d.second.Clone());
   if (fHeaderExtension)
      clone->fHeaderExtension = std::make_unique<RHeaderExtension>(*fHeaderExtension);
   return clone;
}

////////////////////////////////////////////////////////////////////////////////

bool ROOT::Experimental::RColumnGroupDescriptor::operator==(const RColumnGroupDescriptor &other) const
{
   return fColumnGroupId == other.fColumnGroupId && fPhysicalColumnIds == other.fPhysicalColumnIds;
}

////////////////////////////////////////////////////////////////////////////////

bool ROOT::Experimental::RClusterGroupDescriptor::operator==(const RClusterGroupDescriptor &other) const
{
   return fClusterGroupId == other.fClusterGroupId && fClusterIds == other.fClusterIds;
}

ROOT::Experimental::RClusterGroupDescriptor ROOT::Experimental::RClusterGroupDescriptor::Clone() const
{
   RClusterGroupDescriptor clone;
   clone.fClusterGroupId = fClusterGroupId;
   clone.fClusterIds = fClusterIds;
   clone.fPageListLocator = fPageListLocator;
   clone.fPageListLength = fPageListLength;
   return clone;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::RResult<void>
ROOT::Experimental::RClusterDescriptorBuilder::CommitColumnRange(DescriptorId_t physicalId,
                                                                 std::uint64_t firstElementIndex,
                                                                 std::uint32_t compressionSettings,
                                                                 const RClusterDescriptor::RPageRange &pageRange)
{
   if (physicalId != pageRange.fPhysicalColumnId)
      return R__FAIL("column ID mismatch");
   if (fCluster.fPageRanges.count(physicalId) > 0)
      return R__FAIL("column ID conflict");
   RClusterDescriptor::RColumnRange columnRange{physicalId, firstElementIndex, RClusterSize(0)};
   columnRange.fCompressionSettings = compressionSettings;
   for (const auto &pi : pageRange.fPageInfos) {
      columnRange.fNElements += pi.fNElements;
   }
   fCluster.fPageRanges[physicalId] = pageRange.Clone();
   fCluster.fColumnRanges[physicalId] = columnRange;
   return RResult<void>::Success();
}

ROOT::Experimental::RClusterDescriptorBuilder &
ROOT::Experimental::RClusterDescriptorBuilder::AddDeferredColumnRanges(const RNTupleDescriptor &desc)
{
   /// Carries out a depth-first traversal of a field subtree rooted at `rootFieldId`.  For each field, `visitField` is
   /// called passing the field ID and the number of overall repetitions, taking into account the repetitions of each
   /// parent field in the hierarchy.
   auto fnTraverseSubtree = [&](DescriptorId_t rootFieldId, std::uint64_t nRepetitionsAtThisLevel,
                                const auto &visitField, const auto &enterSubtree) -> void {
      visitField(rootFieldId, nRepetitionsAtThisLevel);
      for (const auto &f : desc.GetFieldIterable(rootFieldId)) {
         const std::uint64_t nRepetitions = std::max(f.GetNRepetitions(), std::uint64_t{1U}) * nRepetitionsAtThisLevel;
         enterSubtree(f.GetId(), nRepetitions, visitField, enterSubtree);
      }
   };

   // Deferred columns can only be part of the header extension
   auto xHeader = desc.GetHeaderExtension();
   if (!xHeader)
      return *this;

   // Ensure that all columns in the header extension have their associated `R(Column|Page)Range`
   for (const auto &topLevelFieldId : xHeader->GetTopLevelFields(desc)) {
      fnTraverseSubtree(
         topLevelFieldId, std::max(desc.GetFieldDescriptor(topLevelFieldId).GetNRepetitions(), std::uint64_t{1U}),
         [&](DescriptorId_t fieldId, std::uint64_t nRepetitions) {
            for (const auto &c : desc.GetColumnIterable(fieldId)) {
               const DescriptorId_t physicalId = c.GetPhysicalId();
               auto &columnRange = fCluster.fColumnRanges[physicalId];
               auto &pageRange = fCluster.fPageRanges[physicalId];
               // Initialize a RColumnRange for `physicalId` if it was not there
               if (columnRange.fPhysicalColumnId == kInvalidDescriptorId) {
                  columnRange.fPhysicalColumnId = physicalId;
                  pageRange.fPhysicalColumnId = physicalId;
               }
               // Fixup the RColumnRange and RPageRange in deferred columns.  We know what the first element index and
               // number of elements should have been if the column was not deferred; fix those and let
               // `ExtendToFitColumnRange()` synthesize RPageInfos accordingly.
               // Note that a column whose first element index is != 0 already met the criteria of
               // `RFieldBase::EntryToColumnElementIndex()`, i.e. it is a principal column reachable from the field zero
               // excluding subfields of collection and variant fields.
               if (c.IsDeferredColumn()) {
                  columnRange.fFirstElementIndex = fCluster.GetFirstEntryIndex() * nRepetitions;
                  columnRange.fNElements = fCluster.GetNEntries() * nRepetitions;
                  const auto element = Detail::RColumnElementBase::Generate<void>(c.GetModel().GetType());
                  pageRange.ExtendToFitColumnRange(columnRange, *element, Detail::RPage::kPageZeroSize);
               }
            }
         },
         fnTraverseSubtree);
   }
   return *this;
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
   if (fColumn.GetLogicalId() == kInvalidDescriptorId)
      return R__FAIL("invalid logical column id");
   if (fColumn.GetPhysicalId() == kInvalidDescriptorId)
      return R__FAIL("invalid physical column id");
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
   if (fDescriptor.fHeaderExtension)
      fDescriptor.fHeaderExtension->AddFieldId(fieldDesc.GetId());
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

void ROOT::Experimental::RNTupleDescriptorBuilder::AddColumn(DescriptorId_t logicalId, DescriptorId_t physicalId,
                                                             DescriptorId_t fieldId, const RColumnModel &model,
                                                             std::uint32_t index, std::uint64_t firstElementIdx)
{
   RColumnDescriptor c;
   c.fLogicalColumnId = logicalId;
   c.fPhysicalColumnId = physicalId;
   c.fFieldId = fieldId;
   c.fModel = model;
   c.fIndex = index;
   c.fFirstElementIndex = firstElementIdx;
   if (!c.IsAliasColumn())
      fDescriptor.fNPhysicalColumns++;
   if (fDescriptor.fHeaderExtension)
      fDescriptor.fHeaderExtension->AddColumn(/*isAliasColumn=*/c.IsAliasColumn());
   fDescriptor.fColumnDescriptors.emplace(logicalId, std::move(c));
}


ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleDescriptorBuilder::AddColumn(RColumnDescriptor &&columnDesc)
{
   const auto fieldId = columnDesc.GetFieldId();
   const auto index = columnDesc.GetIndex();

   auto fieldExists = EnsureFieldExists(fieldId);
   if (!fieldExists)
      return R__FORWARD_ERROR(fieldExists);
   if (fDescriptor.FindLogicalColumnId(fieldId, index) != kInvalidDescriptorId) {
      return R__FAIL("column index clash");
   }
   if (index > 0) {
      if (fDescriptor.FindLogicalColumnId(fieldId, index - 1) == kInvalidDescriptorId)
         return R__FAIL("out of bounds column index");
   }
   if (columnDesc.IsAliasColumn()) {
      if (columnDesc.GetModel() != fDescriptor.GetColumnDescriptor(columnDesc.GetPhysicalId()).GetModel())
         return R__FAIL("alias column type mismatch");
   }

   auto logicalId = columnDesc.GetLogicalId();
   if (!columnDesc.IsAliasColumn())
      fDescriptor.fNPhysicalColumns++;
   fDescriptor.fColumnDescriptors.emplace(logicalId, std::move(columnDesc));
   if (fDescriptor.fHeaderExtension)
      fDescriptor.fHeaderExtension->AddColumn(/*isAliasColumn=*/columnDesc.IsAliasColumn());

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
   fDescriptor.fHeaderExtension.reset();
}

void ROOT::Experimental::RNTupleDescriptorBuilder::BeginHeaderExtension()
{
   if (!fDescriptor.fHeaderExtension)
      fDescriptor.fHeaderExtension = std::make_unique<RNTupleDescriptor::RHeaderExtension>();
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
