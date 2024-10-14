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
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPage.hxx>
#include <string_view>

#include <RZip.h>
#include <TError.h>
#include <TVirtualStreamerInfo.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <set>
#include <utility>

bool ROOT::Experimental::RFieldDescriptor::operator==(const RFieldDescriptor &other) const
{
   return fFieldId == other.fFieldId && fFieldVersion == other.fFieldVersion && fTypeVersion == other.fTypeVersion &&
          fFieldName == other.fFieldName && fFieldDescription == other.fFieldDescription &&
          fTypeName == other.fTypeName && fTypeAlias == other.fTypeAlias && fNRepetitions == other.fNRepetitions &&
          fStructure == other.fStructure && fParentId == other.fParentId &&
          fProjectionSourceId == other.fProjectionSourceId && fLinkIds == other.fLinkIds &&
          fLogicalColumnIds == other.fLogicalColumnIds && other.fTypeChecksum == other.fTypeChecksum;
}

ROOT::Experimental::RFieldDescriptor ROOT::Experimental::RFieldDescriptor::Clone() const
{
   RFieldDescriptor clone;
   clone.fFieldId = fFieldId;
   clone.fFieldVersion = fFieldVersion;
   clone.fTypeVersion = fTypeVersion;
   clone.fFieldName = fFieldName;
   clone.fFieldDescription = fFieldDescription;
   clone.fTypeName = fTypeName;
   clone.fTypeAlias = fTypeAlias;
   clone.fNRepetitions = fNRepetitions;
   clone.fStructure = fStructure;
   clone.fParentId = fParentId;
   clone.fProjectionSourceId = fProjectionSourceId;
   clone.fLinkIds = fLinkIds;
   clone.fColumnCardinality = fColumnCardinality;
   clone.fLogicalColumnIds = fLogicalColumnIds;
   clone.fTypeChecksum = fTypeChecksum;
   return clone;
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RFieldDescriptor::CreateField(const RNTupleDescriptor &ntplDesc, bool continueOnError) const
{
   if (GetStructure() == ENTupleStructure::kUnsplit) {
      auto unsplitField = std::make_unique<RUnsplitField>(GetFieldName(), GetTypeName());
      unsplitField->SetOnDiskId(fFieldId);
      return unsplitField;
   }

   if (GetTypeName().empty()) {
      switch (GetStructure()) {
      case ENTupleStructure::kRecord: {
         std::vector<std::unique_ptr<RFieldBase>> memberFields;
         for (auto id : fLinkIds) {
            const auto &memberDesc = ntplDesc.GetFieldDescriptor(id);
            auto field = memberDesc.CreateField(ntplDesc, continueOnError);
            if (dynamic_cast<RInvalidField *>(field.get()))
               return field;
            memberFields.emplace_back(std::move(field));
         }
         auto recordField = std::make_unique<RRecordField>(GetFieldName(), memberFields);
         recordField->SetOnDiskId(fFieldId);
         return recordField;
      }
      case ENTupleStructure::kCollection: {
         if (fLinkIds.size() != 1) {
            throw RException(R__FAIL("unsupported untyped collection for field \"" + GetFieldName() + "\""));
         }
         auto itemField = ntplDesc.GetFieldDescriptor(fLinkIds[0]).CreateField(ntplDesc, continueOnError);
         if (dynamic_cast<RInvalidField *>(itemField.get()))
            return itemField;
         auto collectionField = RVectorField::CreateUntyped(GetFieldName(), std::move(itemField));
         collectionField->SetOnDiskId(fFieldId);
         return collectionField;
      }
      default: throw RException(R__FAIL("unsupported untyped field structure for field \"" + GetFieldName() + "\""));
      }
   }

   try {
      auto field = RFieldBase::Create(GetFieldName(), GetTypeAlias().empty() ? GetTypeName() : GetTypeAlias()).Unwrap();
      field->SetOnDiskId(fFieldId);
      for (auto &f : *field)
         f.SetOnDiskId(ntplDesc.FindFieldId(f.GetFieldName(), f.GetParent()->GetOnDiskId()));
      return field;
   } catch (RException &ex) {
      if (continueOnError)
         return std::make_unique<RInvalidField>(GetFieldName(), GetTypeName(), ex.GetError().GetReport());
      else
         throw ex;
   }
}

bool ROOT::Experimental::RFieldDescriptor::IsCustomClass() const
{
   if (fStructure != ENTupleStructure::kRecord && fStructure != ENTupleStructure::kUnsplit)
      return false;

   // Skip untyped structs
   if (fTypeName.empty())
      return false;

   if (fStructure == ENTupleStructure::kRecord) {
      if (fTypeName.compare(0, 10, "std::pair<") == 0)
         return false;
      if (fTypeName.compare(0, 11, "std::tuple<") == 0)
         return false;
   }

   return true;
}

////////////////////////////////////////////////////////////////////////////////

bool ROOT::Experimental::RColumnDescriptor::operator==(const RColumnDescriptor &other) const
{
   return fLogicalColumnId == other.fLogicalColumnId && fPhysicalColumnId == other.fPhysicalColumnId &&
          fBitsOnStorage == other.fBitsOnStorage && fType == other.fType && fFieldId == other.fFieldId &&
          fIndex == other.fIndex && fRepresentationIndex == other.fRepresentationIndex &&
          fValueRange == other.fValueRange;
}

ROOT::Experimental::RColumnDescriptor ROOT::Experimental::RColumnDescriptor::Clone() const
{
   RColumnDescriptor clone;
   clone.fLogicalColumnId = fLogicalColumnId;
   clone.fPhysicalColumnId = fPhysicalColumnId;
   clone.fBitsOnStorage = fBitsOnStorage;
   clone.fType = fType;
   clone.fFieldId = fFieldId;
   clone.fIndex = fIndex;
   clone.fFirstElementIndex = fFirstElementIndex;
   clone.fRepresentationIndex = fRepresentationIndex;
   clone.fValueRange = fValueRange;
   return clone;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::RClusterDescriptor::RPageRange::RPageInfoExtended
ROOT::Experimental::RClusterDescriptor::RPageRange::Find(ClusterSize_t::ValueType idxInCluster) const
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
                                                                           const Internal::RColumnElementBase &element,
                                                                           std::size_t pageSize)
{
   R__ASSERT(fPhysicalColumnId == columnRange.fPhysicalColumnId);
   R__ASSERT(!columnRange.fIsSuppressed);

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
          fNEntries == other.fNEntries && fColumnRanges == other.fColumnRanges && fPageRanges == other.fPageRanges;
}

std::uint64_t ROOT::Experimental::RClusterDescriptor::GetBytesOnStorage() const
{
   std::uint64_t nbytes = 0;
   for (const auto &pr : fPageRanges) {
      for (const auto &pi : pr.second.fPageInfos) {
         nbytes += pi.fLocator.fBytesOnStorage;
      }
   }
   return nbytes;
}

ROOT::Experimental::RClusterDescriptor ROOT::Experimental::RClusterDescriptor::Clone() const
{
   RClusterDescriptor clone;
   clone.fClusterId = fClusterId;
   clone.fFirstEntryIndex = fFirstEntryIndex;
   clone.fNEntries = fNEntries;
   clone.fColumnRanges = fColumnRanges;
   for (const auto &d : fPageRanges)
      clone.fPageRanges.emplace(d.first, d.second.Clone());
   return clone;
}

////////////////////////////////////////////////////////////////////////////////

bool ROOT::Experimental::RExtraTypeInfoDescriptor::operator==(const RExtraTypeInfoDescriptor &other) const
{
   return fContentId == other.fContentId && fTypeName == other.fTypeName &&
          fTypeVersionFrom == other.fTypeVersionFrom && fTypeVersionTo == other.fTypeVersionTo;
}

ROOT::Experimental::RExtraTypeInfoDescriptor ROOT::Experimental::RExtraTypeInfoDescriptor::Clone() const
{
   RExtraTypeInfoDescriptor clone;
   clone.fContentId = fContentId;
   clone.fTypeVersionFrom = fTypeVersionFrom;
   clone.fTypeVersionTo = fTypeVersionTo;
   clone.fTypeName = fTypeName;
   clone.fContent = fContent;
   return clone;
}

////////////////////////////////////////////////////////////////////////////////

bool ROOT::Experimental::RNTupleDescriptor::operator==(const RNTupleDescriptor &other) const
{
   // clang-format off
   return fName == other.fName &&
          fDescription == other.fDescription &&
          fNEntries == other.fNEntries &&
          fGeneration == other.fGeneration &&
          fFieldZeroId == other.fFieldZeroId &&
          fFieldDescriptors == other.fFieldDescriptors &&
          fColumnDescriptors == other.fColumnDescriptors &&
          fClusterGroupDescriptors == other.fClusterGroupDescriptors &&
          fClusterDescriptors == other.fClusterDescriptors;
   // clang-format on
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
   auto itrFieldDesc = fFieldDescriptors.find(parentId);
   if (itrFieldDesc == fFieldDescriptors.end())
      return kInvalidDescriptorId;
   for (const auto linkId : itrFieldDesc->second.GetLinkIds()) {
      if (fFieldDescriptors.at(linkId).GetFieldName() == leafName)
         return linkId;
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

ROOT::Experimental::DescriptorId_t ROOT::Experimental::RNTupleDescriptor::FindFieldId(std::string_view fieldName) const
{
   return FindFieldId(fieldName, GetFieldZeroId());
}

ROOT::Experimental::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::FindLogicalColumnId(DescriptorId_t fieldId, std::uint32_t columnIndex,
                                                           std::uint16_t representationIndex) const
{
   auto itr = fFieldDescriptors.find(fieldId);
   if (itr == fFieldDescriptors.cend())
      return kInvalidDescriptorId;
   if (columnIndex >= itr->second.GetColumnCardinality())
      return kInvalidDescriptorId;
   const auto idx = representationIndex * itr->second.GetColumnCardinality() + columnIndex;
   if (itr->second.GetLogicalColumnIds().size() <= idx)
      return kInvalidDescriptorId;
   return itr->second.GetLogicalColumnIds()[idx];
}

ROOT::Experimental::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::FindPhysicalColumnId(DescriptorId_t fieldId, std::uint32_t columnIndex,
                                                            std::uint16_t representationIndex) const
{
   auto logicalId = FindLogicalColumnId(fieldId, columnIndex, representationIndex);
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
   for (const DescriptorId_t fieldId : fFieldIdsOrder) {
      if (desc.GetFieldDescriptor(fieldId).GetParentId() == fieldZeroId)
         fields.emplace_back(fieldId);
   }
   return fields;
}

ROOT::Experimental::RNTupleDescriptor::RColumnDescriptorIterable::RColumnDescriptorIterable(
   const RNTupleDescriptor &ntuple, const RFieldDescriptor &field)
   : fNTuple(ntuple), fColumns(field.GetLogicalColumnIds())
{
}

ROOT::Experimental::RNTupleDescriptor::RColumnDescriptorIterable::RColumnDescriptorIterable(
   const RNTupleDescriptor &ntuple)
   : fNTuple(ntuple)
{
   std::deque<DescriptorId_t> fieldIdQueue{ntuple.GetFieldZeroId()};

   while (!fieldIdQueue.empty()) {
      auto currFieldId = fieldIdQueue.front();
      fieldIdQueue.pop_front();

      const auto &columns = ntuple.GetFieldDescriptor(currFieldId).GetLogicalColumnIds();
      fColumns.insert(fColumns.end(), columns.begin(), columns.end());

      for (const auto &field : ntuple.GetFieldIterable(currFieldId)) {
         auto fieldId = field.GetId();
         fieldIdQueue.push_back(fieldId);
      }
   }
}

std::vector<std::uint64_t> ROOT::Experimental::RNTupleDescriptor::GetFeatureFlags() const
{
   std::vector<std::uint64_t> result;
   unsigned int base = 0;
   std::uint64_t flags = 0;
   for (auto f : fFeatureFlags) {
      if ((f > 0) && ((f % 64) == 0))
         throw RException(R__FAIL("invalid feature flag: " + std::to_string(f)));
      while (f > base + 64) {
         result.emplace_back(flags);
         flags = 0;
         base += 64;
      }
      f -= base;
      flags |= 1 << f;
   }
   result.emplace_back(flags);
   return result;
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleDescriptor::AddClusterGroupDetails(DescriptorId_t clusterGroupId,
                                                              std::vector<RClusterDescriptor> &clusterDescs)
{
   auto iter = fClusterGroupDescriptors.find(clusterGroupId);
   if (iter == fClusterGroupDescriptors.end())
      return R__FAIL("invalid attempt to add details of unknown cluster group");
   if (iter->second.HasClusterDetails())
      return R__FAIL("invalid attempt to re-populate cluster group details");
   if (iter->second.GetNClusters() != clusterDescs.size())
      return R__FAIL("mismatch of number of clusters");

   std::vector<DescriptorId_t> clusterIds;
   for (unsigned i = 0; i < clusterDescs.size(); ++i) {
      clusterIds.emplace_back(clusterDescs[i].GetId());
      auto [_, success] = fClusterDescriptors.emplace(clusterIds.back(), std::move(clusterDescs[i]));
      if (!success) {
         return R__FAIL("invalid attempt to re-populate existing cluster");
      }
   }
   auto cgBuilder = Internal::RClusterGroupDescriptorBuilder::FromSummary(iter->second);
   cgBuilder.AddClusters(clusterIds);
   iter->second = cgBuilder.MoveDescriptor().Unwrap();
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleDescriptor::DropClusterGroupDetails(DescriptorId_t clusterGroupId)
{
   auto iter = fClusterGroupDescriptors.find(clusterGroupId);
   if (iter == fClusterGroupDescriptors.end())
      return R__FAIL("invalid attempt to drop cluster details of unknown cluster group");
   if (!iter->second.HasClusterDetails())
      return R__FAIL("invalid attempt to drop details of cluster group summary");

   for (auto clusterId : iter->second.GetClusterIds())
      fClusterDescriptors.erase(clusterId);
   iter->second = iter->second.CloneSummary();
   return RResult<void>::Success();
}

std::unique_ptr<ROOT::Experimental::RNTupleModel>
ROOT::Experimental::RNTupleDescriptor::CreateModel(const RCreateModelOptions &options) const
{
   auto fieldZero = std::make_unique<RFieldZero>();
   fieldZero->SetOnDiskId(GetFieldZeroId());
   auto model =
      options.fCreateBare ? RNTupleModel::CreateBare(std::move(fieldZero)) : RNTupleModel::Create(std::move(fieldZero));
   bool continueOnError = options.fForwardCompatible;
   for (const auto &topDesc : GetTopLevelFields()) {
      auto field = topDesc.CreateField(*this, continueOnError);
      if (dynamic_cast<RInvalidField *>(field.get()))
         continue;

      if (options.fReconstructProjections && topDesc.IsProjectedField()) {
         model->AddProjectedField(std::move(field), [this](const std::string &targetName) -> std::string {
            return GetQualifiedFieldName(GetFieldDescriptor(FindFieldId(targetName)).GetProjectionSourceId());
         });
      } else {
         model->AddField(std::move(field));
      }
   }
   model->Freeze();
   return model;
}

std::unique_ptr<ROOT::Experimental::RNTupleDescriptor> ROOT::Experimental::RNTupleDescriptor::Clone() const
{
   auto clone = std::make_unique<RNTupleDescriptor>();
   clone->fName = fName;
   clone->fDescription = fDescription;
   clone->fOnDiskHeaderXxHash3 = fOnDiskHeaderXxHash3;
   clone->fOnDiskHeaderSize = fOnDiskHeaderSize;
   clone->fOnDiskFooterSize = fOnDiskFooterSize;
   clone->fNEntries = fNEntries;
   clone->fNClusters = fNClusters;
   clone->fNPhysicalColumns = fNPhysicalColumns;
   clone->fFieldZeroId = fFieldZeroId;
   clone->fGeneration = fGeneration;
   for (const auto &d : fFieldDescriptors)
      clone->fFieldDescriptors.emplace(d.first, d.second.Clone());
   for (const auto &d : fColumnDescriptors)
      clone->fColumnDescriptors.emplace(d.first, d.second.Clone());
   for (const auto &d : fClusterGroupDescriptors)
      clone->fClusterGroupDescriptors.emplace(d.first, d.second.Clone());
   for (const auto &d : fClusterDescriptors)
      clone->fClusterDescriptors.emplace(d.first, d.second.Clone());
   for (const auto &d : fExtraTypeInfoDescriptors)
      clone->fExtraTypeInfoDescriptors.emplace_back(d.Clone());
   if (fHeaderExtension)
      clone->fHeaderExtension = std::make_unique<RHeaderExtension>(*fHeaderExtension);
   return clone;
}

////////////////////////////////////////////////////////////////////////////////

bool ROOT::Experimental::RClusterGroupDescriptor::operator==(const RClusterGroupDescriptor &other) const
{
   return fClusterGroupId == other.fClusterGroupId && fClusterIds == other.fClusterIds &&
          fMinEntry == other.fMinEntry && fEntrySpan == other.fEntrySpan && fNClusters == other.fNClusters;
}

ROOT::Experimental::RClusterGroupDescriptor ROOT::Experimental::RClusterGroupDescriptor::CloneSummary() const
{
   RClusterGroupDescriptor clone;
   clone.fClusterGroupId = fClusterGroupId;
   clone.fPageListLocator = fPageListLocator;
   clone.fPageListLength = fPageListLength;
   clone.fMinEntry = fMinEntry;
   clone.fEntrySpan = fEntrySpan;
   clone.fNClusters = fNClusters;
   return clone;
}

ROOT::Experimental::RClusterGroupDescriptor ROOT::Experimental::RClusterGroupDescriptor::Clone() const
{
   RClusterGroupDescriptor clone = CloneSummary();
   clone.fClusterIds = fClusterIds;
   return clone;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::RResult<void> ROOT::Experimental::Internal::RClusterDescriptorBuilder::CommitColumnRange(
   DescriptorId_t physicalId, std::uint64_t firstElementIndex, std::uint32_t compressionSettings,
   const RClusterDescriptor::RPageRange &pageRange)
{
   if (physicalId != pageRange.fPhysicalColumnId)
      return R__FAIL("column ID mismatch");
   if (fCluster.fColumnRanges.count(physicalId) > 0)
      return R__FAIL("column ID conflict");
   RClusterDescriptor::RColumnRange columnRange{physicalId, firstElementIndex, ClusterSize_t{0}};
   columnRange.fCompressionSettings = compressionSettings;
   for (const auto &pi : pageRange.fPageInfos) {
      columnRange.fNElements += pi.fNElements;
   }
   fCluster.fPageRanges[physicalId] = pageRange.Clone();
   fCluster.fColumnRanges[physicalId] = columnRange;
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::Internal::RClusterDescriptorBuilder::MarkSuppressedColumnRange(DescriptorId_t physicalId)
{
   if (fCluster.fColumnRanges.count(physicalId) > 0)
      return R__FAIL("column ID conflict");

   RClusterDescriptor::RColumnRange columnRange;
   columnRange.fPhysicalColumnId = physicalId;
   columnRange.fCompressionSettings = kUnknownCompressionSettings;
   columnRange.fIsSuppressed = true;
   fCluster.fColumnRanges[physicalId] = columnRange;
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::Internal::RClusterDescriptorBuilder::CommitSuppressedColumnRanges(const RNTupleDescriptor &desc)
{
   for (auto &[_, columnRange] : fCluster.fColumnRanges) {
      if (!columnRange.fIsSuppressed)
         continue;
      R__ASSERT(columnRange.fFirstElementIndex == kInvalidNTupleIndex);

      const auto &columnDesc = desc.GetColumnDescriptor(columnRange.fPhysicalColumnId);
      const auto &fieldDesc = desc.GetFieldDescriptor(columnDesc.GetFieldId());
      // We expect only few columns and column representations per field, so we do a linear search
      for (const auto otherColumnLogicalId : fieldDesc.GetLogicalColumnIds()) {
         const auto &otherColumnDesc = desc.GetColumnDescriptor(otherColumnLogicalId);
         if (otherColumnDesc.GetRepresentationIndex() == columnDesc.GetRepresentationIndex())
            continue;
         if (otherColumnDesc.GetIndex() != columnDesc.GetIndex())
            continue;

         // Found corresponding column of a different column representation
         const auto &otherColumnRange = fCluster.GetColumnRange(otherColumnDesc.GetPhysicalId());
         if (otherColumnRange.fIsSuppressed)
            continue;

         columnRange.fFirstElementIndex = otherColumnRange.fFirstElementIndex;
         columnRange.fNElements = otherColumnRange.fNElements;
         break;
      }

      if (columnRange.fFirstElementIndex == kInvalidNTupleIndex) {
         return R__FAIL(std::string("cannot find non-suppressed column for column ID ") +
                        std::to_string(columnRange.fPhysicalColumnId) +
                        ", cluster ID: " + std::to_string(fCluster.GetId()));
      }
   }
   return RResult<void>::Success();
}

ROOT::Experimental::Internal::RClusterDescriptorBuilder &
ROOT::Experimental::Internal::RClusterDescriptorBuilder::AddExtendedColumnRanges(const RNTupleDescriptor &desc)
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

   // Extended columns can only be part of the header extension
   if (!desc.GetHeaderExtension())
      return *this;

   // Ensure that all columns in the header extension have their associated `R(Column|Page)Range`
   // Extended columns can be attached both to fields of the regular header and to fields of the extension header
   for (const auto &topLevelField : desc.GetTopLevelFields()) {
      fnTraverseSubtree(
         topLevelField.GetId(), std::max(topLevelField.GetNRepetitions(), std::uint64_t{1U}),
         [&](DescriptorId_t fieldId, std::uint64_t nRepetitions) {
            for (const auto &c : desc.GetColumnIterable(fieldId)) {
               const DescriptorId_t physicalId = c.GetPhysicalId();
               auto &columnRange = fCluster.fColumnRanges[physicalId];

               // Initialize a RColumnRange for `physicalId` if it was not there. Columns that were created during model
               // extension won't have on-disk metadata for the clusters that were already committed before the model
               // was extended. Therefore, these need to be synthetically initialized upon reading.
               if (columnRange.fPhysicalColumnId == kInvalidDescriptorId) {
                  columnRange.fPhysicalColumnId = physicalId;
                  columnRange.fFirstElementIndex = 0;
                  columnRange.fNElements = 0;
                  columnRange.fIsSuppressed = c.IsSuppressedDeferredColumn();
               }
               // Fixup the RColumnRange and RPageRange in deferred columns. We know what the first element index and
               // number of elements should have been if the column was not deferred; fix those and let
               // `ExtendToFitColumnRange()` synthesize RPageInfos accordingly.
               // Note that a deferred column (i.e, whose first element index is > 0) already met the criteria of
               // `RFieldBase::EntryToColumnElementIndex()`, i.e. it is a principal column reachable from the field zero
               // excluding subfields of collection and variant fields.
               if (c.IsDeferredColumn()) {
                  columnRange.fFirstElementIndex = fCluster.GetFirstEntryIndex() * nRepetitions;
                  columnRange.fNElements = fCluster.GetNEntries() * nRepetitions;
                  if (!columnRange.fIsSuppressed) {
                     auto &pageRange = fCluster.fPageRanges[physicalId];
                     pageRange.fPhysicalColumnId = physicalId;
                     const auto element = Internal::RColumnElementBase::Generate<void>(c.GetType());
                     pageRange.ExtendToFitColumnRange(columnRange, *element, Internal::RPage::kPageZeroSize);
                  }
               } else if (!columnRange.fIsSuppressed) {
                  fCluster.fPageRanges[physicalId].fPhysicalColumnId = physicalId;
               }
            }
         },
         fnTraverseSubtree);
   }
   return *this;
}

ROOT::Experimental::RResult<ROOT::Experimental::RClusterDescriptor>
ROOT::Experimental::Internal::RClusterDescriptorBuilder::MoveDescriptor()
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
   RClusterDescriptor result;
   std::swap(result, fCluster);
   return result;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::Internal::RClusterGroupDescriptorBuilder
ROOT::Experimental::Internal::RClusterGroupDescriptorBuilder::FromSummary(
   const RClusterGroupDescriptor &clusterGroupDesc)
{
   RClusterGroupDescriptorBuilder builder;
   builder.ClusterGroupId(clusterGroupDesc.GetId())
      .PageListLocator(clusterGroupDesc.GetPageListLocator())
      .PageListLength(clusterGroupDesc.GetPageListLength())
      .MinEntry(clusterGroupDesc.GetMinEntry())
      .EntrySpan(clusterGroupDesc.GetEntrySpan())
      .NClusters(clusterGroupDesc.GetNClusters());
   return builder;
}

ROOT::Experimental::RResult<ROOT::Experimental::RClusterGroupDescriptor>
ROOT::Experimental::Internal::RClusterGroupDescriptorBuilder::MoveDescriptor()
{
   if (fClusterGroup.fClusterGroupId == kInvalidDescriptorId)
      return R__FAIL("unset cluster group ID");
   RClusterGroupDescriptor result;
   std::swap(result, fClusterGroup);
   return result;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::RResult<ROOT::Experimental::RExtraTypeInfoDescriptor>
ROOT::Experimental::Internal::RExtraTypeInfoDescriptorBuilder::MoveDescriptor()
{
   if (fExtraTypeInfo.fContentId == EExtraTypeInfoIds::kInvalid)
      throw RException(R__FAIL("invalid extra type info content id"));
   RExtraTypeInfoDescriptor result;
   std::swap(result, fExtraTypeInfo);
   return result;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::RResult<void>
ROOT::Experimental::Internal::RNTupleDescriptorBuilder::EnsureFieldExists(DescriptorId_t fieldId) const
{
   if (fDescriptor.fFieldDescriptors.count(fieldId) == 0)
      return R__FAIL("field with id '" + std::to_string(fieldId) + "' doesn't exist");
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void> ROOT::Experimental::Internal::RNTupleDescriptorBuilder::EnsureValidDescriptor() const
{
   // Reuse field name validity check
   auto validName = RFieldBase::EnsureValidFieldName(fDescriptor.GetName());
   if (!validName) {
      return R__FORWARD_ERROR(validName);
   }

   for (const auto &[fieldId, fieldDesc] : fDescriptor.fFieldDescriptors) {
      // parent not properly set?
      if (fieldId != fDescriptor.GetFieldZeroId() && fieldDesc.GetParentId() == kInvalidDescriptorId) {
         return R__FAIL("field with id '" + std::to_string(fieldId) + "' has an invalid parent id");
      }

      // Same number of columns in every column representation?
      const auto columnCardinality = fieldDesc.GetColumnCardinality();
      if (columnCardinality == 0)
         continue;

      // In AddColumn, we already checked that all but the last representation are complete.
      // Check that the last column representation is complete, i.e. has all columns.
      const auto &logicalColumnIds = fieldDesc.GetLogicalColumnIds();
      const auto nColumns = logicalColumnIds.size();
      // If we have only a single column representation, the following condition is true by construction
      if ((nColumns + 1) == columnCardinality)
         continue;

      const auto &lastColumn = fDescriptor.GetColumnDescriptor(logicalColumnIds.back());
      if (lastColumn.GetIndex() + 1 != columnCardinality)
         return R__FAIL("field with id '" + std::to_string(fieldId) + "' has incomplete column representations");
   }

   return RResult<void>::Success();
}

ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Internal::RNTupleDescriptorBuilder::MoveDescriptor()
{
   EnsureValidDescriptor().ThrowOnError();
   RNTupleDescriptor result;
   std::swap(result, fDescriptor);
   return result;
}

void ROOT::Experimental::Internal::RNTupleDescriptorBuilder::SetNTuple(const std::string_view name,
                                                                       const std::string_view description)
{
   fDescriptor.fName = std::string(name);
   fDescriptor.fDescription = std::string(description);
}

void ROOT::Experimental::Internal::RNTupleDescriptorBuilder::SetFeature(unsigned int flag)
{
   if (flag % 64 == 0)
      throw RException(R__FAIL("invalid feature flag: " + std::to_string(flag)));
   fDescriptor.fFeatureFlags.insert(flag);
}

ROOT::Experimental::RResult<ROOT::Experimental::RColumnDescriptor>
ROOT::Experimental::Internal::RColumnDescriptorBuilder::MakeDescriptor() const
{
   if (fColumn.GetLogicalId() == kInvalidDescriptorId)
      return R__FAIL("invalid logical column id");
   if (fColumn.GetPhysicalId() == kInvalidDescriptorId)
      return R__FAIL("invalid physical column id");
   if (fColumn.GetFieldId() == kInvalidDescriptorId)
      return R__FAIL("invalid field id, dangling column");

   // NOTE: if the column type is unknown we don't want to fail, as we might be reading an RNTuple
   // created with a future version of ROOT. In this case we just skip the valid bit range check,
   // as we have no idea what the valid range is.
   // In general, reading the metadata of an unknown column is fine, it becomes an error only when
   // we try to read the actual data contained in it.
   if (fColumn.GetType() != EColumnType::kUnknown) {
      const auto [minBits, maxBits] = RColumnElementBase::GetValidBitRange(fColumn.GetType());
      if (fColumn.GetBitsOnStorage() < minBits || fColumn.GetBitsOnStorage() > maxBits)
         return R__FAIL("invalid column bit width");
   }

   return fColumn.Clone();
}

ROOT::Experimental::Internal::RFieldDescriptorBuilder::RFieldDescriptorBuilder(const RFieldDescriptor &fieldDesc)
   : fField(fieldDesc.Clone())
{
   fField.fParentId = kInvalidDescriptorId;
   fField.fLinkIds = {};
   fField.fLogicalColumnIds = {};
}

ROOT::Experimental::Internal::RFieldDescriptorBuilder
ROOT::Experimental::Internal::RFieldDescriptorBuilder::FromField(const RFieldBase &field)
{
   RFieldDescriptorBuilder fieldDesc;
   fieldDesc.FieldVersion(field.GetFieldVersion())
      .TypeVersion(field.GetTypeVersion())
      .FieldName(field.GetFieldName())
      .FieldDescription(field.GetDescription())
      .TypeName(field.GetTypeName())
      .TypeAlias(field.GetTypeAlias())
      .Structure(field.GetStructure())
      .NRepetitions(field.GetNRepetitions());
   if (field.GetTraits() & RFieldBase::kTraitTypeChecksum)
      fieldDesc.TypeChecksum(field.GetTypeChecksum());
   return fieldDesc;
}

ROOT::Experimental::RResult<ROOT::Experimental::RFieldDescriptor>
ROOT::Experimental::Internal::RFieldDescriptorBuilder::MakeDescriptor() const
{
   if (fField.GetId() == kInvalidDescriptorId) {
      return R__FAIL("invalid field id");
   }
   if (fField.GetStructure() == ENTupleStructure::kInvalid) {
      return R__FAIL("invalid field structure");
   }
   // FieldZero is usually named "" and would be a false positive here
   if (fField.GetParentId() != kInvalidDescriptorId) {
      auto validName = RFieldBase::EnsureValidFieldName(fField.GetFieldName());
      if (!validName) {
         return R__FORWARD_ERROR(validName);
      }
   }
   return fField.Clone();
}

void ROOT::Experimental::Internal::RNTupleDescriptorBuilder::AddField(const RFieldDescriptor &fieldDesc)
{
   fDescriptor.fFieldDescriptors.emplace(fieldDesc.GetId(), fieldDesc.Clone());
   if (fDescriptor.fHeaderExtension)
      fDescriptor.fHeaderExtension->AddExtendedField(fieldDesc);
   if (fieldDesc.GetFieldName().empty() && fieldDesc.GetParentId() == kInvalidDescriptorId) {
      fDescriptor.fFieldZeroId = fieldDesc.GetId();
   }
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::Internal::RNTupleDescriptorBuilder::AddFieldLink(DescriptorId_t fieldId, DescriptorId_t linkId)
{
   auto fieldExists = RResult<void>::Success();
   if (!(fieldExists = EnsureFieldExists(fieldId)))
      return R__FORWARD_ERROR(fieldExists);
   if (!(fieldExists = EnsureFieldExists(linkId)))
      return R__FAIL("child field with id '" + std::to_string(linkId) + "' doesn't exist in NTuple");

   if (linkId == fDescriptor.GetFieldZeroId()) {
      return R__FAIL("cannot make FieldZero a child field");
   }
   // fail if field already has another valid parent
   auto parentId = fDescriptor.fFieldDescriptors.at(linkId).GetParentId();
   if ((parentId != kInvalidDescriptorId) && (parentId != fieldId)) {
      return R__FAIL("field '" + std::to_string(linkId) + "' already has a parent ('" + std::to_string(parentId) + ")");
   }
   if (fieldId == linkId) {
      return R__FAIL("cannot make field '" + std::to_string(fieldId) + "' a child of itself");
   }
   fDescriptor.fFieldDescriptors.at(linkId).fParentId = fieldId;
   fDescriptor.fFieldDescriptors.at(fieldId).fLinkIds.push_back(linkId);
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::Internal::RNTupleDescriptorBuilder::AddFieldProjection(DescriptorId_t sourceId,
                                                                           DescriptorId_t targetId)
{
   auto fieldExists = RResult<void>::Success();
   if (!(fieldExists = EnsureFieldExists(sourceId)))
      return R__FORWARD_ERROR(fieldExists);
   if (!(fieldExists = EnsureFieldExists(targetId)))
      return R__FAIL("projected field with id '" + std::to_string(targetId) + "' doesn't exist in NTuple");

   if (targetId == fDescriptor.GetFieldZeroId()) {
      return R__FAIL("cannot make FieldZero a projected field");
   }
   if (sourceId == targetId) {
      return R__FAIL("cannot make field '" + std::to_string(targetId) + "' a projection of itself");
   }
   if (fDescriptor.fFieldDescriptors.at(sourceId).IsProjectedField()) {
      return R__FAIL("cannot make field '" + std::to_string(targetId) + "' a projection of an already projected field");
   }
   // fail if target field already has another valid projection source
   auto &targetDesc = fDescriptor.fFieldDescriptors.at(targetId);
   if (targetDesc.IsProjectedField() && targetDesc.GetProjectionSourceId() != sourceId) {
      return R__FAIL("field '" + std::to_string(targetId) + "' has already a projection source ('" +
                     std::to_string(targetDesc.GetProjectionSourceId()) + ")");
   }
   fDescriptor.fFieldDescriptors.at(targetId).fProjectionSourceId = sourceId;
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::Internal::RNTupleDescriptorBuilder::AddColumn(RColumnDescriptor &&columnDesc)
{
   const auto fieldId = columnDesc.GetFieldId();
   const auto columnIndex = columnDesc.GetIndex();
   const auto representationIndex = columnDesc.GetRepresentationIndex();

   auto fieldExists = EnsureFieldExists(fieldId);
   if (!fieldExists) {
      return R__FORWARD_ERROR(fieldExists);
   }
   auto &fieldDesc = fDescriptor.fFieldDescriptors.find(fieldId)->second;

   if (columnDesc.IsAliasColumn()) {
      if (columnDesc.GetType() != fDescriptor.GetColumnDescriptor(columnDesc.GetPhysicalId()).GetType())
         return R__FAIL("alias column type mismatch");
   }
   if (fDescriptor.FindLogicalColumnId(fieldId, columnIndex, representationIndex) != kInvalidDescriptorId) {
      return R__FAIL("column index clash");
   }
   if (columnIndex > 0) {
      if (fDescriptor.FindLogicalColumnId(fieldId, columnIndex - 1, representationIndex) == kInvalidDescriptorId)
         return R__FAIL("out of bounds column index");
   }
   if (representationIndex > 0) {
      if (fDescriptor.FindLogicalColumnId(fieldId, 0, representationIndex - 1) == kInvalidDescriptorId) {
         return R__FAIL("out of bounds representation index");
      }
      if (columnIndex == 0) {
         assert(fieldDesc.fColumnCardinality > 0);
         if (fDescriptor.FindLogicalColumnId(fieldId, fieldDesc.fColumnCardinality - 1, representationIndex - 1) ==
             kInvalidDescriptorId) {
            return R__FAIL("incomplete column representations");
         }
      } else {
         if (columnIndex >= fieldDesc.fColumnCardinality)
            return R__FAIL("irregular column representations");
      }
   } else {
      // This will set the column cardinality to the number of columns of the first representation
      fieldDesc.fColumnCardinality = columnIndex + 1;
   }

   const auto logicalId = columnDesc.GetLogicalId();
   fieldDesc.fLogicalColumnIds.emplace_back(logicalId);

   if (!columnDesc.IsAliasColumn())
      fDescriptor.fNPhysicalColumns++;
   fDescriptor.fColumnDescriptors.emplace(logicalId, std::move(columnDesc));
   if (fDescriptor.fHeaderExtension)
      fDescriptor.fHeaderExtension->AddExtendedColumn(columnDesc);

   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::Internal::RNTupleDescriptorBuilder::AddClusterGroup(RClusterGroupDescriptor &&clusterGroup)
{
   const auto id = clusterGroup.GetId();
   if (fDescriptor.fClusterGroupDescriptors.count(id) > 0)
      return R__FAIL("cluster group id clash");
   fDescriptor.fNEntries = std::max(fDescriptor.fNEntries, clusterGroup.GetMinEntry() + clusterGroup.GetEntrySpan());
   fDescriptor.fNClusters += clusterGroup.GetNClusters();
   fDescriptor.fClusterGroupDescriptors.emplace(id, std::move(clusterGroup));
   return RResult<void>::Success();
}

void ROOT::Experimental::Internal::RNTupleDescriptorBuilder::Reset()
{
   fDescriptor.fName = "";
   fDescriptor.fDescription = "";
   fDescriptor.fFieldDescriptors.clear();
   fDescriptor.fColumnDescriptors.clear();
   fDescriptor.fClusterDescriptors.clear();
   fDescriptor.fClusterGroupDescriptors.clear();
   fDescriptor.fHeaderExtension.reset();
}

void ROOT::Experimental::Internal::RNTupleDescriptorBuilder::BeginHeaderExtension()
{
   if (!fDescriptor.fHeaderExtension)
      fDescriptor.fHeaderExtension = std::make_unique<RNTupleDescriptor::RHeaderExtension>();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::Internal::RNTupleDescriptorBuilder::AddCluster(RClusterDescriptor &&clusterDesc)
{
   auto clusterId = clusterDesc.GetId();
   if (fDescriptor.fClusterDescriptors.count(clusterId) > 0)
      return R__FAIL("cluster id clash");
   fDescriptor.fClusterDescriptors.emplace(clusterId, std::move(clusterDesc));
   return RResult<void>::Success();
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::Internal::RNTupleDescriptorBuilder::AddExtraTypeInfo(RExtraTypeInfoDescriptor &&extraTypeInfoDesc)
{
   // Make sure we have no duplicates
   if (std::find(fDescriptor.fExtraTypeInfoDescriptors.begin(), fDescriptor.fExtraTypeInfoDescriptors.end(),
                 extraTypeInfoDesc) != fDescriptor.fExtraTypeInfoDescriptors.end()) {
      return R__FAIL("extra type info duplicates");
   }
   fDescriptor.fExtraTypeInfoDescriptors.emplace_back(std::move(extraTypeInfoDesc));
   return RResult<void>::Success();
}

ROOT::Experimental::Internal::RNTupleSerializer::StreamerInfoMap_t
ROOT::Experimental::Internal::RNTupleDescriptorBuilder::BuildStreamerInfos() const
{
   RNTupleSerializer::StreamerInfoMap_t streamerInfoMap;
   const auto &desc = GetDescriptor();

   std::function<void(const RFieldDescriptor &)> fnWalkFieldTree;
   fnWalkFieldTree = [&desc, &streamerInfoMap, &fnWalkFieldTree](const RFieldDescriptor &fieldDesc) {
      if (fieldDesc.IsCustomClass()) {
         // Add streamer info for this class to streamerInfoMap
         auto cl = TClass::GetClass(fieldDesc.GetTypeName().c_str());
         if (!cl) {
            throw RException(R__FAIL(std::string("cannot get TClass for ") + fieldDesc.GetTypeName()));
         }
         auto streamerInfo = cl->GetStreamerInfo(fieldDesc.GetTypeVersion());
         if (!streamerInfo) {
            throw RException(R__FAIL(std::string("cannot get streamerInfo for ") + fieldDesc.GetTypeName()));
         }
         streamerInfoMap[streamerInfo->GetNumber()] = streamerInfo;
      }

      // Recursively traverse sub fields
      for (const auto &subFieldDesc : desc.GetFieldIterable(fieldDesc)) {
         fnWalkFieldTree(subFieldDesc);
      }
   };

   fnWalkFieldTree(desc.GetFieldZero());

   // Add the streamer info records from unsplit fields: because of runtime polymorphism we may need to add additional
   // types not covered by the type names stored in the field headers
   for (const auto &extraTypeInfo : desc.GetExtraTypeInfoIterable()) {
      if (extraTypeInfo.GetContentId() != EExtraTypeInfoIds::kStreamerInfo)
         continue;
      // Ideally, we would avoid deserializing the streamer info records of the unsplit fields that we just serialized.
      // However, this happens only once at the end of writing and only when unsplit fields are used, so the
      // preference here is for code simplicity.
      streamerInfoMap.merge(RNTupleSerializer::DeserializeStreamerInfos(extraTypeInfo.GetContent()).Unwrap());
   }

   return streamerInfoMap;
}

ROOT::Experimental::RClusterDescriptor::RColumnRangeIterable
ROOT::Experimental::RClusterDescriptor::GetColumnRangeIterable() const
{
   return RColumnRangeIterable(*this);
}

ROOT::Experimental::RNTupleDescriptor::RFieldDescriptorIterable
ROOT::Experimental::RNTupleDescriptor::GetFieldIterable(const RFieldDescriptor &fieldDesc) const
{
   return RFieldDescriptorIterable(*this, fieldDesc);
}

ROOT::Experimental::RNTupleDescriptor::RFieldDescriptorIterable ROOT::Experimental::RNTupleDescriptor::GetFieldIterable(
   const RFieldDescriptor &fieldDesc, const std::function<bool(DescriptorId_t, DescriptorId_t)> &comparator) const
{
   return RFieldDescriptorIterable(*this, fieldDesc, comparator);
}

ROOT::Experimental::RNTupleDescriptor::RFieldDescriptorIterable
ROOT::Experimental::RNTupleDescriptor::GetFieldIterable(DescriptorId_t fieldId) const
{
   return GetFieldIterable(GetFieldDescriptor(fieldId));
}

ROOT::Experimental::RNTupleDescriptor::RFieldDescriptorIterable ROOT::Experimental::RNTupleDescriptor::GetFieldIterable(
   DescriptorId_t fieldId, const std::function<bool(DescriptorId_t, DescriptorId_t)> &comparator) const
{
   return GetFieldIterable(GetFieldDescriptor(fieldId), comparator);
}

ROOT::Experimental::RNTupleDescriptor::RFieldDescriptorIterable
ROOT::Experimental::RNTupleDescriptor::GetTopLevelFields() const
{
   return GetFieldIterable(GetFieldZeroId());
}

ROOT::Experimental::RNTupleDescriptor::RFieldDescriptorIterable
ROOT::Experimental::RNTupleDescriptor::GetTopLevelFields(
   const std::function<bool(DescriptorId_t, DescriptorId_t)> &comparator) const
{
   return GetFieldIterable(GetFieldZeroId(), comparator);
}

ROOT::Experimental::RNTupleDescriptor::RColumnDescriptorIterable
ROOT::Experimental::RNTupleDescriptor::GetColumnIterable() const
{
   return RColumnDescriptorIterable(*this);
}

ROOT::Experimental::RNTupleDescriptor::RColumnDescriptorIterable
ROOT::Experimental::RNTupleDescriptor::GetColumnIterable(const RFieldDescriptor &fieldDesc) const
{
   return RColumnDescriptorIterable(*this, fieldDesc);
}

ROOT::Experimental::RNTupleDescriptor::RColumnDescriptorIterable
ROOT::Experimental::RNTupleDescriptor::GetColumnIterable(DescriptorId_t fieldId) const
{
   return RColumnDescriptorIterable(*this, GetFieldDescriptor(fieldId));
}

ROOT::Experimental::RNTupleDescriptor::RClusterGroupDescriptorIterable
ROOT::Experimental::RNTupleDescriptor::GetClusterGroupIterable() const
{
   return RClusterGroupDescriptorIterable(*this);
}

ROOT::Experimental::RNTupleDescriptor::RClusterDescriptorIterable
ROOT::Experimental::RNTupleDescriptor::GetClusterIterable() const
{
   return RClusterDescriptorIterable(*this);
}

ROOT::Experimental::RNTupleDescriptor::RExtraTypeInfoDescriptorIterable
ROOT::Experimental::RNTupleDescriptor::GetExtraTypeInfoIterable() const
{
   return RExtraTypeInfoDescriptorIterable(*this);
}
