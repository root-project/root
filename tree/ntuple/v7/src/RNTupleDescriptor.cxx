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
ROOT::Experimental::RFieldDescriptor::CreateField(const RNTupleDescriptor &ntplDesc,
                                                  const ROOT::RCreateFieldOptions &options) const
{
   if (GetStructure() == ROOT::ENTupleStructure::kStreamer) {
      auto streamerField = std::make_unique<RStreamerField>(GetFieldName(), GetTypeName());
      streamerField->SetOnDiskId(fFieldId);
      return streamerField;
   }

   // The structure may be unknown if the descriptor comes from a deserialized field with an unknown structural role.
   // For forward compatibility, we allow this case and return an InvalidField.
   if (GetStructure() == ROOT::ENTupleStructure::kUnknown) {
      if (options.GetReturnInvalidOnError()) {
         auto invalidField = std::make_unique<RInvalidField>(GetFieldName(), GetTypeName(), "",
                                                             RInvalidField::RCategory::kUnknownStructure);
         invalidField->SetOnDiskId(fFieldId);
         return invalidField;
      } else {
         throw RException(R__FAIL("unexpected on-disk field structure value for field \"" + GetFieldName() + "\""));
      }
   }

   if (GetTypeName().empty()) {
      switch (GetStructure()) {
      case ROOT::ENTupleStructure::kRecord: {
         std::vector<std::unique_ptr<RFieldBase>> memberFields;
         memberFields.reserve(fLinkIds.size());
         for (auto id : fLinkIds) {
            const auto &memberDesc = ntplDesc.GetFieldDescriptor(id);
            auto field = memberDesc.CreateField(ntplDesc, options);
            if (field->GetTraits() & RFieldBase::kTraitInvalidField)
               return field;
            memberFields.emplace_back(std::move(field));
         }
         auto recordField = std::make_unique<RRecordField>(GetFieldName(), std::move(memberFields));
         recordField->SetOnDiskId(fFieldId);
         return recordField;
      }
      case ROOT::ENTupleStructure::kCollection: {
         if (fLinkIds.size() != 1) {
            throw RException(R__FAIL("unsupported untyped collection for field \"" + GetFieldName() + "\""));
         }
         auto itemField = ntplDesc.GetFieldDescriptor(fLinkIds[0]).CreateField(ntplDesc, options);
         if (itemField->GetTraits() & RFieldBase::kTraitInvalidField)
            return itemField;
         auto collectionField = RVectorField::CreateUntyped(GetFieldName(), std::move(itemField));
         collectionField->SetOnDiskId(fFieldId);
         return collectionField;
      }
      default: throw RException(R__FAIL("unsupported untyped field structure for field \"" + GetFieldName() + "\""));
      }
   }

   try {
      const auto &fieldName = GetFieldName();
      const auto &typeName = GetTypeAlias().empty() ? GetTypeName() : GetTypeAlias();
      // NOTE: Unwrap() here may throw an exception, hence the try block.
      // If options.fReturnInvalidOnError is false we just rethrow it, otherwise we return an InvalidField wrapping the
      // error.
      auto field = Internal::CallFieldBaseCreate(fieldName, typeName, options, &ntplDesc, fFieldId).Unwrap();
      field->SetOnDiskId(fFieldId);

      for (auto &subfield : *field) {
         const auto subfieldId = ntplDesc.FindFieldId(subfield.GetFieldName(), subfield.GetParent()->GetOnDiskId());
         subfield.SetOnDiskId(subfieldId);
         if (subfield.GetTraits() & RFieldBase::kTraitInvalidField) {
            auto &invalidField = static_cast<RInvalidField &>(subfield);
            // A subfield being invalid "infects" its entire ancestry.
            return invalidField.Clone(fieldName);
         }
      }

      return field;
   } catch (RException &ex) {
      if (options.GetReturnInvalidOnError())
         return std::make_unique<RInvalidField>(GetFieldName(), GetTypeName(), ex.GetError().GetReport(),
                                                RInvalidField::RCategory::kGeneric);
      else
         throw ex;
   }
}

bool ROOT::Experimental::RFieldDescriptor::IsCustomClass() const
{
   if (fStructure != ROOT::ENTupleStructure::kRecord && fStructure != ROOT::ENTupleStructure::kStreamer)
      return false;

   // Skip untyped structs
   if (fTypeName.empty())
      return false;

   if (fStructure == ROOT::ENTupleStructure::kRecord) {
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
ROOT::Experimental::RClusterDescriptor::RPageRange::Find(ROOT::NTupleSize_t idxInCluster) const
{
   const auto N = fCumulativeNElements.size();
   R__ASSERT(N > 0);
   R__ASSERT(N == fPageInfos.size());

   std::size_t left = 0;
   std::size_t right = N - 1;
   std::size_t midpoint = N;
   while (left <= right) {
      midpoint = (left + right) / 2;
      if (fCumulativeNElements[midpoint] <= idxInCluster) {
         left = midpoint + 1;
         continue;
      }

      if ((midpoint == 0) || (fCumulativeNElements[midpoint - 1] <= idxInCluster))
         break;

      right = midpoint - 1;
   }
   R__ASSERT(midpoint < N);

   auto pageInfo = fPageInfos[midpoint];
   decltype(idxInCluster) firstInPage = (midpoint == 0) ? 0 : fCumulativeNElements[midpoint - 1];
   R__ASSERT(firstInPage <= idxInCluster);
   R__ASSERT((firstInPage + pageInfo.GetNElements()) > idxInCluster);
   return RPageInfoExtended{pageInfo, firstInPage, midpoint};
}

std::size_t
ROOT::Experimental::RClusterDescriptor::RPageRange::ExtendToFitColumnRange(const RColumnRange &columnRange,
                                                                           const Internal::RColumnElementBase &element,
                                                                           std::size_t pageSize)
{
   R__ASSERT(fPhysicalColumnId == columnRange.GetPhysicalColumnId());
   R__ASSERT(!columnRange.IsSuppressed());

   const auto nElements =
      std::accumulate(fPageInfos.begin(), fPageInfos.end(), 0U,
                      [](std::size_t n, const auto &pageInfo) { return n + pageInfo.GetNElements(); });
   const auto nElementsRequired = static_cast<std::uint64_t>(columnRange.GetNElements());

   if (nElementsRequired == nElements)
      return 0U;
   R__ASSERT((nElementsRequired > nElements) && "invalid attempt to shrink RPageRange");

   std::vector<RPageInfo> pageInfos;
   // Synthesize new `RPageInfo`s as needed
   const std::uint64_t nElementsPerPage = pageSize / element.GetSize();
   R__ASSERT(nElementsPerPage > 0);
   for (auto nRemainingElements = nElementsRequired - nElements; nRemainingElements > 0;) {
      RPageInfo pageInfo;
      pageInfo.SetNElements(std::min(nElementsPerPage, nRemainingElements));
      RNTupleLocator locator;
      locator.SetType(RNTupleLocator::kTypePageZero);
      locator.SetNBytesOnStorage(element.GetPackedSize(pageInfo.GetNElements()));
      pageInfo.SetLocator(locator);
      pageInfos.emplace_back(pageInfo);
      nRemainingElements -= pageInfo.GetNElements();
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

std::uint64_t ROOT::Experimental::RClusterDescriptor::GetNBytesOnStorage() const
{
   std::uint64_t nbytes = 0;
   for (const auto &pr : fPageRanges) {
      for (const auto &pi : pr.second.GetPageInfos()) {
         nbytes += pi.GetLocator().GetNBytesOnStorage();
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
   return fContentId == other.fContentId && fTypeName == other.fTypeName && fTypeVersion == other.fTypeVersion;
}

ROOT::Experimental::RExtraTypeInfoDescriptor ROOT::Experimental::RExtraTypeInfoDescriptor::Clone() const
{
   RExtraTypeInfoDescriptor clone;
   clone.fContentId = fContentId;
   clone.fTypeVersion = fTypeVersion;
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

ROOT::NTupleSize_t ROOT::Experimental::RNTupleDescriptor::GetNElements(ROOT::DescriptorId_t physicalColumnId) const
{
   ROOT::NTupleSize_t result = 0;
   for (const auto &cd : fClusterDescriptors) {
      if (!cd.second.ContainsColumn(physicalColumnId))
         continue;
      auto columnRange = cd.second.GetColumnRange(physicalColumnId);
      result = std::max(result, columnRange.GetFirstElementIndex() + columnRange.GetNElements());
   }
   return result;
}

ROOT::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::FindFieldId(std::string_view fieldName, ROOT::DescriptorId_t parentId) const
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
      return ROOT::kInvalidDescriptorId;
   for (const auto linkId : itrFieldDesc->second.GetLinkIds()) {
      if (fFieldDescriptors.at(linkId).GetFieldName() == leafName)
         return linkId;
   }
   return ROOT::kInvalidDescriptorId;
}

std::string ROOT::Experimental::RNTupleDescriptor::GetQualifiedFieldName(ROOT::DescriptorId_t fieldId) const
{
   if (fieldId == ROOT::kInvalidDescriptorId)
      return "";

   const auto &fieldDescriptor = fFieldDescriptors.at(fieldId);
   auto prefix = GetQualifiedFieldName(fieldDescriptor.GetParentId());
   if (prefix.empty())
      return fieldDescriptor.GetFieldName();
   return prefix + "." + fieldDescriptor.GetFieldName();
}

ROOT::DescriptorId_t ROOT::Experimental::RNTupleDescriptor::FindFieldId(std::string_view fieldName) const
{
   return FindFieldId(fieldName, GetFieldZeroId());
}

ROOT::DescriptorId_t ROOT::Experimental::RNTupleDescriptor::FindLogicalColumnId(ROOT::DescriptorId_t fieldId,
                                                                                std::uint32_t columnIndex,
                                                                                std::uint16_t representationIndex) const
{
   auto itr = fFieldDescriptors.find(fieldId);
   if (itr == fFieldDescriptors.cend())
      return ROOT::kInvalidDescriptorId;
   if (columnIndex >= itr->second.GetColumnCardinality())
      return ROOT::kInvalidDescriptorId;
   const auto idx = representationIndex * itr->second.GetColumnCardinality() + columnIndex;
   if (itr->second.GetLogicalColumnIds().size() <= idx)
      return ROOT::kInvalidDescriptorId;
   return itr->second.GetLogicalColumnIds()[idx];
}

ROOT::DescriptorId_t
ROOT::Experimental::RNTupleDescriptor::FindPhysicalColumnId(ROOT::DescriptorId_t fieldId, std::uint32_t columnIndex,
                                                            std::uint16_t representationIndex) const
{
   auto logicalId = FindLogicalColumnId(fieldId, columnIndex, representationIndex);
   if (logicalId == ROOT::kInvalidDescriptorId)
      return ROOT::kInvalidDescriptorId;
   return GetColumnDescriptor(logicalId).GetPhysicalId();
}

ROOT::DescriptorId_t ROOT::Experimental::RNTupleDescriptor::FindClusterId(ROOT::DescriptorId_t physicalColumnId,
                                                                          ROOT::NTupleSize_t index) const
{
   if (GetNClusterGroups() == 0)
      return ROOT::kInvalidDescriptorId;

   // Binary search in the cluster group list, followed by a binary search in the clusters of that cluster group

   std::size_t cgLeft = 0;
   std::size_t cgRight = GetNClusterGroups() - 1;
   while (cgLeft <= cgRight) {
      const std::size_t cgMidpoint = (cgLeft + cgRight) / 2;
      const auto &clusterIds = GetClusterGroupDescriptor(fSortedClusterGroupIds[cgMidpoint]).GetClusterIds();
      R__ASSERT(!clusterIds.empty());

      const auto &clusterDesc = GetClusterDescriptor(clusterIds.front());
      // this may happen if the RNTuple has an empty schema
      if (!clusterDesc.ContainsColumn(physicalColumnId))
         return ROOT::kInvalidDescriptorId;

      const auto firstElementInGroup = clusterDesc.GetColumnRange(physicalColumnId).GetFirstElementIndex();
      if (firstElementInGroup > index) {
         // Look into the lower half of cluster groups
         R__ASSERT(cgMidpoint > 0);
         cgRight = cgMidpoint - 1;
         continue;
      }

      const auto &lastColumnRange = GetClusterDescriptor(clusterIds.back()).GetColumnRange(physicalColumnId);
      if ((lastColumnRange.GetFirstElementIndex() + lastColumnRange.GetNElements()) <= index) {
         // Look into the upper half of cluster groups
         cgLeft = cgMidpoint + 1;
         continue;
      }

      // Binary search in the current cluster group; since we already checked the element range boundaries,
      // the element must be in that cluster group.
      std::size_t clusterLeft = 0;
      std::size_t clusterRight = clusterIds.size() - 1;
      while (clusterLeft <= clusterRight) {
         const std::size_t clusterMidpoint = (clusterLeft + clusterRight) / 2;
         const auto clusterId = clusterIds[clusterMidpoint];
         const auto &columnRange = GetClusterDescriptor(clusterId).GetColumnRange(physicalColumnId);

         if (columnRange.Contains(index))
            return clusterId;

         if (columnRange.GetFirstElementIndex() > index) {
            R__ASSERT(clusterMidpoint > 0);
            clusterRight = clusterMidpoint - 1;
            continue;
         }

         if (columnRange.GetFirstElementIndex() + columnRange.GetNElements() <= index) {
            clusterLeft = clusterMidpoint + 1;
            continue;
         }
      }
      R__ASSERT(false);
   }
   return ROOT::kInvalidDescriptorId;
}

ROOT::DescriptorId_t ROOT::Experimental::RNTupleDescriptor::FindClusterId(ROOT::NTupleSize_t entryIdx) const
{
   if (GetNClusterGroups() == 0)
      return ROOT::kInvalidDescriptorId;

   // Binary search in the cluster group list, followed by a binary search in the clusters of that cluster group

   std::size_t cgLeft = 0;
   std::size_t cgRight = GetNClusterGroups() - 1;
   while (cgLeft <= cgRight) {
      const std::size_t cgMidpoint = (cgLeft + cgRight) / 2;
      const auto &cgDesc = GetClusterGroupDescriptor(fSortedClusterGroupIds[cgMidpoint]);

      if (cgDesc.GetMinEntry() > entryIdx) {
         R__ASSERT(cgMidpoint > 0);
         cgRight = cgMidpoint - 1;
         continue;
      }

      if (cgDesc.GetMinEntry() + cgDesc.GetEntrySpan() <= entryIdx) {
         cgLeft = cgMidpoint + 1;
         continue;
      }

      // Binary search in the current cluster group; since we already checked the element range boundaries,
      // the element must be in that cluster group.
      const auto &clusterIds = cgDesc.GetClusterIds();
      R__ASSERT(!clusterIds.empty());
      std::size_t clusterLeft = 0;
      std::size_t clusterRight = clusterIds.size() - 1;
      while (clusterLeft <= clusterRight) {
         const std::size_t clusterMidpoint = (clusterLeft + clusterRight) / 2;
         const auto &clusterDesc = GetClusterDescriptor(clusterIds[clusterMidpoint]);

         if (clusterDesc.GetFirstEntryIndex() > entryIdx) {
            R__ASSERT(clusterMidpoint > 0);
            clusterRight = clusterMidpoint - 1;
            continue;
         }

         if (clusterDesc.GetFirstEntryIndex() + clusterDesc.GetNEntries() <= entryIdx) {
            clusterLeft = clusterMidpoint + 1;
            continue;
         }

         return clusterIds[clusterMidpoint];
      }
      R__ASSERT(false);
   }
   return ROOT::kInvalidDescriptorId;
}

ROOT::DescriptorId_t ROOT::Experimental::RNTupleDescriptor::FindNextClusterId(ROOT::DescriptorId_t clusterId) const
{
   // TODO(jblomer): we may want to shortcut the common case and check if clusterId + 1 contains
   // firstEntryInNextCluster. This shortcut would currently always trigger. We do not want, however, to depend
   // on the linearity of the descriptor IDs, so we should only enable the shortcut if we can ensure that the
   // binary search code path remains tested.
   const auto &clusterDesc = GetClusterDescriptor(clusterId);
   const auto firstEntryInNextCluster = clusterDesc.GetFirstEntryIndex() + clusterDesc.GetNEntries();
   return FindClusterId(firstEntryInNextCluster);
}

ROOT::DescriptorId_t ROOT::Experimental::RNTupleDescriptor::FindPrevClusterId(ROOT::DescriptorId_t clusterId) const
{
   // TODO(jblomer): we may want to shortcut the common case and check if clusterId - 1 contains
   // firstEntryInNextCluster. This shortcut would currently always trigger. We do not want, however, to depend
   // on the linearity of the descriptor IDs, so we should only enable the shortcut if we can ensure that the
   // binary search code path remains tested.
   const auto &clusterDesc = GetClusterDescriptor(clusterId);
   if (clusterDesc.GetFirstEntryIndex() == 0)
      return ROOT::kInvalidDescriptorId;
   return FindClusterId(clusterDesc.GetFirstEntryIndex() - 1);
}

std::vector<ROOT::DescriptorId_t>
ROOT::Experimental::RNTupleDescriptor::RHeaderExtension::GetTopLevelFields(const RNTupleDescriptor &desc) const
{
   auto fieldZeroId = desc.GetFieldZeroId();

   std::vector<ROOT::DescriptorId_t> fields;
   for (const auto fieldId : fFieldIdsOrder) {
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
   std::deque<ROOT::DescriptorId_t> fieldIdQueue{ntuple.GetFieldZeroId()};

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

ROOT::RResult<void>
ROOT::Experimental::RNTupleDescriptor::AddClusterGroupDetails(ROOT::DescriptorId_t clusterGroupId,
                                                              std::vector<RClusterDescriptor> &clusterDescs)
{
   auto iter = fClusterGroupDescriptors.find(clusterGroupId);
   if (iter == fClusterGroupDescriptors.end())
      return R__FAIL("invalid attempt to add details of unknown cluster group");
   if (iter->second.HasClusterDetails())
      return R__FAIL("invalid attempt to re-populate cluster group details");
   if (iter->second.GetNClusters() != clusterDescs.size())
      return R__FAIL("mismatch of number of clusters");

   std::vector<ROOT::DescriptorId_t> clusterIds;
   for (unsigned i = 0; i < clusterDescs.size(); ++i) {
      clusterIds.emplace_back(clusterDescs[i].GetId());
      auto [_, success] = fClusterDescriptors.emplace(clusterIds.back(), std::move(clusterDescs[i]));
      if (!success) {
         return R__FAIL("invalid attempt to re-populate existing cluster");
      }
   }
   std::sort(clusterIds.begin(), clusterIds.end(), [this](ROOT::DescriptorId_t a, ROOT::DescriptorId_t b) {
      return fClusterDescriptors[a].GetFirstEntryIndex() < fClusterDescriptors[b].GetFirstEntryIndex();
   });
   auto cgBuilder = Internal::RClusterGroupDescriptorBuilder::FromSummary(iter->second);
   cgBuilder.AddSortedClusters(clusterIds);
   iter->second = cgBuilder.MoveDescriptor().Unwrap();
   return RResult<void>::Success();
}

ROOT::RResult<void> ROOT::Experimental::RNTupleDescriptor::DropClusterGroupDetails(ROOT::DescriptorId_t clusterGroupId)
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
   auto model = options.GetCreateBare() ? RNTupleModel::CreateBare(std::move(fieldZero))
                                        : RNTupleModel::Create(std::move(fieldZero));
   ROOT::RCreateFieldOptions createFieldOpts;
   createFieldOpts.SetReturnInvalidOnError(options.GetForwardCompatible());
   createFieldOpts.SetEmulateUnknownTypes(options.GetEmulateUnknownTypes());
   for (const auto &topDesc : GetTopLevelFields()) {
      auto field = topDesc.CreateField(*this, createFieldOpts);
      if (field->GetTraits() & RFieldBase::kTraitInvalidField)
         continue;

      if (options.GetReconstructProjections() && topDesc.IsProjectedField()) {
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

ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::RNTupleDescriptor::CloneSchema() const
{
   RNTupleDescriptor clone;
   clone.fName = fName;
   clone.fDescription = fDescription;
   clone.fNPhysicalColumns = fNPhysicalColumns;
   clone.fFieldZeroId = fFieldZeroId;
   clone.fFeatureFlags = fFeatureFlags;
   // OnDiskHeaderSize, OnDiskHeaderXxHash3 not copied because they may come from a merged header + extension header
   // and therefore not represent the actual sources's header.
   // OnDiskFooterSize not copied because it contains information beyond the schema, for example the clustering.

   for (const auto &d : fFieldDescriptors)
      clone.fFieldDescriptors.emplace(d.first, d.second.Clone());
   for (const auto &d : fColumnDescriptors)
      clone.fColumnDescriptors.emplace(d.first, d.second.Clone());

   for (const auto &d : fExtraTypeInfoDescriptors)
      clone.fExtraTypeInfoDescriptors.emplace_back(d.Clone());
   if (fHeaderExtension)
      clone.fHeaderExtension = std::make_unique<RHeaderExtension>(*fHeaderExtension);

   return clone;
}

ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::RNTupleDescriptor::Clone() const
{
   RNTupleDescriptor clone = CloneSchema();

   clone.fOnDiskHeaderSize = fOnDiskHeaderSize;
   clone.fOnDiskHeaderXxHash3 = fOnDiskHeaderXxHash3;
   clone.fOnDiskFooterSize = fOnDiskFooterSize;
   clone.fNEntries = fNEntries;
   clone.fNClusters = fNClusters;
   clone.fGeneration = fGeneration;
   for (const auto &d : fClusterGroupDescriptors)
      clone.fClusterGroupDescriptors.emplace(d.first, d.second.Clone());
   clone.fSortedClusterGroupIds = fSortedClusterGroupIds;
   for (const auto &d : fClusterDescriptors)
      clone.fClusterDescriptors.emplace(d.first, d.second.Clone());
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

ROOT::RResult<void> ROOT::Experimental::Internal::RClusterDescriptorBuilder::CommitColumnRange(
   ROOT::DescriptorId_t physicalId, std::uint64_t firstElementIndex, std::uint32_t compressionSettings,
   const RClusterDescriptor::RPageRange &pageRange)
{
   if (physicalId != pageRange.fPhysicalColumnId)
      return R__FAIL("column ID mismatch");
   if (fCluster.fColumnRanges.count(physicalId) > 0)
      return R__FAIL("column ID conflict");
   RClusterDescriptor::RColumnRange columnRange{physicalId, firstElementIndex, 0, compressionSettings};
   for (const auto &pi : pageRange.fPageInfos) {
      columnRange.IncrementNElements(pi.GetNElements());
   }
   fCluster.fPageRanges[physicalId] = pageRange.Clone();
   fCluster.fColumnRanges[physicalId] = columnRange;
   return RResult<void>::Success();
}

ROOT::RResult<void>
ROOT::Experimental::Internal::RClusterDescriptorBuilder::MarkSuppressedColumnRange(ROOT::DescriptorId_t physicalId)
{
   if (fCluster.fColumnRanges.count(physicalId) > 0)
      return R__FAIL("column ID conflict");

   RClusterDescriptor::RColumnRange columnRange;
   columnRange.SetPhysicalColumnId(physicalId);
   columnRange.SetIsSuppressed(true);
   fCluster.fColumnRanges[physicalId] = columnRange;
   return RResult<void>::Success();
}

ROOT::RResult<void>
ROOT::Experimental::Internal::RClusterDescriptorBuilder::CommitSuppressedColumnRanges(const RNTupleDescriptor &desc)
{
   for (auto &[_, columnRange] : fCluster.fColumnRanges) {
      if (!columnRange.IsSuppressed())
         continue;
      R__ASSERT(columnRange.GetFirstElementIndex() == ROOT::kInvalidNTupleIndex);

      const auto &columnDesc = desc.GetColumnDescriptor(columnRange.GetPhysicalColumnId());
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
         if (otherColumnRange.IsSuppressed())
            continue;

         columnRange.SetFirstElementIndex(otherColumnRange.GetFirstElementIndex());
         columnRange.SetNElements(otherColumnRange.GetNElements());
         break;
      }

      if (columnRange.GetFirstElementIndex() == ROOT::kInvalidNTupleIndex) {
         return R__FAIL(std::string("cannot find non-suppressed column for column ID ") +
                        std::to_string(columnRange.GetPhysicalColumnId()) +
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
   auto fnTraverseSubtree = [&](ROOT::DescriptorId_t rootFieldId, std::uint64_t nRepetitionsAtThisLevel,
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
         [&](ROOT::DescriptorId_t fieldId, std::uint64_t nRepetitions) {
            for (const auto &c : desc.GetColumnIterable(fieldId)) {
               const ROOT::DescriptorId_t physicalId = c.GetPhysicalId();
               auto &columnRange = fCluster.fColumnRanges[physicalId];

               // Initialize a RColumnRange for `physicalId` if it was not there. Columns that were created during model
               // extension won't have on-disk metadata for the clusters that were already committed before the model
               // was extended. Therefore, these need to be synthetically initialized upon reading.
               if (columnRange.GetPhysicalColumnId() == ROOT::kInvalidDescriptorId) {
                  columnRange.SetPhysicalColumnId(physicalId);
                  columnRange.SetFirstElementIndex(0);
                  columnRange.SetNElements(0);
                  columnRange.SetIsSuppressed(c.IsSuppressedDeferredColumn());
               }
               // Fixup the RColumnRange and RPageRange in deferred columns. We know what the first element index and
               // number of elements should have been if the column was not deferred; fix those and let
               // `ExtendToFitColumnRange()` synthesize RPageInfos accordingly.
               // Note that a deferred column (i.e, whose first element index is > 0) already met the criteria of
               // `RFieldBase::EntryToColumnElementIndex()`, i.e. it is a principal column reachable from the field zero
               // excluding subfields of collection and variant fields.
               if (c.IsDeferredColumn()) {
                  columnRange.SetFirstElementIndex(fCluster.GetFirstEntryIndex() * nRepetitions);
                  columnRange.SetNElements(fCluster.GetNEntries() * nRepetitions);
                  if (!columnRange.IsSuppressed()) {
                     auto &pageRange = fCluster.fPageRanges[physicalId];
                     pageRange.fPhysicalColumnId = physicalId;
                     const auto element = Internal::RColumnElementBase::Generate<void>(c.GetType());
                     pageRange.ExtendToFitColumnRange(columnRange, *element, ROOT::Internal::RPage::kPageZeroSize);
                  }
               } else if (!columnRange.IsSuppressed()) {
                  fCluster.fPageRanges[physicalId].fPhysicalColumnId = physicalId;
               }
            }
         },
         fnTraverseSubtree);
   }
   return *this;
}

ROOT::RResult<ROOT::Experimental::RClusterDescriptor>
ROOT::Experimental::Internal::RClusterDescriptorBuilder::MoveDescriptor()
{
   if (fCluster.fClusterId == ROOT::kInvalidDescriptorId)
      return R__FAIL("unset cluster ID");
   if (fCluster.fNEntries == 0)
      return R__FAIL("empty cluster");
   for (auto &pr : fCluster.fPageRanges) {
      if (fCluster.fColumnRanges.count(pr.first) == 0) {
         return R__FAIL("missing column range");
      }
      pr.second.fCumulativeNElements.clear();
      pr.second.fCumulativeNElements.reserve(pr.second.fPageInfos.size());
      ROOT::NTupleSize_t sum = 0;
      for (const auto &pi : pr.second.fPageInfos) {
         sum += pi.GetNElements();
         pr.second.fCumulativeNElements.emplace_back(sum);
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

ROOT::RResult<ROOT::Experimental::RClusterGroupDescriptor>
ROOT::Experimental::Internal::RClusterGroupDescriptorBuilder::MoveDescriptor()
{
   if (fClusterGroup.fClusterGroupId == ROOT::kInvalidDescriptorId)
      return R__FAIL("unset cluster group ID");
   RClusterGroupDescriptor result;
   std::swap(result, fClusterGroup);
   return result;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::RResult<ROOT::Experimental::RExtraTypeInfoDescriptor>
ROOT::Experimental::Internal::RExtraTypeInfoDescriptorBuilder::MoveDescriptor()
{
   if (fExtraTypeInfo.fContentId == EExtraTypeInfoIds::kInvalid)
      throw RException(R__FAIL("invalid extra type info content id"));
   RExtraTypeInfoDescriptor result;
   std::swap(result, fExtraTypeInfo);
   return result;
}

////////////////////////////////////////////////////////////////////////////////

ROOT::RResult<void>
ROOT::Experimental::Internal::RNTupleDescriptorBuilder::EnsureFieldExists(ROOT::DescriptorId_t fieldId) const
{
   if (fDescriptor.fFieldDescriptors.count(fieldId) == 0)
      return R__FAIL("field with id '" + std::to_string(fieldId) + "' doesn't exist");
   return RResult<void>::Success();
}

ROOT::RResult<void> ROOT::Experimental::Internal::RNTupleDescriptorBuilder::EnsureValidDescriptor() const
{
   // Reuse field name validity check
   auto validName = ROOT::Internal::EnsureValidNameForRNTuple(fDescriptor.GetName(), "Field");
   if (!validName) {
      return R__FORWARD_ERROR(validName);
   }

   for (const auto &[fieldId, fieldDesc] : fDescriptor.fFieldDescriptors) {
      // parent not properly set?
      if (fieldId != fDescriptor.GetFieldZeroId() && fieldDesc.GetParentId() == ROOT::kInvalidDescriptorId) {
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
   fDescriptor.fSortedClusterGroupIds.reserve(fDescriptor.fClusterGroupDescriptors.size());
   for (const auto &[id, _] : fDescriptor.fClusterGroupDescriptors)
      fDescriptor.fSortedClusterGroupIds.emplace_back(id);
   std::sort(fDescriptor.fSortedClusterGroupIds.begin(), fDescriptor.fSortedClusterGroupIds.end(),
             [this](ROOT::DescriptorId_t a, ROOT::DescriptorId_t b) {
                return fDescriptor.fClusterGroupDescriptors[a].GetMinEntry() <
                       fDescriptor.fClusterGroupDescriptors[b].GetMinEntry();
             });
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

ROOT::RResult<ROOT::Experimental::RColumnDescriptor>
ROOT::Experimental::Internal::RColumnDescriptorBuilder::MakeDescriptor() const
{
   if (fColumn.GetLogicalId() == ROOT::kInvalidDescriptorId)
      return R__FAIL("invalid logical column id");
   if (fColumn.GetPhysicalId() == ROOT::kInvalidDescriptorId)
      return R__FAIL("invalid physical column id");
   if (fColumn.GetFieldId() == ROOT::kInvalidDescriptorId)
      return R__FAIL("invalid field id, dangling column");

   // NOTE: if the column type is unknown we don't want to fail, as we might be reading an RNTuple
   // created with a future version of ROOT. In this case we just skip the valid bit range check,
   // as we have no idea what the valid range is.
   // In general, reading the metadata of an unknown column is fine, it becomes an error only when
   // we try to read the actual data contained in it.
   if (fColumn.GetType() != ENTupleColumnType::kUnknown) {
      const auto [minBits, maxBits] = RColumnElementBase::GetValidBitRange(fColumn.GetType());
      if (fColumn.GetBitsOnStorage() < minBits || fColumn.GetBitsOnStorage() > maxBits)
         return R__FAIL("invalid column bit width");
   }

   return fColumn.Clone();
}

ROOT::Experimental::Internal::RFieldDescriptorBuilder::RFieldDescriptorBuilder(const RFieldDescriptor &fieldDesc)
   : fField(fieldDesc.Clone())
{
   fField.fParentId = ROOT::kInvalidDescriptorId;
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

ROOT::RResult<ROOT::Experimental::RFieldDescriptor>
ROOT::Experimental::Internal::RFieldDescriptorBuilder::MakeDescriptor() const
{
   if (fField.GetId() == ROOT::kInvalidDescriptorId) {
      return R__FAIL("invalid field id");
   }
   if (fField.GetStructure() == ROOT::ENTupleStructure::kInvalid) {
      return R__FAIL("invalid field structure");
   }
   // FieldZero is usually named "" and would be a false positive here
   if (fField.GetParentId() != ROOT::kInvalidDescriptorId) {
      auto validName = ROOT::Internal::EnsureValidNameForRNTuple(fField.GetFieldName(), "Field");
      if (!validName) {
         return R__FORWARD_ERROR(validName);
      }
      if (fField.GetFieldName().empty()) {
         return R__FAIL("name cannot be empty string \"\"");
      }
   }
   return fField.Clone();
}

void ROOT::Experimental::Internal::RNTupleDescriptorBuilder::AddField(const RFieldDescriptor &fieldDesc)
{
   fDescriptor.fFieldDescriptors.emplace(fieldDesc.GetId(), fieldDesc.Clone());
   if (fDescriptor.fHeaderExtension)
      fDescriptor.fHeaderExtension->MarkExtendedField(fieldDesc);
   if (fieldDesc.GetFieldName().empty() && fieldDesc.GetParentId() == ROOT::kInvalidDescriptorId) {
      fDescriptor.fFieldZeroId = fieldDesc.GetId();
   }
}

ROOT::RResult<void> ROOT::Experimental::Internal::RNTupleDescriptorBuilder::AddFieldLink(ROOT::DescriptorId_t fieldId,
                                                                                         ROOT::DescriptorId_t linkId)
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
   if ((parentId != ROOT::kInvalidDescriptorId) && (parentId != fieldId)) {
      return R__FAIL("field '" + std::to_string(linkId) + "' already has a parent ('" + std::to_string(parentId) + ")");
   }
   if (fieldId == linkId) {
      return R__FAIL("cannot make field '" + std::to_string(fieldId) + "' a child of itself");
   }
   fDescriptor.fFieldDescriptors.at(linkId).fParentId = fieldId;
   fDescriptor.fFieldDescriptors.at(fieldId).fLinkIds.push_back(linkId);
   return RResult<void>::Success();
}

ROOT::RResult<void>
ROOT::Experimental::Internal::RNTupleDescriptorBuilder::AddFieldProjection(ROOT::DescriptorId_t sourceId,
                                                                           ROOT::DescriptorId_t targetId)
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

ROOT::RResult<void> ROOT::Experimental::Internal::RNTupleDescriptorBuilder::AddColumn(RColumnDescriptor &&columnDesc)
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
   if (fDescriptor.FindLogicalColumnId(fieldId, columnIndex, representationIndex) != ROOT::kInvalidDescriptorId) {
      return R__FAIL("column index clash");
   }
   if (columnIndex > 0) {
      if (fDescriptor.FindLogicalColumnId(fieldId, columnIndex - 1, representationIndex) == ROOT::kInvalidDescriptorId)
         return R__FAIL("out of bounds column index");
   }
   if (representationIndex > 0) {
      if (fDescriptor.FindLogicalColumnId(fieldId, 0, representationIndex - 1) == ROOT::kInvalidDescriptorId) {
         return R__FAIL("out of bounds representation index");
      }
      if (columnIndex == 0) {
         assert(fieldDesc.fColumnCardinality > 0);
         if (fDescriptor.FindLogicalColumnId(fieldId, fieldDesc.fColumnCardinality - 1, representationIndex - 1) ==
             ROOT::kInvalidDescriptorId) {
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
      fDescriptor.fHeaderExtension->MarkExtendedColumn(columnDesc);

   return RResult<void>::Success();
}

ROOT::RResult<void>
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

void ROOT::Experimental::Internal::RNTupleDescriptorBuilder::SetSchemaFromExisting(const RNTupleDescriptor &descriptor)
{
   fDescriptor = descriptor.CloneSchema();
}

void ROOT::Experimental::Internal::RNTupleDescriptorBuilder::BeginHeaderExtension()
{
   if (!fDescriptor.fHeaderExtension)
      fDescriptor.fHeaderExtension = std::make_unique<RNTupleDescriptor::RHeaderExtension>();
}

void ROOT::Experimental::Internal::RNTupleDescriptorBuilder::ShiftAliasColumns(std::uint32_t offset)
{
   if (fDescriptor.GetNLogicalColumns() == 0)
      return;
   R__ASSERT(fDescriptor.GetNPhysicalColumns() > 0);

   for (ROOT::DescriptorId_t id = fDescriptor.GetNLogicalColumns() - 1; id >= fDescriptor.GetNPhysicalColumns(); --id) {
      auto c = fDescriptor.fColumnDescriptors[id].Clone();
      R__ASSERT(c.IsAliasColumn());
      R__ASSERT(id == c.GetLogicalId());
      fDescriptor.fColumnDescriptors.erase(id);
      for (auto &link : fDescriptor.fFieldDescriptors[c.fFieldId].fLogicalColumnIds) {
         if (link == c.fLogicalColumnId) {
            link += offset;
            break;
         }
      }
      c.fLogicalColumnId += offset;
      R__ASSERT(fDescriptor.fColumnDescriptors.count(c.fLogicalColumnId) == 0);
      fDescriptor.fColumnDescriptors.emplace(c.fLogicalColumnId, std::move(c));
   }
}

ROOT::RResult<void> ROOT::Experimental::Internal::RNTupleDescriptorBuilder::AddCluster(RClusterDescriptor &&clusterDesc)
{
   auto clusterId = clusterDesc.GetId();
   if (fDescriptor.fClusterDescriptors.count(clusterId) > 0)
      return R__FAIL("cluster id clash");
   fDescriptor.fClusterDescriptors.emplace(clusterId, std::move(clusterDesc));
   return RResult<void>::Success();
}

ROOT::RResult<void>
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

void ROOT::Experimental::Internal::RNTupleDescriptorBuilder::ReplaceExtraTypeInfo(
   RExtraTypeInfoDescriptor &&extraTypeInfoDesc)
{
   auto it = std::find(fDescriptor.fExtraTypeInfoDescriptors.begin(), fDescriptor.fExtraTypeInfoDescriptors.end(),
                       extraTypeInfoDesc);
   if (it != fDescriptor.fExtraTypeInfoDescriptors.end())
      *it = std::move(extraTypeInfoDesc);
   else
      fDescriptor.fExtraTypeInfoDescriptors.emplace_back(std::move(extraTypeInfoDesc));
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

   // Add the streamer info records from streamer fields: because of runtime polymorphism we may need to add additional
   // types not covered by the type names stored in the field headers
   for (const auto &extraTypeInfo : desc.GetExtraTypeInfoIterable()) {
      if (extraTypeInfo.GetContentId() != EExtraTypeInfoIds::kStreamerInfo)
         continue;
      // Ideally, we would avoid deserializing the streamer info records of the streamer fields that we just serialized.
      // However, this happens only once at the end of writing and only when streamer fields are used, so the
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
   const RFieldDescriptor &fieldDesc,
   const std::function<bool(ROOT::DescriptorId_t, ROOT::DescriptorId_t)> &comparator) const
{
   return RFieldDescriptorIterable(*this, fieldDesc, comparator);
}

ROOT::Experimental::RNTupleDescriptor::RFieldDescriptorIterable
ROOT::Experimental::RNTupleDescriptor::GetFieldIterable(ROOT::DescriptorId_t fieldId) const
{
   return GetFieldIterable(GetFieldDescriptor(fieldId));
}

ROOT::Experimental::RNTupleDescriptor::RFieldDescriptorIterable ROOT::Experimental::RNTupleDescriptor::GetFieldIterable(
   ROOT::DescriptorId_t fieldId,
   const std::function<bool(ROOT::DescriptorId_t, ROOT::DescriptorId_t)> &comparator) const
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
   const std::function<bool(ROOT::DescriptorId_t, ROOT::DescriptorId_t)> &comparator) const
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
ROOT::Experimental::RNTupleDescriptor::GetColumnIterable(ROOT::DescriptorId_t fieldId) const
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
