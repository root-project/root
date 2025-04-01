/// \file RNTupleJoinTable.cxx
/// \ingroup NTuple
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2024-04-02
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleJoinTable.hxx>

namespace {
ROOT::Experimental::Internal::RNTupleJoinTable::JoinValue_t CastValuePtr(void *valuePtr, std::size_t fieldValueSize)
{
   ROOT::Experimental::Internal::RNTupleJoinTable::JoinValue_t value;

   switch (fieldValueSize) {
   case 1: value = *reinterpret_cast<std::uint8_t *>(valuePtr); break;
   case 2: value = *reinterpret_cast<std::uint16_t *>(valuePtr); break;
   case 4: value = *reinterpret_cast<std::uint32_t *>(valuePtr); break;
   case 8: value = *reinterpret_cast<std::uint64_t *>(valuePtr); break;
   default: throw ROOT::RException(R__FAIL("value size not supported"));
   }

   return value;
}
} // anonymous namespace

ROOT::Experimental::Internal::RNTupleJoinTable::REntryMapping::REntryMapping(
   ROOT::Internal::RPageSource &pageSource, const std::vector<std::string> &joinFieldNames,
   ROOT::NTupleSize_t entryOffset)
   : fJoinFieldNames(joinFieldNames)
{
   static const std::unordered_set<std::string> allowedTypes = {"std::int8_t",   "std::int16_t", "std::int32_t",
                                                                "std::int64_t",  "std::uint8_t", "std::uint16_t",
                                                                "std::uint32_t", "std::uint64_t"};

   pageSource.Attach();
   auto desc = pageSource.GetSharedDescriptorGuard();

   std::vector<std::unique_ptr<ROOT::RFieldBase>> fields;
   std::vector<ROOT::RFieldBase::RValue> fieldValues;
   fieldValues.reserve(fJoinFieldNames.size());

   for (const auto &fieldName : fJoinFieldNames) {
      auto fieldId = desc->FindFieldId(fieldName);
      if (fieldId == ROOT::kInvalidDescriptorId)
         throw RException(R__FAIL("could not find join field \"" + std::string(fieldName) + "\" in RNTuple \"" +
                                  pageSource.GetNTupleName() + "\""));

      const auto &fieldDesc = desc->GetFieldDescriptor(fieldId);

      if (allowedTypes.find(fieldDesc.GetTypeName()) == allowedTypes.end()) {
         throw RException(R__FAIL("cannot use field \"" + fieldName + "\" with type \"" + fieldDesc.GetTypeName() +
                                  "\" in join table: only integral types are allowed"));
      }

      auto field = fieldDesc.CreateField(desc.GetRef());
      ROOT::Internal::CallConnectPageSourceOnField(*field, pageSource);

      fieldValues.emplace_back(field->CreateValue());
      fJoinFieldValueSizes.emplace_back(field->GetValueSize());
      fields.emplace_back(std::move(field));
   }

   std::vector<JoinValue_t> castJoinValues;
   castJoinValues.reserve(fJoinFieldNames.size());

   for (unsigned i = 0; i < pageSource.GetNEntries(); ++i) {
      castJoinValues.clear();

      for (auto &fieldValue : fieldValues) {
         // TODO(fdegeus): use bulk reading
         fieldValue.Read(i);

         auto valuePtr = fieldValue.GetPtr<void>();
         castJoinValues.push_back(CastValuePtr(valuePtr.get(), fieldValue.GetField().GetValueSize()));
      }

      fMapping[RCombinedJoinFieldValue(castJoinValues)].push_back(i + entryOffset);
   }
}

const std::vector<ROOT::NTupleSize_t> *
ROOT ::Experimental::Internal::RNTupleJoinTable::REntryMapping::GetEntryIndexes(std::vector<void *> valuePtrs) const
{
   if (valuePtrs.size() != fJoinFieldNames.size())
      throw RException(R__FAIL("number of value pointers must match number of join fields"));

   std::vector<JoinValue_t> castJoinValues;
   castJoinValues.reserve(valuePtrs.size());

   for (unsigned i = 0; i < valuePtrs.size(); ++i) {
      castJoinValues.push_back(CastValuePtr(valuePtrs[i], fJoinFieldValueSizes[i]));
   }

   if (const auto &entries = fMapping.find(RCombinedJoinFieldValue(castJoinValues)); entries != fMapping.end()) {
      return &entries->second;
   }

   return nullptr;
}

//------------------------------------------------------------------------------

std::unique_ptr<ROOT::Experimental::Internal::RNTupleJoinTable>
ROOT::Experimental::Internal::RNTupleJoinTable::Create(const std::vector<std::string> &fieldNames)
{
   return std::unique_ptr<RNTupleJoinTable>(new RNTupleJoinTable(fieldNames));
}

ROOT::Experimental::Internal::RNTupleJoinTable &
ROOT::Experimental::Internal::RNTupleJoinTable::Add(ROOT::Internal::RPageSource &pageSource,
                                                    PartitionKey_t partitionKey, ROOT::NTupleSize_t entryOffset)
{
   auto joinMapping = std::unique_ptr<REntryMapping>(new REntryMapping(pageSource, fJoinFieldNames, entryOffset));
   fPartitions[partitionKey].emplace_back(std::move(joinMapping));

   return *this;
}

ROOT::NTupleSize_t
ROOT::Experimental::Internal::RNTupleJoinTable::GetEntryIndex(const std::vector<void *> &valuePtrs) const
{

   for (const auto &partition : fPartitions) {
      for (const auto &joinMapping : partition.second) {
         auto entriesForMapping = joinMapping->GetEntryIndexes(valuePtrs);
         if (entriesForMapping) {
            return (*entriesForMapping)[0];
         }
      }
   }

   return kInvalidNTupleIndex;
}

std::vector<ROOT::NTupleSize_t>
ROOT::Experimental::Internal::RNTupleJoinTable::GetEntryIndexes(const std::vector<void *> &valuePtrs,
                                                                PartitionKey_t partitionKey) const
{
   auto partition = fPartitions.find(partitionKey);
   if (partition == fPartitions.end())
      return {};

   std::vector<ROOT::NTupleSize_t> entryIdxs{};

   for (const auto &joinMapping : partition->second) {
      auto entriesForMapping = joinMapping->GetEntryIndexes(valuePtrs);
      if (entriesForMapping)
         entryIdxs.insert(entryIdxs.end(), entriesForMapping->begin(), entriesForMapping->end());
   }

   return entryIdxs;
}

std::unordered_map<ROOT::Experimental::Internal::RNTupleJoinTable::PartitionKey_t, std::vector<ROOT::NTupleSize_t>>
ROOT::Experimental::Internal::RNTupleJoinTable::GetPartitionedEntryIndexes(
   const std::vector<void *> &valuePtrs, const std::vector<PartitionKey_t> &partitionKeys) const
{
   std::unordered_map<PartitionKey_t, std::vector<ROOT::NTupleSize_t>> entryIdxs{};

   for (const auto &partitionKey : partitionKeys) {
      auto entriesForPartition = GetEntryIndexes(valuePtrs, partitionKey);
      if (!entriesForPartition.empty()) {
         entryIdxs[partitionKey].insert(entryIdxs[partitionKey].end(), entriesForPartition.begin(),
                                        entriesForPartition.end());
      }
   }

   return entryIdxs;
}

std::unordered_map<ROOT::Experimental::Internal::RNTupleJoinTable::PartitionKey_t, std::vector<ROOT::NTupleSize_t>>
ROOT::Experimental::Internal::RNTupleJoinTable::GetPartitionedEntryIndexes(const std::vector<void *> &valuePtrs) const
{
   std::unordered_map<PartitionKey_t, std::vector<ROOT::NTupleSize_t>> entryIdxs{};

   for (const auto &partition : fPartitions) {
      for (const auto &joinMapping : partition.second) {
         auto entriesForMapping = joinMapping->GetEntryIndexes(valuePtrs);
         if (entriesForMapping) {
            entryIdxs[partition.first].insert(entryIdxs[partition.first].end(), entriesForMapping->begin(),
                                              entriesForMapping->end());
         }
      }
   }

   return entryIdxs;
}
