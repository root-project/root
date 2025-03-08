/// \file RNTupleJoinTable.cxx
/// \ingroup NTuple ROOT7
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
using NTupleJoinFieldValue_t = std::uint64_t;

NTupleJoinFieldValue_t CastValuePtr(void *valuePtr, std::size_t fieldValueSize)
{
   NTupleJoinFieldValue_t value;

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

void ROOT::Experimental::Internal::RNTupleJoinTable::REntryMapping::Build()
{
   assert(fMapping.size() == 0 && "mapping has already been built");

   static const std::unordered_set<std::string> allowedTypes = {"std::int8_t",   "std::int16_t", "std::int32_t",
                                                                "std::int64_t",  "std::uint8_t", "std::uint16_t",
                                                                "std::uint32_t", "std::uint64_t"};

   fPageSource->Attach();
   auto desc = fPageSource->GetSharedDescriptorGuard();

   std::vector<std::unique_ptr<RFieldBase>> fields;

   for (const auto &fieldName : fJoinFieldNames) {
      auto fieldId = desc->FindFieldId(fieldName);
      if (fieldId == ROOT::kInvalidDescriptorId)
         throw RException(R__FAIL("could not find join field \"" + std::string(fieldName) + "\" in RNTuple \"" +
                                  fPageSource->GetNTupleName() + "\""));

      const auto &fieldDesc = desc->GetFieldDescriptor(fieldId);

      if (allowedTypes.find(fieldDesc.GetTypeName()) == allowedTypes.end()) {
         throw RException(R__FAIL("cannot use field \"" + fieldName + "\" with type \"" + fieldDesc.GetTypeName() +
                                  "\" in join table: only integral types are allowed"));
      }

      auto field = fieldDesc.CreateField(desc.GetRef());
      CallConnectPageSourceOnField(*field, *fPageSource);

      fJoinFieldValueSizes.emplace_back(field->GetValueSize());
      fields.emplace_back(std::move(field));
   }

   std::vector<NTupleJoinFieldValue_t> castJoinValues;

   for (unsigned i = 0; i < fPageSource->GetNEntries(); ++i) {
      castJoinValues.clear();
      for (auto &field : fields) {
         auto value = field->CreateValue();
         // TODO(fdegeus): use bulk reading
         value.Read(i);

         auto valuePtr = value.GetPtr<void>();
         castJoinValues.push_back(CastValuePtr(valuePtr.get(), value.GetField().GetValueSize()));
      }

      fMapping[RCombinedJoinFieldValue(castJoinValues)].push_back(i);
   }
}

const std::vector<ROOT::NTupleSize_t> *
ROOT ::Experimental::Internal::RNTupleJoinTable::REntryMapping::GetEntryIndexes(std::vector<void *> valuePtrs) const
{
   assert(fMapping.size() != 0 && "mapping has not been built yet");

   if (valuePtrs.size() != fJoinFieldNames.size())
      throw RException(R__FAIL("number of value pointers must match number of join fields"));

   std::vector<NTupleJoinFieldValue_t> joinFieldValues;
   joinFieldValues.reserve(valuePtrs.size());

   for (unsigned i = 0; i < valuePtrs.size(); ++i) {
      joinFieldValues.push_back(CastValuePtr(valuePtrs[i], fJoinFieldValueSizes[i]));
   }

   if (const auto &entries = fMapping.find(RCombinedJoinFieldValue(joinFieldValues)); entries != fMapping.end()) {
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

const std::vector<ROOT::Experimental::Internal::RNTupleJoinTable::REntryMapping *>
ROOT::Experimental::Internal::RNTupleJoinTable::GetAllMappings() const
{
   std::vector<REntryMapping *> joinMappingPtrs;

   for (const auto &key : fPartitionKeys) {
      auto mappings = GetMappingsForKey(key);
      joinMappingPtrs.insert(joinMappingPtrs.end(), mappings.begin(), mappings.end());
   }

   return joinMappingPtrs;
}

const std::vector<ROOT::Experimental::Internal::RNTupleJoinTable::REntryMapping *>
ROOT::Experimental::Internal::RNTupleJoinTable::GetMappingsForKey(PartitionKey_t partitionKey) const
{
   auto partitionIter = std::find(fPartitionKeys.begin(), fPartitionKeys.end(), partitionKey);
   if (partitionIter == fPartitionKeys.end())
      return {};

   auto partitionIdx = std::distance(fPartitionKeys.begin(), partitionIter);
   std::vector<REntryMapping *> joinMappingPtrs;
   joinMappingPtrs.reserve(fPartitions[partitionIdx].size());

   for (const auto &joinMapping : fPartitions[partitionIdx]) {
      joinMappingPtrs.emplace_back(joinMapping.get());
   }
   return joinMappingPtrs;
}

ROOT::Experimental::Internal::RNTupleJoinTable &
ROOT::Experimental::Internal::RNTupleJoinTable::Add(RPageSource &pageSource, PartitionKey_t partitionKey)
{
   if (fIsBuilt) {
      throw RException(R__FAIL("cannot add to an already-built join table"));
   }

   auto joinMapping = std::unique_ptr<REntryMapping>(new REntryMapping(pageSource, fJoinFieldNames));

   auto partitionIter = std::find(fPartitionKeys.begin(), fPartitionKeys.end(), partitionKey);
   if (partitionIter == fPartitionKeys.end()) {
      std::vector<std::unique_ptr<REntryMapping>> newPartition;
      newPartition.emplace_back(std::move(joinMapping));
      fPartitions.emplace_back(std::move(newPartition));
      fPartitionKeys.push_back(partitionKey);
   } else {
      auto partitionIdx = std::distance(fPartitionKeys.begin(), partitionIter);
      fPartitions[partitionIdx].emplace_back(std::move(joinMapping));
   }

   return *this;
}

void ROOT::Experimental::Internal::RNTupleJoinTable::Build()
{
   if (fIsBuilt)
      return;

   for (auto &mapping : GetAllMappings()) {
      mapping->Build();
   }

   fIsBuilt = true;
}

std::vector<ROOT::NTupleSize_t>
ROOT::Experimental::Internal::RNTupleJoinTable::GetEntryIndexes(const std::vector<void *> &valuePtrs,
                                                                const std::vector<PartitionKey_t> &partitionKeys) const
{
   if (!fIsBuilt)
      throw RException(R__FAIL("join table has not been built yet"));

   std::vector<ROOT::NTupleSize_t> entryIdxs{};

   for (const auto &partitionKey : partitionKeys) {
      for (const auto &joinMapping : GetMappingsForKey(partitionKey)) {
         auto entriesForMapping = joinMapping->GetEntryIndexes(valuePtrs);
         if (entriesForMapping)
            entryIdxs.insert(entryIdxs.end(), entriesForMapping->begin(), entriesForMapping->end());
      }
   }

   return entryIdxs;
}
