/// \file RNTupleIndex.cxx
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

#include <ROOT/RNTupleIndex.hxx>

#include <TROOT.h>
#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif

namespace {
ROOT::Experimental::Internal::RNTupleIndex::NTupleIndexValue_t
CastValuePtr(void *valuePtr, const ROOT::Experimental::RFieldBase &field)
{
   ROOT::Experimental::Internal::RNTupleIndex::NTupleIndexValue_t value;

   switch (field.GetValueSize()) {
   case 1: value = *reinterpret_cast<std::uint8_t *>(valuePtr); break;
   case 2: value = *reinterpret_cast<std::uint16_t *>(valuePtr); break;
   case 4: value = *reinterpret_cast<std::uint32_t *>(valuePtr); break;
   case 8: value = *reinterpret_cast<std::uint64_t *>(valuePtr); break;
   default: throw ROOT::Experimental::RException(R__FAIL("value size not supported"));
   }

   return value;
}
} // anonymous namespace

void ROOT::Experimental::Internal::RNTupleIndex::RNTupleIndexPartition::Build()
{
   std::vector<RFieldBase::RValue> fieldValues;
   fieldValues.reserve(fIndexFields.size());
   for (const auto &field : fIndexFields) {
      fieldValues.emplace_back(field->CreateValue());
   }

   std::vector<NTupleIndexValue_t> indexFieldValues;
   indexFieldValues.reserve(fieldValues.size());

   for (unsigned i = fFirstEntry; i < fLastEntry; ++i) {
      indexFieldValues.clear();
      for (auto &fieldValue : fieldValues) {
         // TODO(fdegeus): use bulk reading
         fieldValue.Read(i);

         auto valuePtr = fieldValue.GetPtr<void>();
         indexFieldValues.push_back(CastValuePtr(valuePtr.get(), fieldValue.GetField()));
      }

      RIndexValue indexValue(indexFieldValues);
      fIndex[indexValue].push_back(i);
   }
}

//------------------------------------------------------------------------------

ROOT::Experimental::Internal::RNTupleIndex::RNTupleIndex(const std::vector<std::string> &fieldNames,
                                                         const RPageSource &pageSource)
   : fPageSource(pageSource.Clone())
{
   fPageSource->Attach();
   auto desc = fPageSource->GetSharedDescriptorGuard();

   fIndexFields.reserve(fieldNames.size());

   static const std::unordered_set<std::string> allowedTypes = {"std::int8_t",   "std::int16_t", "std::int32_t",
                                                                "std::int64_t",  "std::uint8_t", "std::uint16_t",
                                                                "std::uint32_t", "std::uint64_t"};

   for (const auto &fieldName : fieldNames) {
      auto fieldId = desc->FindFieldId(fieldName);
      if (fieldId == kInvalidDescriptorId)
         throw RException(R__FAIL("Could not find field \"" + std::string(fieldName) + "."));

      const auto &fieldDesc = desc->GetFieldDescriptor(fieldId);
      auto field = fieldDesc.CreateField(desc.GetRef());

      if (allowedTypes.find(field->GetTypeName()) == allowedTypes.end()) {
         throw RException(R__FAIL("Cannot use field \"" + field->GetFieldName() + "\" with type \"" +
                                  field->GetTypeName() + "\" for indexing. Only integral types are allowed."));
      }

      fIndexFields.push_back(std::move(field));
   }
}

void ROOT::Experimental::Internal::RNTupleIndex::EnsureBuilt() const
{
   if (!fIsBuilt)
      throw RException(R__FAIL("Index has not been built yet"));
}

std::unique_ptr<ROOT::Experimental::Internal::RNTupleIndex>
ROOT::Experimental::Internal::RNTupleIndex::Create(const std::vector<std::string> &fieldNames,
                                                   const RPageSource &pageSource, bool deferBuild)
{
   auto index = std::unique_ptr<RNTupleIndex>(new RNTupleIndex(fieldNames, pageSource));

   if (!deferBuild)
      index->Build();

   return index;
}

void ROOT::Experimental::Internal::RNTupleIndex::Build()
{
   if (fIsBuilt)
      return;

   auto desc = fPageSource->GetSharedDescriptorGuard();

   fIndexPartitions.reserve(desc->GetNClusters());

   if (ROOT::IsImplicitMTEnabled()) {
#ifdef R__USE_IMT
      for (const auto &cluster : desc->GetClusterIterable()) {
         fIndexPartitions.emplace_back(cluster, fIndexFields, *fPageSource);
      }

      auto fnBuildIndexPartition = [](RNTupleIndexPartition &indexPartition) -> void { indexPartition.Build(); };

      if (!fPool)
         fPool = std::make_unique<ROOT::TThreadExecutor>();
      fPool->Foreach(fnBuildIndexPartition, fIndexPartitions);
#else
      assert(false);
#endif
   } else {
      for (const auto &cluster : desc->GetClusterIterable()) {
         auto &indexPartition = fIndexPartitions.emplace_back(cluster, fIndexFields, *fPageSource);
         indexPartition.Build();
      }
   }

   fIsBuilt = true;
}

ROOT::Experimental::NTupleSize_t
ROOT::Experimental::Internal::RNTupleIndex::GetFirstEntryNumber(const std::vector<void *> &valuePtrs) const
{
   const auto entryNumbers = GetAllEntryNumbers(valuePtrs);
   if (entryNumbers.empty())
      return kInvalidNTupleIndex;
   return entryNumbers.front();
}

const std::vector<ROOT::Experimental::NTupleSize_t>
ROOT::Experimental::Internal::RNTupleIndex::GetAllEntryNumbers(const std::vector<void *> &valuePtrs) const
{
   if (valuePtrs.size() != fIndexFields.size())
      throw RException(R__FAIL("Number of value pointers must match number of indexed fields."));

   EnsureBuilt();

   std::vector<std::vector<NTupleIndexValue_t>> entryNumbersPerCluster;

   std::vector<NTupleIndexValue_t> indexFieldValues;
   indexFieldValues.reserve(fIndexFields.size());

   for (unsigned i = 0; i < valuePtrs.size(); ++i) {
      indexFieldValues.push_back(CastValuePtr(valuePtrs[i], *fIndexFields[i]));
   }

   RIndexValue indexValue(indexFieldValues);

   for (const auto &indexPartition : fIndexPartitions) {
      auto clusterEntryNumbers = indexPartition.fIndex.find(indexValue);

      if (clusterEntryNumbers == indexPartition.fIndex.end())
         continue;

      entryNumbersPerCluster.push_back(clusterEntryNumbers->second);
   }

   std::vector<NTupleIndexValue_t> entryNumbers;

   for (const auto &clusterEntries : entryNumbersPerCluster) {
      entryNumbers.insert(entryNumbers.end(), clusterEntries.cbegin(), clusterEntries.cend());
   }

   return entryNumbers;
}
