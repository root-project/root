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
   default: throw ROOT::RException(R__FAIL("value size not supported"));
   }

   return value;
}
} // anonymous namespace

ROOT::Experimental::Internal::RNTupleIndex::RNTupleIndex(const std::vector<std::string> &fieldNames,
                                                         const RPageSource &pageSource)
   : fPageSource(pageSource.Clone())
{
   fPageSource->Attach();
   auto desc = fPageSource->GetSharedDescriptorGuard();

   fIndexFields.reserve(fieldNames.size());

   for (const auto &fieldName : fieldNames) {
      auto fieldId = desc->FindFieldId(fieldName);
      if (fieldId == kInvalidDescriptorId)
         throw RException(R__FAIL("could not find join field \"" + std::string(fieldName) + "\" in RNTuple \"" +
                                  fPageSource->GetNTupleName() + "\""));

      const auto &fieldDesc = desc->GetFieldDescriptor(fieldId);
      auto field = fieldDesc.CreateField(desc.GetRef());

      CallConnectPageSourceOnField(*field, *fPageSource);

      fIndexFields.push_back(std::move(field));
   }
}

void ROOT::Experimental::Internal::RNTupleIndex::EnsureBuilt() const
{
   if (!fIsBuilt)
      throw RException(R__FAIL("index has not been built yet"));
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

   static const std::unordered_set<std::string> allowedTypes = {"std::int8_t",   "std::int16_t", "std::int32_t",
                                                                "std::int64_t",  "std::uint8_t", "std::uint16_t",
                                                                "std::uint32_t", "std::uint64_t"};

   std::vector<RFieldBase::RValue> fieldValues;
   fieldValues.reserve(fIndexFields.size());

   for (const auto &field : fIndexFields) {
      if (allowedTypes.find(field->GetTypeName()) == allowedTypes.end()) {
         throw RException(R__FAIL("cannot use field \"" + field->GetFieldName() + "\" with type \"" +
                                  field->GetTypeName() + "\" for indexing: only integral types are allowed"));
      }
      fieldValues.emplace_back(field->CreateValue());
   }

   std::vector<NTupleIndexValue_t> indexValues;
   indexValues.reserve(fIndexFields.size());

   for (unsigned i = 0; i < fPageSource->GetNEntries(); ++i) {
      indexValues.clear();
      for (auto &fieldValue : fieldValues) {
         // TODO(fdegeus): use bulk reading
         fieldValue.Read(i);

         auto valuePtr = fieldValue.GetPtr<void>();
         indexValues.push_back(CastValuePtr(valuePtr.get(), fieldValue.GetField()));
      }
      fIndex[RIndexValue(indexValues)].push_back(i);
   }

   fIsBuilt = true;
}

ROOT::Experimental::NTupleSize_t
ROOT::Experimental::Internal::RNTupleIndex::GetFirstEntryNumber(const std::vector<void *> &valuePtrs) const
{
   const auto entryIndices = GetAllEntryNumbers(valuePtrs);
   if (!entryIndices)
      return kInvalidNTupleIndex;
   return entryIndices->front();
}

const std::vector<ROOT::Experimental::NTupleSize_t> *
ROOT::Experimental::Internal::RNTupleIndex::GetAllEntryNumbers(const std::vector<void *> &valuePtrs) const
{
   if (valuePtrs.size() != fIndexFields.size())
      throw RException(R__FAIL("number of value pointers must match number of indexed fields"));

   EnsureBuilt();

   std::vector<NTupleIndexValue_t> indexValues;
   indexValues.reserve(fIndexFields.size());

   for (unsigned i = 0; i < valuePtrs.size(); ++i) {
      indexValues.push_back(CastValuePtr(valuePtrs[i], *fIndexFields[i]));
   }

   auto entryNumber = fIndex.find(RIndexValue(indexValues));

   if (entryNumber == fIndex.end())
      return nullptr;

   return &(entryNumber->second);
}
