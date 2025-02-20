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
ROOT::Experimental::Internal::RNTupleJoinTable::NTupleJoinValue_t
CastValuePtr(void *valuePtr, const ROOT::Experimental::RFieldBase &field)
{
   ROOT::Experimental::Internal::RNTupleJoinTable::NTupleJoinValue_t value;

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

ROOT::Experimental::Internal::RNTupleJoinTable::RNTupleJoinTable(const std::vector<std::string> &fieldNames,
                                                                 const RPageSource &pageSource)
   : fPageSource(pageSource.Clone())
{
   fPageSource->Attach();
   auto desc = fPageSource->GetSharedDescriptorGuard();

   fJoinFields.reserve(fieldNames.size());

   for (const auto &fieldName : fieldNames) {
      auto fieldId = desc->FindFieldId(fieldName);
      if (fieldId == ROOT::kInvalidDescriptorId)
         throw RException(R__FAIL("could not find join field \"" + std::string(fieldName) + "\" in RNTuple \"" +
                                  fPageSource->GetNTupleName() + "\""));

      const auto &fieldDesc = desc->GetFieldDescriptor(fieldId);
      auto field = fieldDesc.CreateField(desc.GetRef());

      CallConnectPageSourceOnField(*field, *fPageSource);

      fJoinFields.push_back(std::move(field));
   }
}

void ROOT::Experimental::Internal::RNTupleJoinTable::EnsureBuilt() const
{
   if (!fIsBuilt)
      throw RException(R__FAIL("join table has not been built yet"));
}

std::unique_ptr<ROOT::Experimental::Internal::RNTupleJoinTable>
ROOT::Experimental::Internal::RNTupleJoinTable::Create(const std::vector<std::string> &fieldNames,
                                                       const RPageSource &pageSource)
{
   auto joinTable = std::unique_ptr<RNTupleJoinTable>(new RNTupleJoinTable(fieldNames, pageSource));

   return joinTable;
}

void ROOT::Experimental::Internal::RNTupleJoinTable::Build()
{
   if (fIsBuilt)
      return;

   static const std::unordered_set<std::string> allowedTypes = {"std::int8_t",   "std::int16_t", "std::int32_t",
                                                                "std::int64_t",  "std::uint8_t", "std::uint16_t",
                                                                "std::uint32_t", "std::uint64_t"};

   std::vector<RFieldBase::RValue> fieldValues;
   fieldValues.reserve(fJoinFields.size());

   for (const auto &field : fJoinFields) {
      if (allowedTypes.find(field->GetTypeName()) == allowedTypes.end()) {
         throw RException(R__FAIL("cannot use field \"" + field->GetFieldName() + "\" with type \"" +
                                  field->GetTypeName() + "\" in join table: only integral types are allowed"));
      }
      fieldValues.emplace_back(field->CreateValue());
   }

   std::vector<NTupleJoinValue_t> joinFieldValues;
   joinFieldValues.reserve(fJoinFields.size());

   for (unsigned i = 0; i < fPageSource->GetNEntries(); ++i) {
      joinFieldValues.clear();
      for (auto &fieldValue : fieldValues) {
         // TODO(fdegeus): use bulk reading
         fieldValue.Read(i);

         auto valuePtr = fieldValue.GetPtr<void>();
         joinFieldValues.push_back(CastValuePtr(valuePtr.get(), fieldValue.GetField()));
      }
      fJoinTable[RCombinedJoinFieldValue(joinFieldValues)].push_back(i);
   }

   fIsBuilt = true;
}

ROOT::NTupleSize_t
ROOT::Experimental::Internal::RNTupleJoinTable::GetFirstEntryNumber(const std::vector<void *> &valuePtrs) const
{
   const auto entryIndices = GetAllEntryNumbers(valuePtrs);
   if (!entryIndices)
      return ROOT::kInvalidNTupleIndex;
   return entryIndices->front();
}

const std::vector<ROOT::NTupleSize_t> *
ROOT::Experimental::Internal::RNTupleJoinTable::GetAllEntryNumbers(const std::vector<void *> &valuePtrs) const
{
   if (valuePtrs.size() != fJoinFields.size())
      throw RException(R__FAIL("number of value pointers must match number of join fields"));

   EnsureBuilt();

   std::vector<NTupleJoinValue_t> joinFieldValues;
   joinFieldValues.reserve(fJoinFields.size());

   for (unsigned i = 0; i < valuePtrs.size(); ++i) {
      joinFieldValues.push_back(CastValuePtr(valuePtrs[i], *fJoinFields[i]));
   }

   auto entryNumber = fJoinTable.find(RCombinedJoinFieldValue(joinFieldValues));

   if (entryNumber == fJoinTable.end())
      return nullptr;

   return &(entryNumber->second);
}
