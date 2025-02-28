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
CastValuePtr(void *valuePtr, std::size_t fieldValueSize)
{
   ROOT::Experimental::Internal::RNTupleJoinTable::NTupleJoinValue_t value;

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

void ROOT::Experimental::Internal::RNTupleJoinTable::EnsureBuilt() const
{
   if (!fIsBuilt)
      throw RException(R__FAIL("join table has not been built yet"));
}

std::unique_ptr<ROOT::Experimental::Internal::RNTupleJoinTable>
ROOT::Experimental::Internal::RNTupleJoinTable::Create(const std::vector<std::string> &fieldNames)
{
   auto joinTable = std::unique_ptr<RNTupleJoinTable>(new RNTupleJoinTable(fieldNames));

   return joinTable;
}

void ROOT::Experimental::Internal::RNTupleJoinTable::Build(RPageSource &pageSource)
{
   if (fIsBuilt)
      return;

   static const std::unordered_set<std::string> allowedTypes = {"std::int8_t",   "std::int16_t", "std::int32_t",
                                                                "std::int64_t",  "std::uint8_t", "std::uint16_t",
                                                                "std::uint32_t", "std::uint64_t"};

   pageSource.Attach();
   auto desc = pageSource.GetSharedDescriptorGuard();

   fJoinFieldValueSizes.reserve(fJoinFieldNames.size());
   std::vector<std::unique_ptr<RFieldBase>> fields;
   std::vector<RFieldBase::RValue> fieldValues;
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

      CallConnectPageSourceOnField(*field, pageSource);

      fieldValues.emplace_back(field->CreateValue());
      fJoinFieldValueSizes.emplace_back(field->GetValueSize());
      fields.emplace_back(std::move(field));
   }

   std::vector<NTupleJoinValue_t> joinFieldValues;
   joinFieldValues.reserve(fJoinFieldNames.size());

   for (unsigned i = 0; i < pageSource.GetNEntries(); ++i) {
      joinFieldValues.clear();
      for (auto &fieldValue : fieldValues) {
         // TODO(fdegeus): use bulk reading
         fieldValue.Read(i);

         auto valuePtr = fieldValue.GetPtr<void>();
         joinFieldValues.push_back(CastValuePtr(valuePtr.get(), fieldValue.GetField().GetValueSize()));
      }
      fJoinTable[RCombinedJoinFieldValue(joinFieldValues)].push_back(i);
   }

   fIsBuilt = true;
}

std::vector<ROOT::NTupleSize_t>
ROOT::Experimental::Internal::RNTupleJoinTable::GetEntryIndexes(const std::vector<void *> &valuePtrs) const
{
   EnsureBuilt();

   if (valuePtrs.size() != fJoinFieldNames.size())
      throw RException(R__FAIL("number of value pointers must match number of join fields"));

   std::vector<NTupleJoinValue_t> joinFieldValues;
   joinFieldValues.reserve(valuePtrs.size());

   for (unsigned i = 0; i < valuePtrs.size(); ++i) {
      joinFieldValues.push_back(CastValuePtr(valuePtrs[i], fJoinFieldValueSizes[i]));
   }

   auto entryIdxs = fJoinTable.find(RCombinedJoinFieldValue(joinFieldValues));

   if (entryIdxs == fJoinTable.end())
      return {};

   return entryIdxs->second;
}
