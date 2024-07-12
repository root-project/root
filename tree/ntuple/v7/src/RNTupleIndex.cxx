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
#include <ROOT/RHashValueVisitor.hxx>

ROOT::Experimental::Internal::RNTupleIndex::RNTupleIndex(std::vector<std::unique_ptr<RFieldBase>> &fields,
                                                         NTupleSize_t nEntries)
   : fFields(std::move(fields))
{
   std::vector<RFieldBase::RValue> fieldValues;
   fieldValues.reserve(fields.size());
   for (const auto &field : fFields) {
      fieldValues.emplace_back(field->CreateValue());
   }

   for (std::uint64_t i = 0; i < nEntries; ++i) {
      std::vector<void *> ptrs;
      ptrs.reserve(fieldValues.size());
      for (auto &fieldValue : fieldValues) {
         // TODO(fdegeus): use bulk reading
         fieldValue.Read(i);
         ptrs.push_back(fieldValue.GetPtr<void>().get());
      }
      Add(ptrs, i);
   }
}

void ROOT::Experimental::Internal::RNTupleIndex::Add(const std::vector<void *> &valuePtrs, NTupleSize_t entry)
{
   std::vector<std::size_t> valueHashes;
   valueHashes.reserve(fFields.size());
   for (unsigned i = 0; i < fFields.size(); ++i) {
      auto &field = fFields[i];
      auto valuePtr = valuePtrs[i];
      RHashValueVisitor visitor(valuePtr);
      field->AcceptVisitor(visitor);
      valueHashes.push_back(visitor.GetHash());
   }
   fIndex[RIndexValue(valueHashes)].push_back(entry);
}

std::unique_ptr<ROOT::Experimental::Internal::RNTupleIndex>
ROOT::Experimental::Internal::RNTupleIndex::Create(const std::vector<std::string_view> &fieldNames,
                                                   RPageSource &pageSource)
{
   pageSource.Attach();
   auto desc = pageSource.GetSharedDescriptorGuard();

   std::vector<std::unique_ptr<RFieldBase>> fields;
   fields.reserve(fieldNames.size());

   for (const auto &fieldName : fieldNames) {
      auto fieldId = desc->FindFieldId(fieldName);
      if (fieldId == kInvalidDescriptorId)
         throw RException(R__FAIL("could not find field \"" + std::string(fieldName) + ""));

      const auto &fieldDesc = desc->GetFieldDescriptor(fieldId);
      auto field = fieldDesc.CreateField(desc.GetRef());

      CallConnectPageSourceOnField(*field, pageSource);

      fields.push_back(std::move(field));
   }

   return std::unique_ptr<RNTupleIndex>(new RNTupleIndex(fields, pageSource.GetNEntries()));
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
   std::vector<std::size_t> valueHashes;
   valueHashes.reserve(fFields.size());
   for (unsigned i = 0; i < fFields.size(); ++i) {
      auto &field = fFields[i];
      auto valuePtr = valuePtrs[i];
      RHashValueVisitor visitor(valuePtr);
      field->AcceptVisitor(visitor);
      valueHashes.push_back(visitor.GetHash());
   }
   RIndexValue indexValue(valueHashes);

   if (!fIndex.count(indexValue))
      return nullptr;

   return &fIndex.at(indexValue);
}
