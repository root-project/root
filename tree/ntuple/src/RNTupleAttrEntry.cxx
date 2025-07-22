/// \file RNTupleAttrEntry.cxx
/// \ingroup NTuple ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-05-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleAttrEntry.hxx>
#include <ROOT/RNTupleModel.hxx>

std::size_t ROOT::Experimental::RNTupleAttrEntry::Append()
{
   auto bytesWritten = fMetaEntry->Append();
   bytesWritten += fScopedEntry->Append();
   return bytesWritten;
}

std::size_t ROOT::Experimental::Internal::RNTupleAttrEntryPair::Append()
{
   std::size_t bytesWritten = 0;
   // Write the meta entry values
   bytesWritten += fMetaEntry.fValues[0].Append(); // XXX: hardcoded
   bytesWritten += fMetaEntry.fValues[1].Append(); // XXX: hardcoded

   // Bind the user model's memory to the meta model's subfields
   const auto &userFields =
      ROOT::Internal::GetFieldZeroOfModel(fMetaModel).GetMutableSubfields()[2]->GetMutableSubfields(); // XXX: hardcoded
   assert(userFields.size() == fScopedEntry.fValues.size());
   for (std::size_t i = 0; i < fScopedEntry.fValues.size(); ++i) {
      void *userPtr = fScopedEntry.fValues[i].GetPtr<void>().get();
      auto value = userFields[i]->CreateValue();
      value.BindRawPtr(userPtr);
      bytesWritten += value.Append();
   }
   return bytesWritten;
}

std::unique_ptr<ROOT::REntry> ROOT::Experimental::RNTupleAttrEntry::CreateScopedEntry(ROOT::RNTupleModel &model)
{
   auto scopedEntry = std::unique_ptr<ROOT::REntry>(new ROOT::REntry(model.GetModelId(), model.GetSchemaId()));

   auto &zeroField = ROOT::Internal::GetFieldZeroOfModel(const_cast<ROOT::RNTupleModel &>(model));
   auto subfields = zeroField.GetMutableSubfields();
   RFieldBase *userRootField = subfields[2]; // XXX: hardcoded

   // Add only the user-defined fields to the scoped entry
   for (auto &f : userRootField->GetMutableSubfields()) {
      scopedEntry->AddValue(f->CreateValue());
      // Change the internal name mapping to use the non-qualified field name, so the scoped entry refers to its
      // field as "fieldName" rather than "AttrSetName.fieldName".
      auto handle = scopedEntry->fFieldName2Token.extract(f->GetQualifiedFieldName());
      handle.key() = f->GetFieldName();
      scopedEntry->fFieldName2Token.insert(std::move(handle));
   }

   return scopedEntry;
}

std::pair<std::unique_ptr<ROOT::REntry>, std::unique_ptr<ROOT::REntry>>
ROOT::Experimental::RNTupleAttrEntry::CreateInternalEntries(ROOT::RNTupleModel &model)
{
   auto metaEntry = std::unique_ptr<ROOT::REntry>(new ROOT::REntry(model.GetModelId(), model.GetSchemaId()));

   auto &zeroField = ROOT::Internal::GetFieldZeroOfModel(const_cast<ROOT::RNTupleModel &>(model));
   auto subfields = zeroField.GetMutableSubfields();
   RFieldBase *rangeStartField = subfields[0]; // XXX: hardcoded
   RFieldBase *rangeLenField = subfields[1];   // XXX: hardcoded

   // Add only the range start/len to `metaEntry`
   metaEntry->AddValue(rangeStartField->CreateValue());
   metaEntry->AddValue(rangeLenField->CreateValue());

   auto scopedEntry = CreateScopedEntry(model);

   return {std::move(metaEntry), std::move(scopedEntry)};
}
