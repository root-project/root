/// \file RNTupleAttributeEntry.cxx
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

#include <ROOT/RNTupleAttributeEntry.hxx>
#include <ROOT/RNTupleModel.hxx>

std::size_t ROOT::Experimental::RNTupleAttributeEntry::Append()
{
   auto bytesWritten = fEntry->Append();
   bytesWritten += fScopedEntry->Append();
   return bytesWritten;
}

std::unique_ptr<ROOT::REntry> ROOT::Experimental::RNTupleAttributeEntry::CreateScopedEntry(ROOT::RNTupleModel &model)
{
   auto scopedEntry = std::unique_ptr<ROOT::REntry>(new ROOT::REntry(model.GetModelId(), model.GetSchemaId()));

   auto &zeroField = ROOT::Internal::GetFieldZeroOfModel(const_cast<ROOT::RNTupleModel &>(model));
   auto subfields = zeroField.GetMutableSubfields();
   RFieldBase *userRootField = subfields[0];

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
ROOT::Experimental::RNTupleAttributeEntry::CreateInternalEntries(ROOT::RNTupleModel &model)
{
   auto entry = std::unique_ptr<ROOT::REntry>(new ROOT::REntry(model.GetModelId(), model.GetSchemaId()));

   auto &zeroField = ROOT::Internal::GetFieldZeroOfModel(const_cast<ROOT::RNTupleModel &>(model));
   auto subfields = zeroField.GetMutableSubfields();
   RFieldBase *rangeStartField = subfields[1];
   RFieldBase *rangeLenField = subfields[2];

   // Add only the range start/len to `entry`
   entry->AddValue(rangeStartField->CreateValue());
   entry->AddValue(rangeLenField->CreateValue());

   auto scopedEntry = CreateScopedEntry(model);

   return {std::move(entry), std::move(scopedEntry)};
}
