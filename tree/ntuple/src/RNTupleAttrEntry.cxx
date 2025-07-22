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
