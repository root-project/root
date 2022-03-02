/// \file REntry.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RFieldValue.hxx>

#include <algorithm>

ROOT::Experimental::REntry::~REntry()
{
   for (auto idx : fManagedValues) {
      fValues[idx].GetField()->DestroyValue(fValues[idx]);
   }
}

void ROOT::Experimental::REntry::AddValue(const Detail::RFieldValue& value)
{
   fManagedValues.emplace_back(fValues.size());
   fValues.push_back(value);
}

void ROOT::Experimental::REntry::CaptureValue(const Detail::RFieldValue& value)
{
   fValues.push_back(value);
}

void ROOT::Experimental::REntry::CaptureValueUnsafe(std::string_view fieldName, void *where)
{
   for (std::size_t i = 0; i < fValues.size(); ++i) {
      if (fValues[i].GetField()->GetName() != fieldName)
         continue;
      auto itr = std::find(fManagedValues.begin(), fManagedValues.end(), i);
      if (itr != fManagedValues.end()) {
         fValues[i].GetField()->DestroyValue(fValues[i]);
         fManagedValues.erase(itr);
      }
      fValues[i] = fValues[i].GetField()->CaptureValue(where);
      return;
   }
   throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
}
