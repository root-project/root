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

#include <algorithm>

void ROOT::Experimental::REntry::AddValue(Detail::RFieldBase::RValue &&value)
{
   fValues.emplace_back(std::move(value));
   fValuePtrs.emplace_back(nullptr);
}

void ROOT::Experimental::REntry::BindRaw(std::string_view fieldName, void *where)
{
   for (std::size_t i = 0; i < fValues.size(); ++i) {
      if (fValues[i].GetField().GetName() != fieldName)
         continue;
      fValues[i].BindRaw(where);
      return;
   }
   throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
}

bool ROOT::Experimental::REntry::IsManaged(std::string_view fieldName) const
{
   for (std::size_t i = 0; i < fValues.size(); ++i) {
      if (fValues[i].GetField().GetName() != fieldName)
         continue;
      return static_cast<bool>(fValuePtrs[i]);
   }
   throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
}
